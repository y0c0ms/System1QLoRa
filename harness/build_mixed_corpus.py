#!/usr/bin/env python3
"""Assemble the mixed decision corpus for QLoRA SFT of the open decision model.

WHY MIXED. The reflex reeval proved a single-domain scorer does not transfer:
an SNI-trained scorer scored 0.255 on reflex (below chance on math). So we SFT
on every in-domain distribution we want it good at, jointly.

FORMAT. The target is exactly what harness/baseline_logprob.py scores at eval:
the prompt ends in "\n\nAnswer:" and the model must emit the option LETTER as
its first token (arriving as ' A'). We therefore SFT prompt -> ' {LETTER}', so
the training objective is the eval metric. Prompt tokens are masked in the
loss (completion-only); only the letter carries gradient.

V2 AUGMENTATION - absence recognition. The measured biggest gap is the model
failing to pick the "no positive relation" option: NLI *neutral* scored 0.14
(below chance), and tool-calling *None of the above* was the one hard-case miss.
Both are the same skill - recognising that none of the positive options fit. We
inject two augmentations (off by default; opt in with the rate flags):
  * ABSTAIN variant: drop the correct option, append "None of the above" and make
    it the answer. Teaches "no positive option fits -> abstain."
  * NOTA-DISTRACTOR variant: append "None of the above" as an extra WRONG option,
    answer unchanged. Teaches that the absence option is NOT a default.
"None of the above" matches BFCL's held-out irrelevance sentinel verbatim, so the
skill is measured zero-shot there. Option order is shuffled in both variants so
the model learns the concept, not a position.

NEVER include BFCL: build_bfcl.py marks it the canonical benchmark; training on
it destroys the ability to report a tool-calling number. It stays zero-shot.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SOURCES = ["sni", "reflex"]  # train splits only; BFCL is held out by policy
NOTA = "None of the above"


def build_prompt(state, question, options):
    """Byte-identical to baseline_logprob.build_prompt."""
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(options))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (state, question, opts))


def make_variant(options, answer_index, drop_correct, rng):
    """Return (options, answer_index) with a NOTA option added. If drop_correct,
    the true option is removed and NOTA becomes correct (abstain); otherwise NOTA
    is an extra wrong option. Order is shuffled."""
    items = [(o, i == answer_index) for i, o in enumerate(options)]
    if drop_correct:
        items = [(o, c) for (o, c) in items if not c]
    items.append((NOTA, drop_correct))
    rng.shuffle(items)
    opts = [t for t, _ in items]
    ai = next(i for i, (_, c) in enumerate(items) if c)
    return opts, ai


# "No positive relation" answer labels (the class the model under-predicts:
# NLI neutral, unanswerable, no-relation). Upweighting these rows increases
# exposure to the absence class across every task, targeting the NLI-neutral gap.
ABSENCE_LABELS = {
    "neutral", "not entailment", "not_entailment", "no entailment", "unrelated",
    "no relation", "no-relation", "cannot be determined", "undetermined",
    "not enough information", "unanswerable", "no answer", "irrelevant",
    "none of the above", "none", "not applicable",
}


def is_absence(text):
    s = str(text).strip().lower()
    return (s in ABSENCE_LABELS or s.startswith("neutral") or s.startswith("unrelated")
            or "cannot be determined" in s or "unanswerable" in s
            or "not entail" in s or "no relation" in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "mixed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-task", type=int, default=0,
                    help="cap rows per task (0 = no cap)")
    ap.add_argument("--abstain-rate", type=float, default=0.0,
                    help="fraction of >=3-option rows to also emit as an abstain variant")
    ap.add_argument("--nota-distractor-rate", type=float, default=0.0,
                    help="fraction of rows to also emit with NOTA as a wrong extra option")
    ap.add_argument("--upweight-absence", type=int, default=1,
                    help="emit rows whose answer is an absence label this many times")
    ap.add_argument("--sources", default=",".join(SOURCES),
                    help="comma-separated train sources under data/<src>/train.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out_rows, stats = [], {}
    n_abstain = n_nota = n_upweight = 0

    def emit(src, task, state, question, options, ai):
        out_rows.append({"source": src, "task": task,
                         "prompt": build_prompt(state, question, options),
                         "completion": " " + LETTERS[ai]})

    for src in args.sources.split(","):
        rows = json.load(open(ROOT / "data" / src / "train.json"))
        by_task = {}
        for r in rows:
            k, ai = len(r["options"]), r["answer_index"]
            if not (0 <= ai < k) or k < 2 or k > len(LETTERS):
                continue  # skip malformed / unrepresentable rows
            by_task.setdefault(r["task"], []).append(r)
        for task, trows in by_task.items():
            if args.max_per_task:
                rng.shuffle(trows)
                trows = trows[: args.max_per_task]
            for r in trows:
                opts, ai = r["options"], r["answer_index"]
                reps = args.upweight_absence if is_absence(opts[ai]) else 1
                for _ in range(reps):
                    emit(src, task, r["state"], r["question"], opts, ai)
                n_upweight += reps - 1
                if len(opts) >= 3 and rng.random() < args.abstain_rate:
                    vo, va = make_variant(opts, ai, True, rng)
                    emit(src, task, r["state"], r["question"], vo, va)
                    n_abstain += 1
                if len(opts) + 1 <= len(LETTERS) and rng.random() < args.nota_distractor_rate:
                    vo, va = make_variant(opts, ai, False, rng)
                    emit(src, task, r["state"], r["question"], vo, va)
                    n_nota += 1
            stats[f"{src}:{task}"] = len(trows)

    rng.shuffle(out_rows)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "train.jsonl"
    with open(fp, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "n_rows": len(out_rows),
        "sources": SOURCES,
        "per_task": stats,
        "letter_dist": _letter_hist(out_rows),
        "augmentation": {"abstain_rate": args.abstain_rate, "abstain_rows": n_abstain,
                         "nota_distractor_rate": args.nota_distractor_rate,
                         "nota_distractor_rows": n_nota, "nota_text": NOTA,
                         "upweight_absence": args.upweight_absence, "upweight_extra_rows": n_upweight},
        "prompt_format": "State/Question/Options/Answer: -> ' LETTER' (matches baseline_logprob)",
        "seed": args.seed,
        "max_per_task": args.max_per_task,
    }
    json.dump(manifest, open(outdir / "manifest.json", "w"), indent=1)
    print(f"wrote {len(out_rows)} rows -> {fp}")
    print(f"  base rows + {n_abstain} abstain + {n_nota} nota-distractor + {n_upweight} absence-upweight")
    print(json.dumps(manifest["letter_dist"], indent=1))
    print(f"tasks: {len(stats)}")


def _letter_hist(rows):
    h = {}
    for r in rows:
        c = r["completion"].strip()
        h[c] = h.get(c, 0) + 1
    return dict(sorted(h.items()))


if __name__ == "__main__":
    main()
