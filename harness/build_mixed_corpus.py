#!/usr/bin/env python3
"""Assemble the mixed decision corpus for QLoRA SFT of the 7-8B "our jevlike".

WHY MIXED. The reflex reeval proved a single-domain scorer does not transfer:
an SNI-trained scorer scored 0.255 on reflex (below chance on math). So the
7-8B is SFT'd on every in-domain distribution we want it good at, jointly.

FORMAT. The target is exactly what harness/baseline_logprob.py scores at eval:
the prompt ends in "\n\nAnswer:" and the model must emit the option LETTER as
its first token (arriving as ' A'). We therefore SFT prompt -> ' {LETTER}', so
the training objective is the eval metric. Prompt tokens are masked in the
loss (completion-only); only the letter carries gradient.

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


def build_prompt(row):
    """Byte-identical to baseline_logprob.build_prompt."""
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(row["options"]))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (row["state"], row["question"], opts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "mixed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-task", type=int, default=0,
                    help="cap rows per task (0 = no cap)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out_rows, stats = [], {}
    for src in SOURCES:
        rows = json.load(open(ROOT / "data" / src / "train.json"))
        by_task = {}
        for r in rows:
            k = len(r["options"])
            ai = r["answer_index"]
            if not (0 <= ai < k) or k < 2 or k > len(LETTERS):
                continue  # skip malformed / unrepresentable rows
            by_task.setdefault(r["task"], []).append(r)
        for task, trows in by_task.items():
            if args.max_per_task:
                rng.shuffle(trows)
                trows = trows[: args.max_per_task]
            for r in trows:
                out_rows.append({
                    "source": src,
                    "task": r["task"],
                    "prompt": build_prompt(r),
                    "completion": " " + LETTERS[r["answer_index"]],
                })
            stats[f"{src}:{task}"] = len(trows) if args.max_per_task else len(trows)

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
        "prompt_format": "State/Question/Options/Answer: -> ' LETTER' (matches baseline_logprob)",
        "seed": args.seed,
        "max_per_task": args.max_per_task,
    }
    json.dump(manifest, open(outdir / "manifest.json", "w"), indent=1)
    print(f"wrote {len(out_rows)} rows -> {fp}")
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
