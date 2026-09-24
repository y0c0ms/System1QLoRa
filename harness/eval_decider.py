#!/usr/bin/env python3
"""Batched decision scorer that SAVES per-row option logits.

Scoring and analysis are separated: this script (GPU) writes, for every row, the
raw logits of the K option letters at the answer position; analyze_decider.py
(CPU) derives accuracy, a global temperature, ECE/Brier, order sensitivity,
per-tier JevBench accuracy and paired-bootstrap CIs from those files. Softmax
over option-letter logits equals softmax over full-vocab log-probs restricted to
the options (log_softmax only subtracts a constant), so raw logits suffice.

Model sources: --adapter (PEFT dir on --base), --model (full fine-tuned dir), or
neither (bare base). --precision nf4 reproduces how QLoRA arms were trained.

Order variants: --orders orig rev scores each row as given and with the option
list reversed (answer remapped); rev logits are stored mapped BACK to the
original option order so rows can be compared/averaged directly.

Padding: RIGHT padding and the logits of each row's last real token (gathered
through an index-tensor logits_to_keep). Right padding never produces a query
whose every key is masked, and recurrent layers see the real sequence first.

--check-invariance N scores N rows one at a time and as one padded batch and
exits non-zero if they disagree: batched numbers are only trusted after this.

Safety: --vram-cap-gb sets a hard per-process VRAM cap so a runaway eval fails
itself instead of starving the desktop compositor.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def text_field(v):
    """Render a text-or-structured field the way the corpus is written: JSON.

    Most rows carry `state` and `question` as strings (the states are JSON text), but some
    carry them as dicts: 35 of the 231 JevBench items have a dict state, and kevsuite has
    58/68 dict questions in val/test. An f-string renders those as a Python repr -
    {'policy': '...'} with single quotes - a format the corpus otherwise never contains.
    The cost is measurable: p3_m1 scores 0.571 on the dict-state JevBench rows against
    0.643 on the rest, and 0.574 on the dict-question kevsuite test rows against 0.787.
    Every scorer and trainer must use this so the prompt format cannot drift.
    """
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


# kept as the name the rest of the harness already imports
state_text = text_field


def build_prompt(state, question, options):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(options))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (text_field(state), text_field(question), opts))


def load(args):
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb / total), 0)
    src = args.model or args.base
    tok = AutoTokenizer.from_pretrained(src)
    tok.truncation_side = "left"
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map={"": 0}, attn_implementation="sdpa")
    if args.precision == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    lid = []
    for L in LETTERS:
        t = tok(" " + L, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise SystemExit(f"' {L}' not a single token: {t}")
        lid.append(t[0])
    return tok, model, torch.tensor(lid)


@torch.no_grad()
def score_batch(tok, model, lid, rows, max_len):
    prompts = [build_prompt(r["state"], r["question"], r["options"]) for r in rows]
    enc = tok(prompts, return_tensors="pt", truncation=True, max_length=max_len,
              padding=True, add_special_tokens=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    last = enc["attention_mask"].sum(1) - 1
    uniq = torch.unique(last)
    out = model(**enc, logits_to_keep=uniq).logits                    # [B, U, V]
    logits = out[torch.arange(out.shape[0], device=out.device),
                 torch.searchsorted(uniq, last)].float()               # [B, V]
    L = logits[:, lid.to(logits.device)].cpu()
    return [L[i, :len(r["options"])].tolist() for i, r in enumerate(rows)], enc["input_ids"].shape[1]


def reverse(row):
    r = dict(row)
    r["options"] = list(reversed(row["options"]))
    r["answer_index"] = len(row["options"]) - 1 - row["answer_index"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--model", default="", help="full fine-tuned model dir")
    ap.add_argument("--precision", choices=["nf4", "bf16"], default="bf16")
    ap.add_argument("--tag", required=True, help="output dir name under results/evals/")
    ap.add_argument("--datasets", nargs="+", default=["sni", "reflex", "bfcl", "bfcl_irr", "kevsuite"])
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--orders", nargs="+", default=["orig"], choices=["orig", "rev"])
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--long-max-len", type=int, default=8192, help="used for jevbench")
    ap.add_argument("--tok-budget", type=int, default=16384, help="padded tokens per batch")
    ap.add_argument("--vram-cap-gb", type=float, default=8.0)
    ap.add_argument("--long-tok-budget", type=int, default=8192, help="token budget for jevbench (long rows)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-invariance", type=int, default=0)
    args = ap.parse_args()

    tok, model, lid = load(args)
    outdir = ROOT / "results" / "evals" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    if args.check_invariance:
        rows = json.load(open(ROOT / "data" / args.datasets[0] / f"{args.splits[0]}.json"))
        rows = [r for r in rows if 2 <= len(r["options"]) <= 26][: args.check_invariance]
        single = [score_batch(tok, model, lid, [r], args.max_len)[0][0] for r in rows]
        batched, _ = score_batch(tok, model, lid, rows, args.max_len)
        diff = max(abs(a - b) for s, bb in zip(single, batched) for a, b in zip(s, bb))

        def probs(v):
            m = max(v)
            e = [math.exp(x - m) for x in v]
            z = sum(e)
            return [x / z for x in e]
        # What decisions and calibration see: option probabilities (T=1) and argmax.
        # bf16 GEMM tilings change with batch shape, so logits drift through depth
        # (observed up to 0.5 at |logit|~20-100); a padding leak or a wrong gather
        # index instead shows up as large probability shifts and decisive flips.
        dp = max(abs(p - q) for s, b in zip(single, batched) for p, q in zip(probs(s), probs(b)))

        def margin(v):
            t = sorted(v, reverse=True)
            return t[0] - t[1]
        # a flip only counts when the single-row decision was not a near-tie
        flips = sum(max(range(len(s)), key=s.__getitem__) != max(range(len(b)), key=b.__getitem__)
                    and margin(s) > 0.2 for s, b in zip(single, batched))
        print(f"INVARIANCE max|single-batched| logit diff = {diff:.4f}, max option-prob diff = {dp:.4f}, "
              f"decisive argmax flips = {flips}/{len(rows)}")
        sys.exit(0 if (dp <= 0.10 and flips == 0) else 3)

    summary = {}
    for name in args.datasets:
        max_len = args.long_max_len if name == "jevbench" else args.max_len
        budget = args.long_tok_budget if name == "jevbench" else args.tok_budget
        for split in args.splits:
            fp = ROOT / "data" / name / f"{split}.json"
            if not fp.exists():
                continue
            rows = [r for r in json.load(open(fp)) if 2 <= len(r["options"]) <= 26
                    and 0 <= r["answer_index"] < len(r["options"])]
            if args.limit:
                rows = rows[: args.limit]
            for order in args.orders:
                t0 = time.time()
                src = [reverse(r) for r in rows] if order == "rev" else rows
                lens = [len(tok(build_prompt(r["state"], r["question"], r["options"]),
                                add_special_tokens=True)["input_ids"]) for r in src]
                idx = sorted(range(len(src)), key=lambda i: lens[i])
                out = [None] * len(src)
                i = 0
                while i < len(idx):
                    j = i + 1
                    while j < len(idx) and (j - i + 1) * min(lens[idx[j]], max_len) <= budget:
                        j += 1
                    chunk = idx[i:j]
                    got, _ = score_batch(tok, model, lid, [src[c] for c in chunk], max_len)
                    for c, g in zip(chunk, got):
                        out[c] = g
                    i = j
                fn = outdir / f"{name}.{split}.{order}.jsonl"
                correct = 0
                with open(fn, "w") as f:
                    for n, (r, lg) in enumerate(zip(rows, out)):
                        if order == "rev":
                            lg = list(reversed(lg))          # map back to original option order
                        pred = max(range(len(lg)), key=lg.__getitem__)
                        correct += pred == r["answer_index"]
                        f.write(json.dumps({"i": n, "task": r.get("task"), "tier": r.get("tier"),
                                            "k": len(lg), "answer": r["answer_index"],
                                            "trunc": lens[n] > max_len, "logits": lg}) + "\n")
                acc = correct / max(1, len(rows))
                ntr = sum(l > max_len for l in lens)
                summary[f"{name}.{split}.{order}"] = {"n": len(rows), "acc": acc, "truncated": ntr,
                                                      "sec": round(time.time() - t0, 1)}
                print(f"[{name}.{split}.{order}] n={len(rows)} acc={acc:.4f} trunc={ntr} "
                      f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"args": vars(args), "summary": summary},
              open(outdir / f"summary.{'-'.join(args.splits)}.json", "w"), indent=1)
    print("SAVED", outdir)


if __name__ == "__main__":
    main()
