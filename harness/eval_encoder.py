#!/usr/bin/env python3
"""Score an encoder decision specialist into the decoder evaluator's artifact format.

Same contract as eval_decider.py: results/evals/<tag>/<name>.<split>.<order>.jsonl with
i/task/tier/k/answer/trunc/logits, plus summary.<splits>.json. analyze_decider.py then
compares encoder and decoder with identical code, identical bootstrap CIs, identical
tier and permutation handling - the comparison is apples-to-apples by construction.

One forward pass per (context, option) pair; the decision is the argmax over a row's
option scores. `trunc` is set when the context had to be cut from the left to fit
max_len, i.e. when the model did not see the whole decision.

Usage:
  python harness/eval_encoder.py --tag enc_22m --model results/enc_22m \
      --datasets jevbench sni reflex --splits test --orders orig rev
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_decider import LETTERS, build_prompt, reverse  # noqa: E402
from train_encoder import build_context, encode_pair, collate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def score_rows(tok, model, rows, max_len, batch_pairs):
    """Return per-row option logits and a per-row truncation flag."""
    todo, meta = [], []
    for i, r in enumerate(rows):
        ctx_ids = tok(build_context(r), add_special_tokens=False)["input_ids"]
        for j, o in enumerate(r["options"]):
            opt_ids = tok(o, add_special_tokens=False)["input_ids"]
            ids = encode_pair(tok, ctx_ids, opt_ids, max_len)
            # context was cut if it did not fit alongside the option and the specials
            todo.append(ids)
            meta.append((i, j, len(ctx_ids) + min(len(opt_ids), 64) + 3 > max_len))
    order = sorted(range(len(todo)), key=lambda x: len(todo[x]))
    dev = next(model.parameters()).device
    out = [None] * len(todo)
    for s in range(0, len(order), batch_pairs):
        chunk = order[s:s + batch_pairs]
        ids, am = collate([todo[c] for c in chunk], tok.pad_token_id or 0)
        lg = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits.squeeze(-1).tolist()
        for c, v in zip(chunk, lg):
            out[c] = v
    logits = [[None] * len(r["options"]) for r in rows]
    trunc = [False] * len(rows)
    for (i, j, t), v in zip(meta, out):
        logits[i][j] = v
        trunc[i] = trunc[i] or t
    return logits, trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True, help="trained encoder dir (or HF id)")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--orders", nargs="+", default=["orig"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--long-max-len", type=int, default=1024, help="used for jevbench")
    ap.add_argument("--batch-pairs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--vram-cap-gb", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if dev == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb / total), 0)
    tok = AutoTokenizer.from_pretrained(args.model)
    # num_labels=1 must be forced: a bare base checkpoint has no num_labels, so the head
    # defaults to 2 classes and the per-option scores come out nested instead of flat.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1).to(dev).eval()
    cap = getattr(model.config, "max_position_embeddings", None)
    outdir = ROOT / "results" / "evals" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in args.datasets:
        max_len = args.long_max_len if name == "jevbench" else args.max_len
        if cap:
            max_len = min(max_len, cap)
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
                logits, trunc = score_rows(tok, model, src, max_len, args.batch_pairs)
                fn = outdir / f"{name}.{split}.{order}.jsonl"
                correct = 0
                with open(fn, "w") as f:
                    for n, (r, lg, tr) in enumerate(zip(rows, logits, trunc)):
                        if order == "rev":
                            lg = list(reversed(lg))
                        pred = max(range(len(lg)), key=lg.__getitem__)
                        correct += pred == r["answer_index"]
                        f.write(json.dumps({"i": n, "task": r.get("task"), "tier": r.get("tier"),
                                            "k": len(lg), "answer": r["answer_index"],
                                            "trunc": bool(tr), "logits": lg}) + "\n")
                acc = correct / max(1, len(rows))
                summary[f"{name}.{split}.{order}"] = {"n": len(rows), "acc": acc,
                                                      "truncated": sum(trunc),
                                                      "sec": round(time.time() - t0, 1)}
                print(f"[{name}.{split}.{order}] n={len(rows)} acc={acc:.4f} "
                      f"trunc={sum(trunc)} {time.time() - t0:.0f}s", flush=True)

    json.dump({"args": vars(args), "summary": summary},
              open(outdir / f"summary.{'-'.join(args.splits)}.json", "w"), indent=1)
    print("SAVED", outdir, flush=True)


if __name__ == "__main__":
    main()
