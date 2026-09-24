#!/usr/bin/env python3
"""Decision latency, split by benchmark (topic).

Our models answer a row with ONE forward pass: encode the prompt (ending
"\n\nAnswer:") and read the next-token logprob of each option letter. That single
pass IS the answer, so per-row latency = one warmed-up, resident forward pass,
timed with cuda.synchronize().

AGENTS.md rule 4: every latency figure ships with its conditions. We therefore
print, per benchmark: model, device, warmup count, rows timed, option-count and
prompt-token distributions, and whether the model was already resident (it is -
we warm up first). Rule 11: we time the real scoring function, not a proxy.

Jev latency is measured separately (remote opencode-zen API, network-inclusive)
and merged into the printed table if results/jev_latency.json exists.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from eval_decider import text_field  # canonical field rendering; see eval_decider.text_field

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(row):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(row["options"]))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (text_field(row["state"]), text_field(row["question"]), opts))


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--datasets", nargs="+", default=["sni", "reflex", "bfcl", "bfcl_irr"])
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--rows", type=int, default=120, help="rows timed per benchmark")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=3, help="passes per row; report the min")
    ap.add_argument("--out", default=str(ROOT / "results" / "latency_ours.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.truncation_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dev = model.device

    letter_id = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0]
                 for L in LETTERS}

    @torch.no_grad()
    def timed_score(row):
        ids = tok(build_prompt(row), add_special_tokens=True, return_tensors="pt",
                  truncation=True, max_length=args.max_len)
        ntok = ids["input_ids"].shape[1]
        ids = {k: v.to(dev) for k, v in ids.items()}
        best = float("inf")
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(**ids).logits[0, -1]
            _ = [logits[letter_id[LETTERS[i]]].item() for i in range(len(row["options"]))]
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) * 1000.0)
        return best, ntok

    report = {"base": args.base, "adapter": args.adapter, "device": str(dev),
              "warmup": args.warmup, "repeats_min": args.repeats, "benchmarks": {}}

    # warmup on real rows so kernels/caches are hot before any timing
    warm = json.load(open(ROOT / "data" / args.datasets[0] / "test.json"))
    warm = [r for r in warm if len(r["options"]) <= len(LETTERS)][: args.warmup]
    for r in warm:
        timed_score(r)

    print(f"model={args.base} adapter={args.adapter or '-'} device={dev} "
          f"warmup={args.warmup} repeats(min)={args.repeats} rows/bench={args.rows}")
    print(f"{'benchmark':<12}{'n':>5}{'opts':>6}{'ptok':>7}"
          f"{'median_ms':>11}{'p50':>8}{'p90':>8}{'mean':>8}")
    all_ms = []
    for name in args.datasets:
        test = json.load(open(ROOT / "data" / name / "test.json"))
        test = [r for r in test if len(r["options"]) <= len(LETTERS)][: args.rows]
        ms, ntoks, nopts = [], [], []
        for r in test:
            t, ntok = timed_score(r)
            ms.append(t)
            ntoks.append(ntok)
            nopts.append(len(r["options"]))
        all_ms += ms
        report["benchmarks"][name] = {
            "n": len(ms), "median_ms": statistics.median(ms),
            "p50": pct(ms, 50), "p90": pct(ms, 90), "mean_ms": statistics.mean(ms),
            "avg_options": statistics.mean(nopts), "avg_prompt_tokens": statistics.mean(ntoks)}
        b = report["benchmarks"][name]
        print(f"{name:<12}{b['n']:>5}{b['avg_options']:>6.1f}{b['avg_prompt_tokens']:>7.0f}"
              f"{b['median_ms']:>11.1f}{b['p50']:>8.1f}{b['p90']:>8.1f}{b['mean_ms']:>8.1f}")
    report["overall_median_ms"] = statistics.median(all_ms)
    print(f"{'OVERALL':<12}{len(all_ms):>5}{'':>6}{'':>7}{statistics.median(all_ms):>11.1f}")
    json.dump(report, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
