#!/usr/bin/env python3
"""Score a HF causal model (optionally + LoRA) as a letter-logprob decision
scorer, matching harness/baseline_logprob exactly: prompt ends "\n\nAnswer:",
read the logprob of each option's ' LETTER' first token, softmax over options,
fit temperature on val, report per-task accuracy on test.

Runs in the ROCm container so we can score our QLoRA 7-8B the same way the 35B
baseline was scored - same prompt, same token, no chat template.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(row):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(row["options"]))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (row["state"], row["question"], opts))


def fit_temperature(logit_sets, answers):
    best_t, best_nll = 1.0, float("inf")
    t = 0.05
    while t <= 5.0:
        nll = 0.0
        for lg, a in zip(logit_sets, answers):
            m = max(x / t for x in lg)
            e = [math.exp(x / t - m) for x in lg]
            s = sum(e)
            nll -= math.log(max(e[a] / s, 1e-12))
        if nll < best_nll:
            best_nll, best_t = nll, t
        t += 0.05
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--datasets", nargs="+", default=["sni", "reflex", "bfcl"])
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0, help="cap rows per split (debug)")
    ap.add_argument("--out", default=str(ROOT / "results" / "jevlike7b_eval.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.truncation_side = "left"  # keep Options/Answer: tail, matching training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # first-token id for each ' LETTER'
    letter_id = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0]
                 for L in LETTERS}

    @torch.no_grad()
    def score(row):
        ids = tok(build_prompt(row), add_special_tokens=True,
                  return_tensors="pt", truncation=True, max_length=args.max_len)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        logits = model(**ids).logits[0, -1]  # next-token logits
        lp = torch.log_softmax(logits.float(), dim=-1)
        return [lp[letter_id[LETTERS[i]]].item() for i in range(len(row["options"]))]

    report = {"base": args.base, "adapter": args.adapter, "datasets": {}}
    for name in args.datasets:
        d = ROOT / "data" / name
        val = json.load(open(d / "val.json"))
        test = json.load(open(d / "test.json"))
        # letters only go A-Z; skip rows with >26 options (matches baseline_logprob)
        nval, ntest = len(val), len(test)
        val = [r for r in val if len(r["options"]) <= len(LETTERS)]
        test = [r for r in test if len(r["options"]) <= len(LETTERS)]
        skipped = (nval - len(val)) + (ntest - len(test))
        if args.limit:
            val, test = val[: args.limit], test[: args.limit]
        vlog = [score(r) for r in val]
        vans = [r["answer_index"] for r in val]
        T = fit_temperature(vlog, vans)
        tlog = [score(r) for r in test]
        tans = [r["answer_index"] for r in test]
        by_task = {}
        for lg, a, r in zip(tlog, tans, test):
            pred = max(range(len(lg)), key=lambda i: lg[i])
            by_task.setdefault(r["task"], [0, 0])
            by_task[r["task"]][1] += 1
            by_task[r["task"]][0] += int(pred == a)
        acc = {t: c[0] / c[1] for t, c in sorted(by_task.items())}
        overall = sum(c[0] for c in by_task.values()) / sum(c[1] for c in by_task.values())
        report["datasets"][name] = {"T": T, "n_test": len(test), "skipped_gt26": skipped,
                                    "acc_all": overall, "acc_by_task": acc}
        print(f"[{name}] T={T:.2f} n={len(test)} skipped={skipped} ALL={overall:.3f}", flush=True)
        for t, a in acc.items():
            print(f"    {t:40s} {a:.3f}", flush=True)
        json.dump(report, open(args.out + ".partial", "w"), indent=1)

    json.dump(report, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
