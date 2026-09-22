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


def _softmax(lg, T):
    m = max(x / T for x in lg)
    e = [math.exp(x / T - m) for x in lg]
    s = sum(e)
    return [x / s for x in e]


def _ece(confs, corrects, bins=15):
    """Expected Calibration Error: |accuracy - confidence| averaged over
    equal-width confidence bins, weighted by bin population."""
    if not confs:
        return 0.0
    tot, e = len(confs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (c > lo or b == 0) and c <= hi]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        e += (len(idx) / tot) * abs(acc - conf)
    return e


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
        by_task = {}                       # task -> [correct, n, brier_sum]
        conf_task = {}                     # task -> ([confidences], [corrects])
        conf_all, corr_all = [], []
        for lg, a, r in zip(tlog, tans, test):
            p = _softmax(lg, T)            # calibrated distribution over this row's options
            pred = max(range(len(p)), key=lambda i: p[i])
            ok = int(pred == a)
            brier = sum((p[i] - (1.0 if i == a else 0.0)) ** 2 for i in range(len(p)))
            t = r["task"]
            by_task.setdefault(t, [0, 0, 0.0])
            by_task[t][0] += ok
            by_task[t][1] += 1
            by_task[t][2] += brier
            conf_task.setdefault(t, ([], []))
            conf_task[t][0].append(p[pred])
            conf_task[t][1].append(ok)
            conf_all.append(p[pred])
            corr_all.append(ok)
        acc = {t: c[0] / c[1] for t, c in sorted(by_task.items())}
        brier_by_task = {t: c[2] / c[1] for t, c in sorted(by_task.items())}
        ece_by_task = {t: _ece(conf_task[t][0], conf_task[t][1]) for t in sorted(by_task)}
        n_all = sum(c[1] for c in by_task.values())
        overall = sum(c[0] for c in by_task.values()) / n_all
        brier_all = sum(c[2] for c in by_task.values()) / n_all
        ece_all = _ece(conf_all, corr_all)
        report["datasets"][name] = {"T": T, "n_test": len(test), "skipped_gt26": skipped,
                                    "acc_all": overall, "ece_all": ece_all, "brier_all": brier_all,
                                    "acc_by_task": acc, "ece_by_task": ece_by_task,
                                    "brier_by_task": brier_by_task}
        print(f"[{name}] T={T:.2f} n={len(test)} skipped={skipped} "
              f"ALL={overall:.3f} ECE={ece_all:.3f} Brier={brier_all:.3f}", flush=True)
        for t, a in acc.items():
            print(f"    {t:40s} {a:.3f}  ece={ece_by_task[t]:.3f} brier={brier_by_task[t]:.3f}", flush=True)
        json.dump(report, open(args.out + ".partial", "w"), indent=1)

    json.dump(report, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
