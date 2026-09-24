#!/usr/bin/env python3
"""Score kev (jaredpalmer/kev-*) on our benchmarks with kev's OWN loader,
pointer head and fitted temperature, so it sits on the same board as our
letter-logprob models and Jev. Same metrics as hf_score/pointer_score:
per-task accuracy + ECE + Brier.

kev enforces a training context (max_state=384, max_branch=1024) and REJECTS
longer records (ContextOverflow) instead of truncating. That rejection is part
of kev's contract, so we count rejects as `skipped` and report coverage +
accuracy over the rows kev actually accepts.
"""

import argparse
import json
from pathlib import Path

import torch  # noqa: F401  (ensures ROCm torch is imported before kev)
from kev.checkpoint import LoadOptions
from kev.predictors import LocalPredictor

try:
    from kev.model import ContextOverflow
except Exception:  # pragma: no cover
    class ContextOverflow(Exception):
        pass

ROOT = Path(__file__).resolve().parents[1]


def _ece(confs, corrects, bins=15):
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


def to_record(row):
    """Our {state, question, options, answer_index} -> kev labelled record."""
    crit, keys = {}, []
    for o in row["options"]:
        k = str(o)
        while k in crit:
            k += " \u200b"  # dedup collisions with a zero-width space
        crit[k] = None      # option text is the key; no separate description
        keys.append(k)
    rec = {"state": row["state"],
           "questions": {"q": {"type": "choice", "instructions": row["question"],
                               "criteria": crit, "label": keys[row["answer_index"]],
                               "src": "bench"}}}
    return rec, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.6b")
    ap.add_argument("--datasets", nargs="+", default=["sni", "reflex", "bfcl", "bfcl_irr", "oadk"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "kev06b_eval.json"))
    args = ap.parse_args()

    pred = LocalPredictor(args.run, args.device, LoadOptions())  # kev's fitted temperature
    print(f"loaded kev {args.run} on {args.device}", flush=True)

    report = {"run": args.run, "readout": "kev_pointer_head", "datasets": {}}
    for name in args.datasets:
        test = json.load(open(ROOT / "data" / name / "test.json"))
        if args.limit:
            test = test[: args.limit]
        by_task, conf_all, corr_all, skipped = {}, [], [], 0
        for r in test:
            rec, keys = to_record(r)
            try:
                res = pred(rec)
            except ContextOverflow:
                skipped += 1
                continue
            except Exception as e:
                if "overflow" in str(e).lower() or "context" in str(e).lower():
                    skipped += 1
                    continue
                raise
            pr = res["probabilities"]["q"]
            p = [pr.get(k, 0.0) for k in keys]
            s = sum(p) or 1.0
            p = [x / s for x in p]
            ans = r["answer_index"]
            pred_i = max(range(len(p)), key=lambda i: p[i])
            ok = int(pred_i == ans)
            brier = sum((p[i] - (1.0 if i == ans else 0.0)) ** 2 for i in range(len(p)))
            t = r["task"]
            by_task.setdefault(t, [0, 0, 0.0])
            by_task[t][0] += ok
            by_task[t][1] += 1
            by_task[t][2] += brier
            conf_all.append(p[pred_i])
            corr_all.append(ok)
        n = sum(c[1] for c in by_task.values())
        if n == 0:
            print(f"[{name}] all {skipped} rows exceeded kev context - no score", flush=True)
            report["datasets"][name] = {"n_scored": 0, "skipped_context": skipped}
            continue
        acc = {t: c[0] / c[1] for t, c in sorted(by_task.items())}
        overall = sum(c[0] for c in by_task.values()) / n
        brier_all = sum(c[2] for c in by_task.values()) / n
        ece_all = _ece(conf_all, corr_all)
        report["datasets"][name] = {"n_scored": n, "skipped_context": skipped,
                                    "coverage": n / (n + skipped),
                                    "acc_all": overall, "ece_all": ece_all, "brier_all": brier_all,
                                    "acc_by_task": acc}
        print(f"[{name}] n={n} skipped_ctx={skipped} cov={n / (n + skipped):.2f} "
              f"ALL={overall:.3f} ECE={ece_all:.3f} Brier={brier_all:.3f}", flush=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
