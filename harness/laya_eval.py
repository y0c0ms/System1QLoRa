#!/usr/bin/env python3
"""Evaluate the open Laya System-1 engine on OUR benchmarks.

Laya (github.com/NandhaKishorM/laya, Apache-2.0): ModernBERT-large backbone +
RLCD-trained scoring head, choice/score/noul over 100+ languages, claims 0.766
on the typed-decisions benchmark (the same corpus this repo measures). Its
published held-out generality is UNMEASURED - this harness runs it with zero
extra training on our SNI predicates and BFCL tool rows, with our protocol:
temperature fitted on each benchmark's val, applied to its tests, plus
per-decision latency.

Mapping: state = row state, one 'choice' question with criteria keyed "0..k-1"
so the returned probabilities align with our option order; answer_index and
options are untouched.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LAYA_DIR = ROOT / "upstream/laya"
if str(LAYA_DIR) not in sys.path:
    sys.path.insert(1, str(LAYA_DIR))
from harness.jevlike_cal import softmax  # noqa: E402
from upstream.system_one import fit_temperature  # noqa: E402


def logit(p):
    p = float(p)
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(np.log(p))


def ece_list(probs, answers, n_bins=10):
    conf = np.array([float(p.max()) for p in probs])
    correct = np.array([int(p.argmax() == a) for p, a in zip(probs, answers)])
    out = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sel = (conf > lo) & (conf <= hi)
        if sel.sum():
            out += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(out)


def brier_list(probs, answers):
    tot = 0.0
    for p, a in zip(probs, answers):
        y = np.zeros(len(p))
        y[int(a)] = 1.0
        tot += float(((np.asarray(p) - y) ** 2).sum())
    return tot / max(len(probs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out", default="results/laya_eval.json")
    ap.add_argument("--benches", default="corpus,sni,bfcl")
    args = ap.parse_args()

    from laya.agent import load

    agent = load(args.checkpoint, device=args.device, subfolder=args.subfolder)
    print("loaded", args.checkpoint, args.subfolder or "(root)", flush=True)

    def run_split(name, rows, t=None):
        probs, answers, times = [], [], []
        for i, r in enumerate(rows):
            if args.max_rows and i >= args.max_rows:
                break
            if len(r["options"]) < 2:  # Laya's topk(2) needs >=2 options
                continue
            criteria = {str(j): o for j, o in enumerate(r["options"])}
            qs = {"q": {"type": "choice", "instructions": r["question"],
                        "criteria": criteria}}
            t0 = time.time()
            res = agent.system_one(r["state"], qs)
            times.append(time.time() - t0)
            p = res["answers"]["q"].get("probabilities")
            if p is None:
                p = res["answers"]["q"].get("probs")
            if isinstance(p, dict):  # keyed by criteria key "0".."k-1"
                p = [p.get(str(j), 0.0) for j in range(len(r["options"]))]
            if not p:
                raise SystemExit(f"no probabilities in response: {res}")
            probs.append(np.asarray(p, dtype=float))
            answers.append(int(r["answer_index"]))
            if (i + 1) % 500 == 0:
                print(f"  {name} {i + 1}/{min(len(rows), args.max_rows) or len(rows)}",
                      flush=True)
        acc = float(np.mean([int(p.argmax() == a) for p, a in zip(probs, answers)]))
        return {"n": len(answers), "acc": acc, "ece": ece_list(probs, answers),
                "brier": brier_list(probs, answers), "raw_probs": probs,
                "answers": answers, "ms_per_row": round(1000 * np.mean(times), 2)}

    results = {}
    for bench in (args.benches.split(",")):
        base = ROOT / "data" if bench == "corpus" else ROOT / f"data/{bench}"
        if not (base / "val.json").exists():
            continue
        val = json.loads((base / "val.json").read_text())
        if bench == "corpus":  # mixes choice/score/noul; reproduce on choice rows
            val = [r for r in val if r.get("task") == "ag_news"]
        v = run_split(f"{bench}_val", val)
        T, nll = fit_temperature([list(map(logit, p)) for p in v["raw_probs"]],
                                 v["answers"])
        print(f"{bench}: T {T:.3f} (val_nll {nll:.4f}, val acc {v['acc']:.4f})", flush=True)
        entry = {"T": T, "val": {k: v[k] for k in ("n", "acc", "ece", "brier", "ms_per_row")}}
        for split in ("test", "irr"):
            p = base / f"{split}.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text())
            if bench == "corpus":
                rows = [r for r in rows if r.get("task") == "ag_news"]
            r = run_split(f"{bench}_{split}", rows)
            cal = [softmax(np.log(x + 1e-9) / T) for x in r["raw_probs"]]
            entry[split] = {"n": r["n"], "acc": r["acc"], "ece": r["ece"],
                            "brier": r["brier"],
                            "cal_acc": float(np.mean(
                                [int(c.argmax() == a) for c, a in zip(cal, r["answers"])])),
                            "cal_ece": ece_list(cal, r["answers"]),
                            "cal_brier": brier_list(cal, r["answers"]),
                            "ms_per_row": r["ms_per_row"]}
            print(f"  {bench}_{split}: acc {r['acc']:.4f} cal_acc "
                  f"{entry[split]['cal_acc']:.4f} cal_ece {entry[split]['cal_ece']:.4f}",
                  flush=True)
        results[bench] = entry

    out = ROOT / args.out
    out.write_text(json.dumps(results, indent=1))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()