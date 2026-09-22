#!/usr/bin/env python3
"""Compare our model versions against frozen Jev answers.

Jev (opencode-zen/jev-1.13) was scored once via omp's `default` model role and its
raw per-row answers saved to results/jev_bench/<dataset>.json. This script re-reads
those (no re-query, no cost) plus our own eval JSONs and prints the gap table, so
every new model version can be diffed against the same Jev baseline forever.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JEVDIR = ROOT / "results" / "jev_bench"
# our eval JSONs: (main eval, optional separate bfcl_irr eval). Add versions here.
OURS = {
    "0.6B v1": ("results/q06_v1_eval.json", "results/q06_v1_irr.json"),
    "0.6B v2": ("results/q06_v2_eval.json", "results/q06_v2_irr.json"),
    "4B v1": ("results/jevlike4b_eval.json", "results/jevlike4b_v1_irr.json"),
    "4B v2 +abstain": ("results/jevlike4b_v2_eval.json", None),
    "4B v3 +upweight": ("results/jevlike4b_v3_eval.json", None),
    "0.6B final": ("results/q06_final_eval.json", None),
    "4B final": ("results/q4b_final_eval.json", None),
}
DATASETS = ["sni", "reflex", "bfcl", "bfcl_irr"]


def read(p):
    try:
        return json.loads(subprocess.run(["sudo", "cat", str(p)], capture_output=True, text=True).stdout)
    except Exception:
        return None


def jev_acc(name):
    d = read(JEVDIR / f"{name}.json")
    return d["acc_all"] if d else None


def ours_acc(entry, name):
    eval_path, irr_path = entry
    if name == "bfcl_irr" and irr_path and read(ROOT / irr_path):
        return read(ROOT / irr_path)["datasets"]["bfcl_irr"]["acc_all"]
    d = read(ROOT / eval_path)
    if not d:
        return None
    return d.get("datasets", {}).get(name, {}).get("acc_all")


def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def main():
    versions = {k: v for k, v in OURS.items() if read(ROOT / v[0])}
    cols = list(versions) + ["Jev 1.13"]
    print("=== per-benchmark accuracy (calibrated test) ===")
    print(f"{'benchmark':<24}" + "".join(f"{c:>16}" for c in cols))
    for name in DATASETS:
        row = [ours_acc(v, name) for v in versions.values()] + [jev_acc(name)]
        print(f"{name:<24}" + "".join(f"{fmt(x):>16}" for x in row))

    # SNI per-task: biggest gaps to Jev (using the last available our-version)
    jev = read(JEVDIR / "sni.json")
    last = list(versions.values())[-1] if versions else None
    our = read(ROOT / last[0]) if last else None
    if jev and our:
        from collections import defaultdict
        jt = defaultdict(lambda: [0, 0])
        for x in jev["rows"]:
            jt[x["task"]][1] += 1
            jt[x["task"]][0] += int(x["correct"])
        jtask = {t: c / n for t, (c, n) in jt.items()}
        otask = our["datasets"]["sni"]["acc_by_task"]
        print("\n=== SNI: biggest remaining gap to Jev (worst first) ===")
        for t in sorted(otask, key=lambda x: jtask.get(x, 0) - otask[x], reverse=True)[:8]:
            print(f"  ours {otask[t]:.3f}  Jev {jtask.get(t,0):.3f}  (gap {jtask.get(t,0)-otask[t]:+.3f})  {t}")


if __name__ == "__main__":
    main()
