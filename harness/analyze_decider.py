#!/usr/bin/env python3
"""Analyze per-row option logits written by eval_decider.py.

For every eval tag: accuracy; ONE global temperature fitted on the pooled val
rows of that tag (the honest deployment setting) applied everywhere, giving
ECE/Brier/NLL; order sensitivity (orig vs rev) and permutation-averaged
predictions; JevBench per-tier accuracy with the benchmark's chance correction.
Across tags: paired bootstrap 95% CIs of accuracy differences on identical rows.

Subcommands
  report  --tags A B ... [--ref A] [--out file.md]
  decide  --tags A B C --dev sni.val reflex.val kevsuite.val --out decision.json
          pre-registered P1 rule: winner = highest dev macro accuracy (mean over the
          listed dev sets); CIs of every arm vs the first tag are reported alongside.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "results" / "evals"
BOOT = 2000
# JevBench v1.3 Intelligence tier weights (easy 14, standard 28, hard 30, judge 28). No judge
# items are public, so the proxy renormalizes over the tiers present - reported as a proxy.
JEV_TIER_W = {"easy": 0.14, "standard": 0.28, "hard": 0.30}


def load_rows(tag, key):
    fp = EVALS / tag / f"{key}.jsonl"
    if not fp.exists():
        return None
    return [json.loads(l) for l in open(fp)]


def softmax(lg, T=1.0):
    z = np.asarray(lg, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_T(rowsets):
    rows = [r for rs in rowsets for r in rs]
    if not rows:
        return 1.0
    best, bestT = float("inf"), 1.0
    for T in np.arange(0.05, 5.0001, 0.05):
        nll = 0.0
        for r in rows:
            p = softmax(r["logits"], T)
            nll -= np.log(max(p[r["answer"]], 1e-12))
        if nll < best:
            best, bestT = nll, float(T)
    return bestT


def metrics(rows, T, probs=None):
    if not rows:
        return None
    P = probs if probs is not None else [softmax(r["logits"], T) for r in rows]
    y = np.array([r["answer"] for r in rows])
    pred = np.array([int(np.argmax(p)) for p in P])
    conf = np.array([float(np.max(p)) for p in P])
    corr = (pred == y).astype(float)
    brier = float(np.mean([np.sum((p - np.eye(len(p))[a]) ** 2) for p, a in zip(P, y)]))
    nll = float(np.mean([-np.log(max(p[a], 1e-12)) for p, a in zip(P, y)]))
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
        if m.any():
            ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return {"n": len(rows), "acc": float(corr.mean()), "ece": float(ece), "brier": brier,
            "nll": nll, "correct": corr.tolist()}


def paired_ci(ca, cb, seed=0):
    a, b = np.asarray(ca), np.asarray(cb)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(BOOT, n))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def tag_keys(tag):
    return sorted(p.stem for p in (EVALS / tag).glob("*.jsonl"))


def analyze_tag(tag):
    keys = tag_keys(tag)
    val_orig = [load_rows(tag, k) for k in keys if k.endswith(".val.orig")]
    T = fit_T([v for v in val_orig if v])
    res = {"T_global": T, "sets": {}}
    for k in keys:
        rows = load_rows(tag, k)
        res["sets"][k] = metrics(rows, T)
    # order sensitivity + permutation averaging
    for k in keys:
        if not k.endswith(".orig"):
            continue
        kr = k[:-5] + ".rev"
        ro, rr = load_rows(tag, k), load_rows(tag, kr)
        if not rr:
            continue
        agree = np.mean([int(np.argmax(a["logits"]) == np.argmax(b["logits"])) for a, b in zip(ro, rr)])
        avg = [(softmax(a["logits"], T) + softmax(b["logits"], T)) / 2 for a, b in zip(ro, rr)]
        res["sets"][k[:-5] + ".permavg"] = metrics(ro, T, probs=avg)
        res["sets"][k[:-5] + ".permavg"]["order_agreement"] = float(agree)
    # JevBench tiers
    for k in keys:
        if not k.startswith("jevbench.test."):
            continue
        rows = load_rows(tag, k)
        tiers = {}
        for r, c in zip(rows, res["sets"][k]["correct"]):
            t = r.get("tier") or "unknown"
            tiers.setdefault(t, {"c": 0, "n": 0, "chance": 0.0})
            tiers[t]["c"] += c
            tiers[t]["n"] += 1
            tiers[t]["chance"] += 1.0 / r["k"]
        out = {}
        for t, v in tiers.items():
            acc = v["c"] / v["n"]
            ch = v["chance"] / v["n"]
            out[t] = {"n": v["n"], "acc": acc, "chance": ch,
                      "chance_corrected": max(0.0, (acc - ch) / (1 - ch)) if ch < 1 else 0.0}
        w = sum(JEV_TIER_W.get(t, 0) for t in out)
        proxy = sum(JEV_TIER_W.get(t, 0) * v["chance_corrected"] for t, v in out.items()) / w if w else None
        res["sets"][k]["tiers"] = out
        res["sets"][k]["intelligence_proxy"] = None if proxy is None else 100 * proxy
    return res


def fmt(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def cmd_report(args):
    A = {t: analyze_tag(t) for t in args.tags}
    ref = args.ref or args.tags[0]
    keys = sorted({k for t in A for k in A[t]["sets"]})
    lines = ["# Decision-model evaluation report", "",
             "Global temperature per model (fit on pooled val, applied to all sets): " +
             ", ".join(f"`{t}` T={A[t]['T_global']:.2f}" for t in args.tags), "",
             "## Accuracy / ECE (global T)", "",
             "| set | " + " | ".join(args.tags) + " |", "|---|" + "---|" * len(args.tags)]
    for k in keys:
        cells = []
        for t in args.tags:
            m = A[t]["sets"].get(k)
            cells.append("-" if not m else f"{m['acc']:.3f} / {m['ece']:.3f}"
                         + (f" (agree {m['order_agreement']:.2f})" if "order_agreement" in m else ""))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", f"## Paired bootstrap vs `{ref}` (accuracy difference, 95% CI, {BOOT} resamples)", "",
              "| set | " + " | ".join(t for t in args.tags if t != ref) + " |",
              "|---|" + "---|" * (len(args.tags) - 1)]
    for k in keys:
        mr = A[ref]["sets"].get(k)
        if not mr:
            continue
        cells = []
        for t in args.tags:
            if t == ref:
                continue
            m = A[t]["sets"].get(k)
            if not m or m["n"] != mr["n"]:
                cells.append("-")
                continue
            d, lo, hi = paired_ci(m["correct"], mr["correct"])
            sig = "**" if (lo > 0 or hi < 0) else ""
            cells.append(f"{sig}{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    for t in args.tags:
        for k, m in A[t]["sets"].items():
            if m and "tiers" in m:
                lines += ["", f"### JevBench public tiers - `{t}` `{k}` (intelligence proxy "
                              f"{fmt(m['intelligence_proxy'], 1)})", "",
                          "| tier | n | acc | chance | chance-corrected |", "|---|---|---|---|---|"]
                for tn, v in sorted(m["tiers"].items()):
                    lines.append(f"| {tn} | {v['n']} | {v['acc']:.3f} | {v['chance']:.3f} | "
                                 f"{v['chance_corrected']:.3f} |")
    txt = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(txt)
        json.dump({t: {"T_global": A[t]["T_global"],
                       "sets": {k: {kk: vv for kk, vv in (m or {}).items() if kk != "correct"}
                                for k, m in A[t]["sets"].items()}} for t in A},
                  open(Path(args.out).with_suffix(".json"), "w"), indent=1)
    print(txt)


def cmd_decide(args):
    A = {t: analyze_tag(t) for t in args.tags}
    dev = [f"{d}.orig" if not d.endswith((".orig", ".rev")) else d for d in args.dev]
    macro = {}
    for t in args.tags:
        accs = [A[t]["sets"][k]["acc"] for k in dev if A[t]["sets"].get(k)]
        macro[t] = float(np.mean(accs)) if len(accs) == len(dev) else None
    valid = {t: m for t, m in macro.items() if m is not None}
    winner = max(valid, key=valid.get) if valid else args.tags[0]
    ref = args.tags[0]
    cis = {}
    for t in args.tags:
        if t == ref:
            continue
        per = {}
        for k in dev:
            a, b = A[t]["sets"].get(k), A[ref]["sets"].get(k)
            if a and b and a["n"] == b["n"]:
                per[k] = paired_ci(a["correct"], b["correct"])
        cis[t] = per
    dec = {"rule": "winner = argmax mean dev accuracy over " + ", ".join(dev),
           "macro": macro, "winner": winner, "ci_vs_" + ref: cis,
           "per_set": {t: {k: A[t]["sets"][k]["acc"] for k in dev if A[t]["sets"].get(k)} for t in args.tags}}
    json.dump(dec, open(args.out, "w"), indent=1)
    print(json.dumps(dec, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("report")
    r.add_argument("--tags", nargs="+", required=True)
    r.add_argument("--ref", default="")
    r.add_argument("--out", default="")
    d = sp.add_parser("decide")
    d.add_argument("--tags", nargs="+", required=True)
    d.add_argument("--dev", nargs="+", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    (cmd_report if a.cmd == "report" else cmd_decide)(a)


if __name__ == "__main__":
    main()
