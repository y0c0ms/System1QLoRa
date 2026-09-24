#!/usr/bin/env python3
"""Convert JevBench's 231 PUBLIC items into our {state, question, options,
answer_index} schema, keeping each item's tier (easy / standard / hard; file 'original' = board tier 'standard') so the
analyzer can apply JevBench's per-tier chance correction.

JevBench (fstandhartinger/jevbench, MIT) is the ecosystem's shared board for
Jev-class decision models; its public items are what Dohnuts (152/231) and
others report locally. The 303 held-out items are only run by the maintainer.

Primitive mapping (option text = "key: description", criteria order preserved,
because order is part of each item as authored):
  choice : criteria {key: desc}, expected key  -> index of key
  noul   : criteria {key: desc} or labels [no, yes], expected 'yes'/'no'
  score  : criteria [level descriptions], expected int level -> that index
EVALUATION ONLY - never add these items to a training corpus.
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://raw.githubusercontent.com/fstandhartinger/jevbench/{rev}/datasets/public/{f}.jsonl"
TIERS = ["easy", "original", "hard"]
TIER_NAME = {"easy": "easy", "original": "standard", "hard": "hard"}  # board naming


def opt_text(key, desc):
    key = str(key)
    return key if desc in (None, "") else f"{key}: {desc}"


def convert(item, tier):
    q = item["question"]
    t = q["type"]
    crit = q.get("criteria")
    labels = item.get("labels")
    exp = item["expected"]
    if t in ("choice", "noul"):
        if isinstance(crit, dict) and crit:
            keys = list(crit)
            if t == "noul":
                # criteria are keyed true/false while gold labels are yes/no
                yn = {"true": "yes", "false": "no"}
                opts = [opt_text(yn.get(k, k), crit[k]) for k in keys]
                keys = [yn.get(k, k) for k in keys]
            else:
                opts = [opt_text(k, crit[k]) for k in keys]
        else:
            keys = [str(x) for x in labels]
            opts = keys[:]
        ans = keys.index(str(exp).lower() if t == "noul" else str(exp))
    elif t == "score":
        if isinstance(crit, list) and crit:
            opts = [opt_text(i, d) for i, d in enumerate(crit)]
        else:
            opts = [str(x) for x in labels]
        ans = int(exp)
    else:
        raise ValueError(f"unknown type {t}")
    if not (0 <= ans < len(opts)):
        raise ValueError(f"{item['id']}: answer {exp!r} not in options")
    return {"task": f"jevbench_{tier}_{item.get('family', 'na')}", "tier": tier, "id": item["id"],
            "qtype": t, "state": item["state"],
            "question": q.get("instructions") or "Which option is correct?",
            "options": opts, "answer_index": ans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="main", help="jevbench git revision to pin")
    ap.add_argument("--src", default="", help="local clone path instead of downloading")
    ap.add_argument("--out", default=str(ROOT / "data" / "jevbench"))
    args = ap.parse_args()
    rev = args.rev
    if args.src:
        rev = subprocess.run(["git", "-C", args.src, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or rev
    rows = []
    for tier in TIERS:
        if args.src:
            lines = open(Path(args.src) / "datasets" / "public" / f"{tier}.jsonl").read().splitlines()
        else:
            with urllib.request.urlopen(REPO.format(rev=rev, f=tier)) as f:
                lines = f.read().decode().splitlines()
        rows += [convert(json.loads(l), TIER_NAME[tier]) for l in lines if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "test.json", "w"))
    json.dump({"source": "fstandhartinger/jevbench public items", "revision": rev, "n": len(rows),
               "license": "MIT", "use": "evaluation only"}, open(out / "manifest.json", "w"), indent=1)
    from collections import Counter
    print(f"wrote {len(rows)} JevBench public items @ {rev[:12]} ->", out)
    print(" tiers:", dict(Counter(r["tier"] for r in rows)), "| types:", dict(Counter(r["qtype"] for r in rows)))
    print(" options:", dict(sorted(Counter(len(r["options"]) for r in rows).items())))


if __name__ == "__main__":
    main()
