#!/usr/bin/env python3
"""Build a 3-way NLI decision source from public MNLI, to test whether teaching
the entailment/neutral/contradiction concept lifts our held-out SNI NLI tasks.

WHY. The SNI category-disjoint split put the entire entailment family in TEST:
train has no 'entailment'/'neutral'/'contradiction' labels at all, so the model
has no NLI concept to generalise from. This adds that concept. NOTE: this trains
on the NLI predicate family that SNI tests, so a lift here answers "is the gap a
data gap or a capacity gap?" - it does NOT preserve SNI as a pure held-out-
generality measure for those tasks. Kept as a diagnostic lever, reported as such.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Entailment", "Neutral", "Contradiction"]  # MNLI: 0,1,2
QUESTION = "What is the logical relationship of the hypothesis to the premise?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1600)
    ap.add_argument("--dataset", default="nyu-mll/multi_nli")
    ap.add_argument("--out", default=str(ROOT / "data" / "nli"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    rng = random.Random(args.seed)

    by_label = {0: [], 1: [], 2: []}
    for ex in ds:
        lab = ex["label"]
        if lab in by_label and ex["premise"] and ex["hypothesis"]:
            by_label[lab].append(ex)
    rows = []
    for lab, exs in by_label.items():
        rng.shuffle(exs)
        for ex in exs[: args.per_class]:
            rows.append({
                "task": "nli_mnli_3way",
                "question_type": "choice",
                "ordered": False,
                "state": f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}",
                "question": QUESTION,
                "options": LABELS,
                "answer_index": lab,
            })
    rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "train.json", "w"))
    # tiny val/test so the dir matches the standard schema (unused for SNI eval)
    json.dump(rows[:200], open(out / "val.json", "w"))
    json.dump(rows[:200], open(out / "test.json", "w"))
    from collections import Counter
    dist = Counter(LABELS[r["answer_index"]] for r in rows)
    print(f"wrote {len(rows)} NLI rows -> {out}/train.json  dist={dict(dist)}")


if __name__ == "__main__":
    main()
