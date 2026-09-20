#!/usr/bin/env python3
"""Build the 'reflex decisions' benchmark - the tasks a System One model is
actually for: fast structured judgments, not deliberation.

Domains (open, permissive, cleanly typed):
  - MATH topic  : choice over 7 areas of mathematics (recognise, don't solve)
  - MATH level  : score over difficulty 1-5 (ordinal)
  - code defect : noul, "does this C function contain a vulnerability?"

Rows use the system_one schema {task, question_type, ordered, state, question,
options, answer_index}. Every option carries its meaning (topic name / level /
yes-no) - no opaque labels. Held-out by DOMAIN: a scorer trained on generic
classification has never seen these, so the test measures real adaptability.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATH_TOPICS = sorted(["Algebra", "Counting & Probability", "Geometry",
                      "Intermediate Algebra", "Number Theory", "Prealgebra",
                      "Precalculus"])
LEVELS = [f"Level {i}" for i in range(1, 6)]


def math_rows(split_rows):
    topic, level = [], []
    for r in split_rows:
        subj = r.get("subject")
        lvl = r.get("level")
        prob = r.get("problem")
        if not (subj in MATH_TOPICS and prob):
            continue
        topic.append({"task": "math_topic", "question_type": "choice", "ordered": False,
                      "state": prob,
                      "question": "Which area of mathematics is this problem from?",
                      "options": MATH_TOPICS, "answer_index": MATH_TOPICS.index(subj)})
        if isinstance(lvl, int) and 1 <= lvl <= 5:
            level.append({"task": "math_level", "question_type": "score", "ordered": True,
                          "state": prob,
                          "question": "What is the difficulty level of this problem, from 1 (easiest) to 5 (hardest)?",
                          "options": LEVELS, "answer_index": lvl - 1})
    return topic, level


def defect_rows(split_rows):
    out = []
    for r in split_rows:
        func = r.get("func")
        tgt = r.get("target")
        if not func or tgt is None:
            continue
        out.append({"task": "code_defect", "question_type": "noul", "ordered": False,
                    "state": func[:4000],
                    "question": "Does this function contain a security vulnerability?",
                    "options": ["no", "yes"], "answer_index": int(bool(tgt))})
    return out


def cap(rows, n, rng):
    return rng.sample(rows, n) if n and len(rows) > n else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/reflex")
    ap.add_argument("--train-per-task", type=int, default=2000)
    ap.add_argument("--val-per-task", type=int, default=200)
    ap.add_argument("--test-per-task", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from datasets import load_dataset
    math = load_dataset("nlile/hendrycks-MATH-benchmark")
    defect = load_dataset("google/code_x_glue_cc_defect_detection")

    m_tr_topic, m_tr_level = math_rows(list(math["train"]))
    m_te_topic, m_te_level = math_rows(list(math["test"]))
    d_tr = defect_rows(list(defect["train"]))
    d_va = defect_rows(list(defect["validation"]))
    d_te = defect_rows(list(defect["test"]))

    # carve val from the train pools; test from the datasets' own test splits
    def carve(pool, n_val):
        rng.shuffle(pool)
        return pool[n_val:], pool[:n_val]

    m_tr_topic, m_va_topic = carve(m_tr_topic, args.val_per_task)
    m_tr_level, m_va_level = carve(m_tr_level, args.val_per_task)

    train = (cap(m_tr_topic, args.train_per_task, rng)
             + cap(m_tr_level, args.train_per_task, rng)
             + cap(d_tr, args.train_per_task, rng))
    val = (cap(m_va_topic, args.val_per_task, rng)
           + cap(m_va_level, args.val_per_task, rng)
           + cap(d_va, args.val_per_task, rng))
    test = (cap(m_te_topic, args.test_per_task, rng)
            + cap(m_te_level, args.test_per_task, rng)
            + cap(d_te, args.test_per_task, rng))

    for rows in (train, val, test):
        rng.shuffle(rows)
    bad = [r for r in train + val + test if not (0 <= r["answer_index"] < len(r["options"]))]
    if bad:
        raise SystemExit(f"{len(bad)} rows with out-of-range answer_index")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        with open(out / f"{name}.json", "w") as f:
            json.dump(rows, f)
    import collections
    manifest = {
        "domains": {"math": "nlile/hendrycks-MATH-benchmark (MIT)",
                    "code": "google/code_x_glue_cc_defect_detection (Devign)"},
        "tasks": {"math_topic": "choice / 7 areas", "math_level": "score / 1-5",
                  "code_defect": "noul / vulnerable?"},
        "held_out": "by domain - a scorer trained on generic classification has not seen these",
        "counts": {s: len(r) for s, r in (("train", train), ("val", val), ("test", test))},
        "per_task_test": dict(collections.Counter(r["task"] for r in test)),
        "seed": args.seed,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("counts:", manifest["counts"], "| test per task:", manifest["per_task_test"], flush=True)

    from datasets import Dataset, DatasetDict
    DatasetDict({s: Dataset.from_list(r) for s, r in
                 (("train", train), ("val", val), ("test", test))}).save_to_disk(ROOT / "data/reflex_ds")
    print("saved data/reflex + data/reflex_ds", flush=True)


if __name__ == "__main__":
    main()