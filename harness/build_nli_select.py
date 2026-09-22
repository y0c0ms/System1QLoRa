#!/usr/bin/env python3
"""Format-matched NLI: the "select which of three candidates has relation R"
task, reconstructed from MNLI triples.

WHY. Diagnosis (see FINDINGS): the held-out SNI NLI tasks
(task200/201/202_mnli_*) are NOT single-pair label classification. They give a
statement plus THREE candidate sentences and ask which one is
entailed/neutral/contradictory - answer '1','2','3'. Our earlier NLI injection
(build_nli.py) taught single-pair "premise+hypothesis -> label", a different
question. Entailment/contradiction survived the mismatch (lexically salient) but
NEUTRAL collapsed to 0.20 (< chance) because "pick the neutral one" has no
salient signal and needs by-elimination reasoning the label task never taught.

This builds the matching format from MNLI's own 3-way triples (grouped by
promptID: one entailment, one neutral, one contradiction hypothesis per premise),
generating a balanced set of select-of-3 questions with the target relation in a
shuffled position. Different instances than SNI's test rows, so a held-out lift
measures FORMAT TRANSFER, not memorisation.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Question strings matched to SNI task200/201/202 (verbatim), keyed by target label.
QUESTION = {
    0: ("In this task, you're given a statement and three sentences as choices. "
        "Your job is to determine which sentence can be inferred from the statement. "
        "Incorrect choices change the meaning in important ways or have details that "
        "are not mentioned in the statement. Indicate your answer as 1,2, or 3 "
        "corresponding to the choice number of the selected sentence."),
    1: ("In this task, you're given a statement and three sentences as choices. "
        "Your job is to determine the neutral choice based on your inference from the "
        "statement and your commonsense knowledge. The neutral choice is a sentence "
        "that neither agrees nor disagrees with the statement. Indicate your answer as "
        "'1', '2', or '3', corresponding to the choice number of the selected sentence. "
        "If sentence X agrees with sentence Y, one's correctness follows from the other "
        "one. If sentence X disagrees with sentence Y, they can not be correct at the "
        "same time."),
    2: ("In this task, you're given a statement, and three sentences as choices. "
        "Your job is to determine which sentence clearly disagrees with the statement. "
        "Indicate your answer as '1', '2', or '3' corresponding to the choice number of "
        "the selected sentence."),
}
TASKNAME = {0: "nli_select_entailment", 1: "nli_select_neutral", 2: "nli_select_contradiction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyu-mll/multi_nli")
    ap.add_argument("--out", default=str(ROOT / "data" / "nli_select"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-triples", type=int, default=0, help="cap triples (0=all)")
    args = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    grp = defaultdict(dict)   # promptID -> {label: hypothesis}
    prem = {}                 # promptID -> premise
    for ex in ds:
        if ex["premise"] and ex["hypothesis"] and ex["label"] in (0, 1, 2):
            grp[ex["promptID"]][ex["label"]] = ex["hypothesis"]
            prem[ex["promptID"]] = ex["premise"]
    triples = [(prem[p], d) for p, d in grp.items() if set(d) == {0, 1, 2}]
    rng = random.Random(args.seed)
    rng.shuffle(triples)
    if args.max_triples:
        triples = triples[: args.max_triples]

    rows = []
    for premise, hyps in triples:
        for target in (0, 1, 2):
            order = [0, 1, 2]
            rng.shuffle(order)
            choices = [hyps[lab] for lab in order]
            ans = order.index(target)  # 0-based position of the target hypothesis
            state = ("Statement: %s Choices: 1. %s 2. %s 3. %s"
                     % (premise, choices[0], choices[1], choices[2]))
            rows.append({
                "task": TASKNAME[target],
                "question_type": "choice",
                "ordered": True,
                "state": state,
                "question": QUESTION[target],
                "options": ["1", "2", "3"],
                "answer_index": ans,
            })
    rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "train.json", "w"))
    json.dump(rows[:300], open(out / "val.json", "w"))
    json.dump(rows[:300], open(out / "test.json", "w"))
    from collections import Counter
    dist = Counter(r["task"].split("_")[-1] for r in rows)
    posd = Counter(r["answer_index"] for r in rows)
    print(f"triples={len(triples)}  rows={len(rows)}  target_dist={dict(dist)}  "
          f"answer_pos_dist={dict(posd)} -> {out}/train.json")


if __name__ == "__main__":
    main()
