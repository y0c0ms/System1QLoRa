#!/usr/bin/env python3
"""Build the Super-NaturalInstructions generality benchmark.

Turns allenai/natural-instructions (GitHub clone, NOT the HF mirror -- the
mirror drops Categories, and Categories are what the category-disjoint val
carve needs) into the row schema system_one.py consumes:

    {task, question_type, ordered, state, question, options, answer_index}

Why SNI: the shipped corpus has 36 distinct question strings, all of which
appear in train, val AND test. This split is the opposite: 756 train / 119
test tasks whose CATEGORIES are disjoint (verified: train∩test = {}), and the
val split is carved OUT OF TRAIN TASKS BY CATEGORY so no val predicate leaks
into the LoRA training set. A model that scores well on this test did not see
any of the 6 test categories at training time -- that is the first
held-out-predicate measurement this project can make.

Selection rules (all reviewable in the emitted manifest):

    TEST    <= 5 distinct outputs, single-output instances, inputs < 4k chars,
            not input-pointer/positional label sets, no NC license.
            Input-pointer = labels point at content presented IN the input
            (winogrande A/B, COPA 1/2, GAP A/B/Neither, ROCStories a/b,
            spolin "Response 1/2"): the global option set is meaningless there
            because the true options are per-instance. Codebook tasks (snli
            E/C/N, mnli 1/2/3, iirc a/b) stay -- the Definition enumerates the
            codes and the label set is global and input-independent.
    VAL     whole categories of train tasks, sampled <= --val-rows-per-task.
    TRAIN   remaining usable train tasks (2..16 labels, semantic sets only),
            sampled <= --train-rows-per-task.

Contamination caveat (state in every result): every SNI task has been public
since 2022 and in the Flan Collection since 2023. Held-out-ness is enforced by
OUR partition, not by the model's ignorance.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

# Input-pointer label sets, one reason each. These are the "Tier C" tasks:
# the options live in the input, so sorted(distinct outputs) is not the option
# set any real request would carry.
INPUT_POINTER = {
    "task1391_winogrande_easy_answer_generation.json": "winogrande: candidate names are in the input, labels A/B point at them",
    "task827_copa_commonsense_reasoning.json": "COPA: the two alternatives are in the input, labels 1/2 point at them",
    "task1393_superglue_copa_text_completion.json": "COPA: alternatives in the input, labels A/B point at them",
    "task329_gap_classification.json": "GAP: candidate names in the input, labels A/B/Neither point at them",
    "task220_rocstories_title_classification.json": "ROCStories: the two candidate titles are in the input, labels a/b point at them",
    "task362_spolin_yesand_prompt_response_sub_classification.json": "spolin: the two candidate responses are in the input, labels 'Response 1/2' point at them",
}

VAL_CATEGORIES = [
    "Sentiment Analysis",
    "Toxic Language Detection",
    "Commonsense Classification",
    "Text Categorization",
    "Text Matching",
]

NONCOMMERCIAL = re.compile(r"CC\s*BY\s*-?\s*NC", re.IGNORECASE)


def load_task(path):
    with open(path) as f:
        return json.load(f)


def analyze(path):
    d = load_task(path)
    insts = d.get("Instances", [])
    outs = [o for inst in insts for o in inst.get("output", [])]
    uniq = sorted(set(outs))
    multi = sum(1 for inst in insts if len(inst.get("output", [])) > 1)
    lens = sorted(len(inst["input"]) for inst in insts)
    lic = (d.get("Instance License") or ["?"])[0]
    return {
        "path": path,
        "name": path.name,
        "categories": d.get("Categories", []),
        "definition": d.get("Definition", [""])[0],
        "n_inst": len(insts),
        "n_out": len(uniq),
        "labels": uniq,
        "multi_out": multi,
        "in_p95": lens[int(len(lens) * 0.95)] if lens else 0,
        "license": lic,
    }


def positional(t):
    """Label set is single letters / digits / 'Response N' -- i.e. the input
    carries the real options or the answers are opaque codes the model would
    have to learn as a shortcut. Excluded from TRAIN."""
    labs = t["labels"]
    return (
        len(labs) >= 2
        and (
            all(re.fullmatch(r"[A-Za-z]", l) for l in labs)
            or all(re.fullmatch(r"\d+", l) for l in labs)
            or all(re.fullmatch(r"Response \d+", l) for l in labs)
        )
    )


def nc(t):
    """Real Creative Commons non-commercial variants only (CC BY-NC[-SA]).

    A substring test is wrong here: 'NC' appears inside 'licence' and 'OANC'
    (MNLI's Open American National Corpus license), which would silently drop
    valid tasks (observed: it excluded mnli x3 and MultiRC)."""
    return bool(NONCOMMERCIAL.search(t["license"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ni-dir", default="upstream/natural-instructions")
    ap.add_argument("--out-dir", default="data/sni")
    ap.add_argument("--test-max-labels", type=int, default=5)
    ap.add_argument("--train-max-labels", type=int, default=16)
    ap.add_argument("--max-input-p95", type=int, default=4000)
    ap.add_argument("--test-rows-per-task", type=int, default=200)
    ap.add_argument("--val-rows-per-task", type=int, default=40)
    ap.add_argument("--train-rows-per-task", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    ni = Path(args.ni_dir)
    tasks = ni / "tasks"


    def split_names(fn):
        return [
            n + ".json" if not n.endswith(".json") else n
            for n in (ni / "splits" / "default" / fn).read_text().split()
        ]

    train_names, test_names = split_names("train_tasks.txt"), split_names("test_tasks.txt")
    print(f"split files: {len(train_names)} train / {len(test_names)} test", flush=True)

    all_t = [analyze(tasks / n) for n in train_names + test_names]
    by_name = {t["name"]: t for t in all_t}

    train_cats = {c for t in (by_name[n] for n in train_names) for c in t["categories"]}
    test_cats = {c for t in (by_name[n] for n in test_names) for c in t["categories"]}
    overlap = train_cats & test_cats
    print(f"categories: train {len(train_cats)} / test {len(test_cats)} / overlap {overlap}", flush=True)
    assert not overlap, "category-disjoint split violated -- this benchmark is broken"

    # ---- TEST selection -------------------------------------------------
    test_sel, test_excl = [], {}
    for n in test_names:
        t = by_name[n]
        if t["n_out"] > args.test_max_labels:
            test_excl[n] = f">{args.test_max_labels} labels ({t['n_out']})"
        elif t["in_p95"] >= args.max_input_p95:
            test_excl[n] = f"input p95 {t['in_p95']} >= {args.max_input_p95}"
        elif n in INPUT_POINTER:
            test_excl[n] = "input-pointer label set: " + INPUT_POINTER[n]
        elif nc(t):
            test_excl[n] = f"non-commercial license: {t['license']}"
        elif t["n_out"] < 2:
            test_excl[n] = "single label, no decision to make"
        else:
            test_sel.append(t)
    test_sel.sort(key=lambda t: t["name"])

    # ---- VAL + TRAIN selection (inside train tasks) ---------------------
    usable_cats = set(VAL_CATEGORIES)
    for c in usable_cats:
        assert c not in test_cats, f"val category {c!r} collides with test categories"
    val_sel, train_sel, train_excl = [], [], {}
    for n in train_names:
        t = by_name[n]
        if not (2 <= t["n_out"] <= args.train_max_labels):
            train_excl[n] = f"labels {t['n_out']} outside [2,{args.train_max_labels}]"
        elif t["multi_out"]:
            train_excl[n] = f"{t['multi_out']} instances with multiple outputs"
        elif t["in_p95"] >= args.max_input_p95:
            train_excl[n] = f"input p95 {t['in_p95']} >= {args.max_input_p95}"
        elif nc(t):
            train_excl[n] = f"non-commercial license: {t['license']}"
        elif positional(t):
            train_excl[n] = "positional/codebook letter-digit label set"
        elif t["categories"][0] in usable_cats:
            val_sel.append(t)
        else:
            train_sel.append(t)
    val_sel.sort(key=lambda t: t["name"])
    train_sel.sort(key=lambda t: t["name"])

    # ---- row mapping ----------------------------------------------------
    import random
    rng = random.Random(args.seed)

    def rows_for(ts, cap):
        out, per_task, empty = [], {}, []
        for t in ts:
            d = load_task(t["path"])
            picked = rng.sample(d["Instances"], min(cap, len(d["Instances"])))
            rows = []
            for inst in picked:
                outs = inst["output"]
                if len(outs) != 1:
                    continue
                rows.append({
                    "task": t["name"][:-5],  # strip .json
                    "question_type": "choice",
                    "ordered": False,
                    "state": inst["input"],
                    "question": t["definition"],
                    "options": t["labels"],
                    "answer_index": t["labels"].index(outs[0]),
                })
            if not rows:
                empty.append(t["name"])
            per_task[t["name"]] = len(rows)
            out.extend(rows)
        return out, per_task, empty

    test_rows, test_per, test_empty = rows_for(test_sel, args.test_rows_per_task)
    val_rows, val_per, val_empty = rows_for(val_sel, args.val_rows_per_task)
    train_rows, train_per, train_empty = rows_for(train_sel, args.train_rows_per_task)
    for split, empty in (("train", train_empty), ("val", val_empty), ("test", test_empty)):
        if empty:
            raise SystemExit(f"{split}: selected tasks produced zero single-answer rows: {empty}")
    print(f"rows: train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)} "
          f"({len(train_sel)} train tasks / {len(val_sel)} val tasks / {len(test_sel)} test tasks)",
          flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        with open(out / f"{split}.json", "w") as f:
            json.dump(rows, f)
    print(f"wrote {out / 'train.json'}, {out / 'val.json'}, {out / 'test.json'}", flush=True)

    # ---- manifest -------------------------------------------------------
    rev = subprocess.run(["git", "-C", str(ni), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    manifest = {
        "source": "github.com/allenai/natural-instructions",
        "source_rev": rev.stdout.strip() if rev.returncode == 0 else "unknown",
        "note": "every SNI task has been public since 2022 and in Flan since 2023; "
                "held-out-ness is enforced by OUR partition, not model ignorance",
        "categories": {
            "train": sorted(train_cats),
            "test": sorted(test_cats),
            "overlap": sorted(overlap),
            "val_carved_from_train": sorted(usable_cats),
        },
        "selection": {
            "test_max_labels": args.test_max_labels,
            "train_max_labels": args.train_max_labels,
            "max_input_p95": args.max_input_p95,
            "seed": args.seed,
        },
        "counts": {
            "train_tasks_selected": len(train_sel),
            "val_tasks_selected": len(val_sel),
            "test_tasks_selected": len(test_sel),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "rows_per_task": {"train": train_per, "val": val_per, "test": test_per},
        },
        "excluded": {"train": train_excl, "test": test_excl},
        "test_tasks": [
            {k: t[k] for k in ("name", "n_inst", "n_out", "labels", "license")}
            for t in test_sel
        ],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out / 'manifest.json'}", flush=True)

    # ---- verification: reload what was written and assert the invariants --
    if args.verify:
        loaded = {s: json.load(open(out / f"{s}.json")) for s in ("train", "val", "test")}
        checks = []

        def ok(name, cond):
            checks.append((name, bool(cond)))

        for split, rows in loaded.items():
            ok(f"{split} nonempty", len(rows) > 0)
            for r in rows:
                ok(f"{split} answer_index in range", 0 <= r["answer_index"] < len(r["options"]))
                ok(f"{split} has state/question/options", r["state"] and r["question"] and r["options"])

        def task_set(rows):
            return {r["task"] for r in rows}

        ts, vs, trs = (task_set(loaded[s]) for s in ("test", "val", "train"))
        ok("train vs val tasks disjoint", trs.isdisjoint(vs))
        ok("train vs test tasks disjoint", trs.isdisjoint(ts))
        ok("val vs test tasks disjoint", vs.isdisjoint(ts))

        # options are exactly the label set of that task in every split
        for split, rows in loaded.items():
            per_task = {}
            for r in rows:
                if r["task"] not in per_task:
                    per_task[r["task"]] = set()
                per_task[r["task"]].add(tuple(r["options"]))
            ok(f"{split} one option set per task", all(len(v) == 1 for v in per_task.values()))
            ok(f"{split} every option set >= 2",
               all(len(next(iter(v))) >= 2 for v in per_task.values()))
        ok("split row counts match manifest",
           {s: len(loaded[s]) for s in loaded}
           == {s: manifest["counts"][s + "_rows"] for s in loaded})
        ok("test task count matches manifest", len(ts) == manifest["counts"]["test_tasks_selected"])
        failed = [n for n, c in checks if not c]
        print(f"verification: {len(checks) - len(failed)}/{len(checks)} checks passed", flush=True)
        manifest["verification"] = {
            "passed": len(checks) - len(failed),
            "total": len(checks),
            "failed": failed,
        }
        with open(out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
        if failed:
            raise SystemExit(f"verification failed: {failed}")


if __name__ == "__main__":
    main()