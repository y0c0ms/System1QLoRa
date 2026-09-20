#!/usr/bin/env python3
"""Build the tool-calling eval benchmark from BFCL v3 (ungated, apache-2.0).

Handoff context: our own agent transcripts are unusable as training data (73%
single tool, near-identical option sets), so tool-calling generality must come
from public data. BFCL `multiple` + `live_multiple` have a SINGLE-function
ground truth on every row - a clean {state, question, options=function names,
answer_index} mapping where the available functions ARE the runtime option set.

Provenance: HF mirror `llamastack/bfcl_v3` (the official gorilla-llm repo no
longer ships v3 JSON files). Subset counts match the handoff's verified
numbers: multiple=200, live_multiple=1053 (=1,253), live_irrelevance=882.

Design decisions (recorded in the manifest):
- NEVER train on BFCL: it is the canonical benchmark; burning it destroys the
  ability to report a number.
- val is a seeded carve from `multiple` (fits temperature); test = remaining
  multiple + live_multiple (live = genuinely newer, held out of T).
- One live_multiple row has an empty ground truth (no-call); it is filtered
  and counted, not smuggled in.
- `irrelevance` + `live_irrelevance` form the "should I call any tool at all"
  slice, with a `None of the above` sentinel option - the most valuable
  calibration slice for a scorer.
- multi_turn rows are excluded (conversation state, not single decisions).
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "llamastack/bfcl_v3"
QUESTION = "Which function should be called?"
NONE_OPTION = "None of the above"
SUBSETS = {
    "multiple": "bfcl_v3_multi",
    "live_multiple": "bfcl_v3_live_multi",
    "irrelevance": "bfcl_v3_irr",
    "live_irrelevance": "bfcl_v3_live_irr",
}


def row_from(ex):
    """One eval row. Returns None when not a clean single-answer decision."""
    if ex.get("multi_turn"):
        return None
    try:
        gt = json.loads(ex["ground_truth"])
        fns = json.loads(ex["functions"])
        turns = json.loads(ex["turns"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(gt, list) or len(gt) != 1 or not isinstance(gt[0], dict):
        return None  # empty (no-call), multi-call, or malformed
    # this mirror encodes the call as {function_name: arguments}
    name = gt[0].get("name") or next(iter(gt[0]), None)
    names = [f.get("name") for f in fns if isinstance(f, dict) and f.get("name")]
    if not name or name not in names:
        return None
    msgs = turns[0] if turns and isinstance(turns[0], list) else turns  # nested conv
    user = [t.get("content", "") for t in msgs
            if isinstance(t, dict) and t.get("role") == "user"]
    if not user:
        return None
    return {
        "task": SUBSETS.get(ex.get("subset"), ex.get("subset", "bfcl")),
        "question_type": "choice",
        "ordered": False,
        "state": "\n".join(user),
        "question": QUESTION,
        "options": names,
        "answer_index": names.index(name),
    }


def irr_row(ex):
    """No-call rows: correct behaviour is NOT invoking any function."""
    if ex.get("multi_turn"):
        return None
    try:
        fns = json.loads(ex["functions"])
        turns = json.loads(ex["turns"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    names = [f.get("name") for f in fns if isinstance(f, dict) and f.get("name")]
    msgs = turns[0] if turns and isinstance(turns[0], list) else turns  # nested conv
    user = [t.get("content", "") for t in msgs
            if isinstance(t, dict) and t.get("role") == "user"]
    if not user or not names:  # need >=1 function so the row is a real >=2-option choice
        return None
    return {
        "task": SUBSETS.get(ex.get("subset"), ex.get("subset", "bfcl_irr")),
        "question_type": "choice",
        "ordered": False,
        "state": "\n".join(user),
        "question": QUESTION,
        "options": names + [NONE_OPTION],
        "answer_index": len(names),  # the sentinel
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/bfcl")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--val-rows", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.repo)["train"]
    print("total rows:", len(ds), flush=True)

    multi, live, irr, irr_live, filtered = [], [], [], [], 0
    for ex in ds:
        sub = ex.get("subset")
        if sub == "multiple":
            r = row_from(ex)
            if r:
                multi.append(r)
            else:
                filtered += 1
        elif sub == "live_multiple":
            r = row_from(ex)
            if r:
                live.append(r)
            else:
                filtered += 1
        elif sub == "irrelevance":
            r = irr_row(ex)
            if r:
                irr.append(r)
            else:
                filtered += 1
        elif sub == "live_irrelevance":
            r = irr_row(ex)
            if r:
                irr_live.append(r)
            else:
                filtered += 1
    print(f"clean: multiple {len(multi)} live_multiple {len(live)} "
          f"irr {len(irr)} live_irr {len(irr_live)} (filtered {filtered})", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(multi)
    val, test = multi[: args.val_rows], multi[args.val_rows:]
    test.extend(live)
    all_irr = irr + irr_live

    bad = [r for r in val + test + all_irr
           if not (0 <= r["answer_index"] < len(r["options"]))]
    dup = [r for r in val + test if len(set(r["options"])) != len(r["options"])]
    if bad:
        raise SystemExit(f"{len(bad)} rows out-of-range answer_index")
    if dup:
        raise SystemExit(f"{len(dup)} rows with duplicate options")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("val", val), ("test", test), ("irr", all_irr)):
        with open(out / f"{name}.json", "w") as f:
            json.dump(rows, f)
    manifest = {
        "source": args.repo,
        "note": "BFCL is the canonical benchmark: eval only, never train. "
                "xlam train-side is gated click-through (HF_TOKEN unset).",
        "seed": args.seed,
        "counts": {"val": len(val), "test": len(test), "irr": len(all_irr),
                   "filtered": filtered},
        "filtered_reason": "multi_turn, empty/multi-call ground truth, or answer not in options "
                           "(live_multiple has exactly 1 empty ground truth row)",
        "val_from": "bfcl_v3_multi (seeded carve)",
        "test_from": "bfcl_v3_multi remainder + bfcl_v3_live_multi",
        "irr_from": "bfcl_v3_irrelevance + bfcl_v3_live_irrelevance "
                    "('None of the above' sentinel option)",
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote:", json.dumps(manifest["counts"]), flush=True)

    from datasets import Dataset, DatasetDict
    splits = {"val": val, "test": test, "irr": all_irr}
    DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()}).save_to_disk(
        ROOT / "data/bfcl_ds")
    print("saved Arrow DatasetDict to data/bfcl_ds", flush=True)


if __name__ == "__main__":
    main()