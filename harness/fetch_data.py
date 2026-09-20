#!/usr/bin/env python3
"""Pull the val and test splits of pngwn/system-one-decisions as plain JSON.

WHY NOT `datasets`. Nothing else in this harness needs torch or pyarrow, and the
whole point of T2 is a baseline that runs with ZERO install. The HF
datasets-server exposes rows over REST, so stdlib is enough.

WHY THIS DATASET AND NOT A REBUILD. `system_one.py build` regenerates the same
rows deterministically from 6 upstream HF datasets, but that needs `datasets` and
re-downloads several GB. This is the same corpus the released scorer was trained
on, which is what makes the comparison fair.

  Revision is PINNED. The repo is 2 days old, has an unedited boilerplate dataset
  card describing ticket CSVs rather than this corpus, and a different owner from
  the code repo. Pinning means a silent upstream edit cannot move our baseline.

LICENCE: the dataset is tagged cc-by-nc-4.0 at the whole-dataset level (not just
the ticket subset). Private measurement only; this constraint travels with any
published number. Recorded in docs/FINDINGS.md.
"""

import json
import os
import time
import urllib.parse
import urllib.request

DATASET = "pngwn/system-one-decisions"
REVISION = "fe081073641c58f336acafe090002e4435313c0a"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
PAGE = 100  # datasets-server hard cap per request
# The probe must be fitted where the LoRA scorer was trained, or the
# comparison is between two different amounts of supervision.
SPLITS = tuple(os.environ.get("SPLITS", "val,test").split(","))


def rows(split: str):
    """Yield every row of a split, paging until the server stops giving more."""
    got, total = [], None
    while total is None or len(got) < total:
        q = urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": split,
            "offset": len(got), "length": PAGE, "revision": REVISION,
        })
        url = "https://datasets-server.huggingface.co/rows?" + q
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    body = json.load(r)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                print("    retry %d (%s)" % (attempt + 1, type(e).__name__))
                time.sleep(2 * (attempt + 1))
        if total is None:
            total = body["num_rows_total"]
            print("  %s: %d rows" % (split, total))
        batch = [r["row"] for r in body["rows"]]
        if not batch:
            # Guard against an infinite loop if the server returns an empty page
            # before reaching num_rows_total.
            print("    server returned an empty page at offset %d; stopping short" % len(got))
            break
        got.extend(batch)
        print("    %d/%d" % (len(got), total), end="\r")
    print("    %d/%d done" % (len(got), total))
    return got


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in SPLITS:
        path = os.path.join(OUT, "%s.json" % split)
        if os.path.exists(path):
            n = len(json.load(open(path)))
            print("  %s: cached, %d rows" % (split, n))
            continue
        data = rows(split)
        with open(path, "w") as f:
            json.dump(data, f)

    # Report the shape that actually matters for the benchmark: how many
    # questions each task contributes and how many options they carry, because
    # anything over 26 options has no letter and must be skipped and COUNTED.
    for split in SPLITS:
        data = json.load(open(os.path.join(OUT, "%s.json" % split)))
        by = {}
        for r in data:
            t = r["task"]
            k = len(r["options"])
            e = by.setdefault(t, {"n": 0, "k": k, "qtype": r["question_type"]})
            e["n"] += 1
            e["k"] = max(e["k"], k)
        print("\n  %s split" % split)
        skip = 0
        for t in sorted(by):
            e = by[t]
            flag = "  <-- >26 options, NO LETTER, must skip" if e["k"] > 26 else ""
            if e["k"] > 26:
                skip += e["n"]
            print("    %-18s n=%-5d max_options=%-3d %-7s%s" % (t, e["n"], e["k"], e["qtype"], flag))
        print("    scoreable by letter: %d / %d  (%d skipped)" % (len(data) - skip, len(data), skip))
        qs = {r["question"] for r in data}
        print("    distinct questions: %d" % len(qs))


if __name__ == "__main__":
    main()
