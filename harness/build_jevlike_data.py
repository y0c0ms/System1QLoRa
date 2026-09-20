#!/usr/bin/env python3
"""Convert benchmark rows to jevlike JSONL (context/options/label).

jevlike (github.com/vinnylarouge/jevlike) scores one context against a runtime
list of options - the same I/O shape as Jev and as our system_one rows, but a
different architecture (option-as-query attention over a frozen/byte encoder,
no decoder). Mapping keeps the exact rows we use elsewhere, so the comparison
is on identical ground truth:

    context = question + "\\n" + state
    options = row options
    label   = answer_index

Emits data/jevlike/{train,val,test}.jsonl. Also converts the BFCL splits for a
second held-out check.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/jevlike"


def convert(name, rows):
    out = []
    for r in rows:
        out.append({
            "context": f"{r['question']}\n{r['state']}",
            "options": r["options"],
            "label": int(r["answer_index"]),
        })
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    return len(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = json.loads((ROOT / f"data/sni/{split}.json").read_text())
        n = convert(f"sni_{split}", rows)
        print(f"sni_{split}: {n} rows", flush=True)
    for split in ("val", "test", "irr"):
        p = ROOT / f"data/bfcl/{split}.json"
        if p.exists():
            n = convert(f"bfcl_{split}", json.loads(p.read_text()))
            print(f"bfcl_{split}: {n} rows", flush=True)
    print("written to", OUT, flush=True)


if __name__ == "__main__":
    main()