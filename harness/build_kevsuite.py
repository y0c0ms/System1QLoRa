#!/usr/bin/env python3
"""Convert kev-suites (decision-v2) into our {state, question, options,
answer_index} schema so our letter-logprob models can be scored on kev's own
home distribution (the reverse direction of harness/kev_score.py).

kev-suites records are the /v1/systemone contract: a `state` plus typed
`questions` (choice / noul yes-no / score ordinal), each with `criteria` and a
gold `label`. We expand every question into one row, tag it with its source
(banking77, mnli, sst5, …) as the task, and keep only questions with 2..26
options — our letter readout caps at 26, exactly the limitation this experiment
measures. Options with >26 (banking77-77, trec, dbpedia14) are dropped and
reported, so coverage is explicit.
"""

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://huggingface.co/datasets/jaredpalmer/kev-suites/resolve/main"


def render_option(key, desc):
    return str(key) if desc in (None, "") else f"{key}: {desc}"


def convert_question(state, q, source):
    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        if not isinstance(crit, dict) or not (2 <= len(crit) <= 26):
            return None
        keys = list(crit)
        options = [render_option(k, crit[k]) for k in keys]
        if q["label"] not in keys:
            return None
        ai = keys.index(q["label"])
    elif t == "noul":
        keys = list(crit) if isinstance(crit, dict) and len(crit) == 2 else ["true", "false"]
        options = [render_option(k, crit[k]) if isinstance(crit, dict) else k for k in keys]
        want = "true" if q["label"] else "false"
        if want not in keys:
            want = keys[0] if q["label"] else keys[-1]
        ai = keys.index(want)
    elif t == "score":
        if not isinstance(crit, list) or not (2 <= len(crit) <= 26):
            return None
        options = [str(x) for x in crit]
        ai = int(q["label"])
        if not (0 <= ai < len(options)):
            return None
    else:
        return None
    return {"task": source, "state": state if isinstance(state, str) else json.dumps(state),
            "question": q.get("instructions") or "Which option best fits?",
            "options": options, "answer_index": ai}


def load_split(split):
    url = f"{BASE}/{split}.jsonl"
    rows = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build(rows):
    out, dropped = [], 0
    for r in rows:
        state = r["state"]
        for qid, q in r["questions"].items():
            src = (q.get("src") or (r.get("_meta") or {}).get("source") or "unknown")
            row = convert_question(state, q, src)
            if row is None:
                dropped += 1
            else:
                out.append(row)
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "kevsuite"))
    args = ap.parse_args()
    test, d_test = build(load_split("decision-v2/test"))
    val, d_val = build(load_split("decision-v2/development"))
    train, d_train = build(load_split("decision-v2/train"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(test, open(out / "test.json", "w"))
    json.dump(val, open(out / "val.json", "w"))
    json.dump(train, open(out / "train.json", "w"))
    from collections import Counter
    src = Counter(r["task"] for r in train)
    print(f"train: {len(train)} (dropped {d_train}) | test: {len(test)} | val: {len(val)}")
    print("train sources:", dict(src.most_common()))


if __name__ == "__main__":
    main()
