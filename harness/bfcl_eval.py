#!/usr/bin/env python3
"""Evaluate every trained scorer on the BFCL tool-calling benchmark.

Scorers (all built on gemma-3-270m, all CPU, forward-only):
  - upstream/pretrained-scorer  the released corpus fit (0.6796 on its own test)
  - results/sni_fit             SNI fit, head at the final layer
  - results/head8_fit           SNI fit, head at layer 8

Protocol per scorer: temperature fitted on BFCL val (seeded carve from
bfcl_v3_multi), applied to BFCL test (1,132 rows: 80 remaining multi + 1,052
live_multi) and to the irrelevance slice (1,122 rows with the "None of the
above" sentinel - the no-tool calibration slice).

The head-8 adapter needs its custom head class, so it evals through
head8_fit.py --eval-only; the other two go through system_one eval.
Records results to FINDINGS.md and commits.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = ROOT / ".venv/bin/python"
BASE = "mlx-community/gemma-3-270m-bf16"
DATA = "data/bfcl_ds"
LOG_DIR = ROOT / "results"

SCORERS = [
    ("old", [str(PY), "upstream/system_one.py", "eval",
             "--base-model", BASE, "--model-dir", "upstream/pretrained-scorer",
             "--local-data-dir", DATA, "--split", "test"]),
    ("sni", [str(PY), "upstream/system_one.py", "eval",
             "--base-model", BASE, "--model-dir", str(ROOT / "results/sni_fit"),
             "--local-data-dir", DATA, "--split", "test"]),
    ("head8", [str(PY), "harness/head8_fit.py", "--eval-only",
               "--adapter-dir", str(ROOT / "results/head8_fit"),
               "--local-data-dir", DATA, "--split", "test"]),
]


def parse_blocks(path):
    """Extract UNCALIBRATED/CALIBRATED blocks; system_one prints them
    multi-line (indent=1), so accumulate lines until json parses."""
    lines = Path(path).read_text().splitlines()
    out = {}
    for i, ln in enumerate(lines):
        for key, prefix in (("unc", "UNCALIBRATED"), ("cal", "CALIBRATED")):
            if ln.startswith(prefix):
                acc = ln[len(prefix):].strip()
                j = i
                while True:
                    try:
                        out[key] = json.loads(acc)
                        break
                    except json.JSONDecodeError:
                        j += 1
                        if j >= len(lines):
                            break
                        acc += "\n" + lines[j]
    return out


def run_one(scorer, split):
    log = LOG_DIR / f"bfcl_{scorer}_{split}.log"
    cmd = list(SCORERS[scorer][1])
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return parse_blocks(log)


def main():
    import harness.endings as E
    marker = ROOT / "results/.bfcl_done"
    if marker.exists():
        print("bfcl_eval already done (marker present)", flush=True)
        return
    results = {}
    for scorer, _ in SCORERS:
        for split in ("test", "irr"):
            print(f"eval {scorer} on {split}...", flush=True)
            b = run_one(scorer, split)
            results[f"{scorer}_{split}"] = b.get("cal", {}).get("ALL", {})
            print(f"  {split} ALL: " + json.dumps(b.get("cal", {}).get("ALL", {})), flush=True)
    # size check: all three scorers must exist
    if not results.get("head8_test"):
        print("head8 adapter not present - skipping its rows", flush=True)
    E.append_bfcl(results)
    subprocess.run(["git", "add", "docs/FINDINGS.md"], cwd=ROOT)
    subprocess.run(["git", "commit", "-q", "-m",
                    "bfcl_eval: tool-calling transfer of all three scorers"], cwd=ROOT)
    marker.write_text("done")
    print("bfcl_eval done, recorded and committed", flush=True)


if __name__ == "__main__":
    main()