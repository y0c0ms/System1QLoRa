#!/usr/bin/env python3
"""Head-to-head on the reflex test set: prompted 35B vs trained 270M, plus the
SNI-trained scorer zero-shot (held-out domain). Reads the finished artifacts;
runs the one missing eval (SNI scorer on reflex) itself."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
BASE = "mlx-community/gemma-3-270m-bf16"


def sni_zero_shot():
    log = ROOT / "results/reflex_sni_zeroshot.log"
    cmd = [str(PY), "upstream/system_one.py", "eval", "--base-model", BASE,
           "--model-dir", str(ROOT / "results/sni_fit"),
           "--local-data-dir", "data/reflex_ds", "--split", "test"]
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    lines = log.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("CALIBRATED"):
            acc = ""
            for j in range(i, len(lines)):
                acc += lines[j][len("CALIBRATED "):] if j == i else "\n" + lines[j]
                try:
                    return json.loads(acc)
                except json.JSONDecodeError:
                    continue
    return {}


def main():
    m = json.loads((ROOT / "results/reflex_fit/metrics.json").read_text())
    tc = m.get("test_calibrated", {})
    base = json.loads((ROOT / "results/baseline_reflex_q36.json").read_text())["test"]
    sni = sni_zero_shot()

    def g(d, task, default="—"):
        e = (d or {}).get(task, {})
        return f"{e.get('acc'):.3f}" if e.get("acc") is not None else default

    tasks = ["math_topic", "math_level", "code_defect", "ALL"]
    floors = {"math_topic": 0.143, "math_level": 0.20, "code_defect": 0.50, "ALL": "-"}
    print("\n=== REFLEX HEAD-TO-HEAD (calibrated test accuracy) ===")
    print(f"{'task':<12} {'floor':>6} {'35B-prompt':>11} {'SNI-zeroshot':>13} {'270M-trained':>13}")
    for t in tasks:
        fl = floors[t]
        fl = f"{fl:.3f}" if isinstance(fl, float) else fl
        print(f"{t:<12} {fl:>6} {g(base, t):>11} {g(sni, t):>13} {g(tc, t):>13}")
    print(f"\n270M-trained T={m.get('temperature')} steps={m.get('steps')} "
          f"{round(m.get('train_seconds', 0)/60,1)}min")


if __name__ == "__main__":
    main()