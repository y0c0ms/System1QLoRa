#!/usr/bin/env python3
"""Autonomous benchmark loop: watches each stage to completion, records the
result, then moves on to the next stage without human intervention.

Stages (the handoff's priority order, all CPU-only, all on the SNI benchmark):

  1. SNI fit (head at final layer)  -> record + run the OLD-corpus scorer on
                                       SNI test (transfer vs memorisation comma
                                       parator)
  2. head-8 fit (head at layer 8)   -> record; the layer-fix test: does the
                                       measured best layer beat the final layer
                                       on held-out predicates too?

Each stage: WAIT for its metrics.json (or resume from the last --save-every
checkpoint on crash, MAX_RESUMES attempts) -> FINISH (append FINDINGS entry,
commit) -> next stage. Idempotent: state is re-derived from artifacts, so a
crash of the watcher itself loses nothing - restart it and it picks up.

Supervised by `hub start` so it survives and restarts on failure.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = ROOT / ".venv/bin/python"
DATA = "data/sni_ds"
BASE = "mlx-community/gemma-3-270m-bf16"
LOOP_LOG = ROOT / "results/bench_loop.log"
MAX_RESUMES = 2

SNI_OUT = ROOT / "results/sni_fit"
H8_OUT = ROOT / "results/head8_fit"

SNI_CMD = [str(PY), "upstream/system_one.py", "train",
           "--base-model", BASE, "--local-data-dir", DATA,
           "--out-dir", str(SNI_OUT), "--batch-size", "4", "--max-options", "8",
           "--max-len", "128", "--epochs", "1", "--lr", "1e-4",
           "--save-every", "500", "--eval-test"]

HEAD8_CMD = [str(PY), "harness/head8_fit.py",
             "--base-model", BASE, "--local-data-dir", DATA,
             "--out-dir", str(H8_OUT), "--batch-size", "4", "--max-options", "8",
             "--max-len", "128", "--epochs", "1", "--lr", "1e-4",
             "--save-every", "300", "--eval-test", "--eval-limit", "0"]

TRANSFER_CMD = [str(PY), "upstream/system_one.py", "eval",
                "--base-model", BASE, "--model-dir", "upstream/pretrained-scorer",
                "--local-data-dir", DATA, "--split", "test"]
TRANSFER_LOG = ROOT / "results/sni_transfer_eval.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOOP_LOG, "a") as f:
        f.write(line + "\n")


def stage_alive(out_dir):
    # match the out-dir basename so it catches both the relative and absolute
    # --out-dir spellings the training commands use
    out = subprocess.run(["pgrep", "-f", out_dir.name], capture_output=True, text=True)
    return bool(out.stdout.strip())


def resume_count(out_dir):
    p = ROOT / f"results/.resume_{out_dir.name}"
    return int(p.read_text().strip()) if p.exists() else 0


def bump_resume(out_dir):
    (ROOT / f"results/.resume_{out_dir.name}").write_text(str(resume_count(out_dir) + 1))


def stage_done(out_dir):
    return (out_dir / "metrics.json").exists()


def record_sni_fit():
    import harness.endings as E  # shared FINDINGS rendering
    m = json.loads((SNI_OUT / "metrics.json").read_text())
    unc, cal, txt = run_transfer_comparator()
    E.append_sni(m)
    E.append_transfer(cal)
    return m


def record_head8():
    import harness.endings as E
    m = json.loads((H8_OUT / "metrics.json").read_text())
    E.append_head8(m)


def parse_blocks(path):
    """Extract UNCALIBRATED/CALIBRATED json blocks from a run log. system_one
    prints them as json.dumps(..., indent=1) - MULTI-LINE - so a plain
    load-next-line parse breaks (observed: JSONDecodeError 'Extra data' crashed
    the loop; the transfer comparator had to re-run)."""
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return {}
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


def run_transfer_comparator():
    # the eval is the expensive part (~25 min): if it already ran, reuse its
    # log so a crash in the record phase never re-executes it
    if (ROOT / "results/.transfer_done").exists():
        b = parse_blocks(TRANSFER_LOG)
        log("transfer comparator: reusing completed log")
        return b.get("unc"), b.get("cal"), TRANSFER_LOG.read_text()
    log("transfer comparator: upstream/pretrained-scorer on SNI test")
    with open(TRANSFER_LOG, "w") as f:
        subprocess.run(TRANSFER_CMD, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    (ROOT / "results/.transfer_done").write_text("done")
    b = parse_blocks(TRANSFER_LOG)
    return b.get("unc"), b.get("cal"), TRANSFER_LOG.read_text()


def commit(msg):
    subprocess.run(["git", "add", "docs/FINDINGS.md"], cwd=ROOT)
    for d in (SNI_OUT, H8_OUT):
        if (d / "metrics.json").exists():
            subprocess.run(["git", "add", str(d / "metrics.json")], cwd=ROOT)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT)
    log("committed: " + msg)


def resume_stage(out_dir, cmd):
    if not (out_dir / "adapter_model.safetensors").exists():
        log(f"CRASH: {out_dir.name} has no checkpoint to resume from")
        return False
    rcmd = list(cmd) + ["--resume-adapter", str(out_dir)]
    log(f"resuming {out_dir.name}: " + " ".join(rcmd))
    subprocess.Popen(rcmd, cwd=ROOT, stdout=open(OUTLOG(out_dir), "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    return True


def OUTLOG(out_dir):
    return ROOT / f"results/{out_dir.name}.log"


def watch_stage(stage):
    name, out_dir, cmd, record = stage["name"], stage["out"], stage["cmd"], stage["record"]
    marker = ROOT / f"results/.stage_{name}_done"
    log(f"stage {name}: watching {out_dir}")
    if stage_done(out_dir) and marker.exists():
        log(f"stage {name}: already recorded - skipping")
        return True
    if not stage_done(out_dir):
        while not stage_done(out_dir):
            if not stage_alive(out_dir):
                if resume_count(out_dir) < MAX_RESUMES:
                    bump_resume(out_dir)
                    if resume_stage(out_dir, cmd):
                        log(f"stage {name}: resumed; continuing to watch")
                        time.sleep(60)
                        continue
                log(f"CRASH: {name} died without metrics.json and resume exhausted")
                return False
            time.sleep(60)
    log(f"stage {name}: done - recording")
    record()
    marker.write_text("done")
    commit(f"bench_loop: record {name} result")
    return True


def bfcl_stage():
    """Build the BFCL tool-calling benchmark (if needed) and eval all scorers.

    Idempotent via results/.bfcl_done; bounded retries; runs AFTER the head-8
    fit so there is never CPU contention between stages."""
    if (ROOT / "results/.bfcl_done").exists():
        log("stage bfcl: already done - skipping")
        return True
    if not (ROOT / "data/bfcl/manifest.json").exists():
        log("stage bfcl: building the BFCL eval benchmark")
        try:
            subprocess.run([str(PY), "harness/build_bfcl.py"], cwd=ROOT, check=True,
                           timeout=1800)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log(f"stage bfcl: build failed: {e!r}")
            return False
    for attempt in range(2):
        try:
            subprocess.run([str(PY), "harness/bfcl_eval.py"], cwd=ROOT, check=True,
                           timeout=4 * 3600)
            return (ROOT / "results/.bfcl_done").exists()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log(f"stage bfcl: eval attempt {attempt + 1} failed: {e!r}")
    return False


JEV_DIR = ROOT / "upstream/jevlike"
JEV_PT = ROOT / "results/jevlike_sni.pt"


def jevlike_stage():
    """Third architecture, same rows: option-as-query attention scorer.

    jevlike (github.com/vinnylarouge/jevlike) scores a context against runtime
    options with a dedicated attention head - no decoder. Tiny byte-encoder
    variant is the feasible CPU track (~5 min/epoch measured). The frozen
    Qwen2.5-0.5B encoder variant measured ~7s/row on this contended box (~30h
    for 16k rows) - parked, documented in FINDINGS, not blocked.
    """
    if (ROOT / "results/.jevlike_done").exists():
        log("stage jevlike: already done - skipping")
        return True
    if not (ROOT / "data/jevlike/sni_train.jsonl").exists():
        log("stage jevlike: building jsonl rows")
        try:
            subprocess.run([str(PY), "harness/build_jevlike_data.py"], cwd=ROOT,
                           check=True, timeout=900)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log(f"stage jevlike: data build failed: {e!r}")
            return False
    variants = [
        ("base", [], "results/jevlike_sni.pt"),
        ("big", ["--width", "128", "--rank", "256", "--epochs", "12",
                 "--option-tokens", "64"], "results/jevlike_sni_big.pt"),
    ]
    evals, cal = {}, {}
    for tag, extra, pt in variants:
        pt = ROOT / pt
        train = [str(PY), "-m", "jevlike.train",
                 str(ROOT / "data/jevlike/sni_train.jsonl"),
                 "--validation", str(ROOT / "data/jevlike/sni_val.jsonl"),
                 "--output", str(pt), "--encoder", "tiny",
                 "--epochs", "8", "--batch-size", "64",
                 "--context-tokens", "512", "--device", "cpu"] + extra
        ok = False
        for attempt in range(2):
            try:
                subprocess.run(train, cwd=JEV_DIR, check=True, timeout=3 * 3600)
                ok = True
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                log(f"stage jevlike: {tag} train attempt {attempt + 1} failed: {e!r}")
        if not ok:
            return False
        for split in ("sni_test", "bfcl_test", "bfcl_irr"):
            p = ROOT / f"data/jevlike/{split}.jsonl"
            if not p.exists():
                continue
            r = subprocess.run([str(PY), "-m", "jevlike.eval", str(pt), str(p),
                                "--device", "cpu"], cwd=JEV_DIR,
                               capture_output=True, text=True)
            try:
                evals[f"{tag}_{split}"] = json.loads(r.stdout)
                log(f"stage jevlike: {tag} {split} "
                    + json.dumps(evals[f"{tag}_{split}"]["model"]))
            except json.JSONDecodeError:
                log(f"stage jevlike: {tag} eval output unparseable for {split}")
        try:
            r = subprocess.run(
                [str(PY), "harness/jevlike_cal.py", str(pt),
                 "--val-jsonl", "data/jevlike/sni_val.jsonl",
                 "--val-jsonl", "data/jevlike/bfcl_val.jsonl",
                 "--test-jsonl", "data/jevlike/sni_test.jsonl",
                 "--test-jsonl", "data/jevlike/bfcl_test.jsonl",
                 "--test-jsonl", "data/jevlike/bfcl_irr.jsonl",
                 "--labels", "sni", "--labels", "bfcl", "--device", "cpu"],
                cwd=ROOT, capture_output=True, text=True)
            cal[tag] = json.loads(r.stdout)
            log(f"stage jevlike: {tag} calibrated+latency ok")
        except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
            log(f"stage jevlike: {tag} calibration harness failed: {e!r}")
    import harness.endings as E
    E.append_jevlike(evals, cal, variants=variants)
    commit("bench_loop: record jevlike stage")
    (ROOT / "results/.jevlike_done").write_text("done")
    log("stage jevlike: done, recorded and committed")
    return True


def laya_stage():
    """Open 421M ModernBERT System-1 engine (Apache-2.0), zero extra training.

    Laya claims 0.766 on the typed-decisions corpus we measure - reproduce that
    and, more importantly, run it on SNI held-out predicates + BFCL with our
    protocol (T from val). Its own docs admit zero-shot ~0.35 and option-token
    budget limits, so the held-out number is genuinely open."""
    if (ROOT / "results/.laya_done").exists():
        log("stage laya: already done - skipping")
        return True
    ok = False
    for attempt in range(2):
        try:
            subprocess.run([str(PY), "harness/laya_eval.py",
                            "--checkpoint", "convaiinnovations/laya-typed-decisions",
                            "--device", "cpu", "--out", "results/laya_eval.json"],
                           cwd=ROOT, check=True, timeout=4 * 3600)
            ok = (ROOT / "results/laya_eval.json").exists()
            if ok:
                break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log(f"stage laya: attempt {attempt + 1} failed: {e!r}")
    if not ok:
        return False
    import harness.endings as E
    import json as _json
    E.append_laya(_json.loads((ROOT / "results/laya_eval.json").read_text()))
    commit("bench_loop: record laya stage")
    (ROOT / "results/.laya_done").write_text("done")
    log("stage laya: done, recorded and committed")
    return True


def gate_stage():
    """Record the ONE legitimate stop: xlam train data is gated click-through.
    Everything else on the CPU track is done."""
    import harness.endings as E
    E.append_gate()
    commit("bench_loop: record xlam gate (train-side tool data needs HF click-through)")
    return True


def sys_eval(model_dir, ds, split):
    """system_one eval of an adapter on a small split; returns calibrated ALL."""
    logp = ROOT / f"results/ref_{Path(model_dir).name}_{Path(ds).name}_{split}.log"
    cmd = [str(PY), "upstream/system_one.py", "eval", "--base-model", BASE,
           "--model-dir", model_dir, "--local-data-dir", ds, "--split", split]
    with open(logp, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=2 * 3600)
    return parse_blocks(logp).get("cal", {}).get("ALL", {})


def referee_stage():
    """Fast no-training scoreboard on the small test: released corpus scorer,
    our SNI-LM fit, and Laya - identical held-out rows, uniform T-on-val."""
    if (ROOT / "results/.referee_done").exists():
        log("stage referee: already done - skipping")
        return True
    import harness.endings as E
    laya = {}
    try:
        subprocess.run([str(PY), "harness/laya_eval.py", "--checkpoint",
                        "convaiinnovations/laya-typed-decisions", "--device", "cpu",
                        "--out", "results/laya_eval.json"], cwd=ROOT, check=True,
                       timeout=4 * 3600)
        laya = json.loads((ROOT / "results/laya_eval.json").read_text())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError) as e:
        log(f"stage referee: laya eval failed: {e!r}")
    board = {}
    for tag, md in (("old", "upstream/pretrained-scorer"), ("sni_lm", str(SNI_OUT))):
        board[tag] = {
            "sni_test": sys_eval(md, "data/sni_ds", "test"),
            "bfcl_test": sys_eval(md, "data/bfcl_ds", "test"),
            "bfcl_irr": sys_eval(md, "data/bfcl_ds", "irr"),
        }
        log(f"stage referee: {tag} sni_test " + json.dumps(board[tag]["sni_test"]))
    E.append_referee(board, laya)
    commit("bench_loop: referee scoreboard (old + sni-LM + laya, small held-out)")
    (ROOT / "results/.referee_done").write_text("done")
    return True


def head8_stage():
    """Layer-8 head fit (the build experiment), blocking with bounded resume."""
    if (ROOT / "results/.head8_done").exists():
        log("stage head8: already done - skipping")
        return True
    ok = False
    for attempt in range(3):
        extra = (["--resume-adapter", str(H8_OUT)]
                 if (H8_OUT / "adapter_model.safetensors").exists() else [])
        try:
            subprocess.run(HEAD8_CMD + extra, cwd=ROOT, check=True, timeout=6 * 3600)
            if (H8_OUT / "metrics.json").exists():
                ok = True
                break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log(f"stage head8: attempt {attempt + 1} failed: {e!r}")
    if not ok:
        return False
    record_head8()
    commit("bench_loop: record head8 result")
    (ROOT / "results/.head8_done").write_text("done")
    return True


def main():
    log("bench_loop starting (GPU policy: CPU-only track; the GPU is llama-swap's: "
        "qwen3.6-35b + translategemma-12b + whisper-gpu share 16 GiB, operator STT/translation)")
    for name, fn in (("referee", referee_stage), ("head8", head8_stage),
                     ("jevlike", jevlike_stage), ("gate", gate_stage)):
        if not fn():
            log(f"bench_loop: aborting at stage {name}")
            return 1
    log("bench_loop: ALL STAGES COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())