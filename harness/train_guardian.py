#!/usr/bin/env python3
"""Guardian for the long QLoRA run: thermal safety + crash auto-resume.

Runs on the HOST (needs `sensors`, `sudo`, `sudo podman`). Adopts the training in
container `jevtrain`; never starts a duplicate.

ROOT CAUSE of CPU heat under GPU training (measured): the training loop keeps ~2
CPU cores pinned (~90% each) on host-side kernel dispatch + host<->device sync;
Zen4 then boosts to its 95C Tjmax and spikes past it. The training is GPU-bound,
so capping CPU max frequency removes the heat at ~5% throughput cost (measured
100C -> 74C). That is the primary control here.

THERMAL. Applies CPU freq cap + GPU power cap at start. Reads discrete GPU
(amdgpu junction/mem) + CPU (k10temp) every POLL_S. On danger: stop, cool down,
and MITIGATE by lowering the CPU freq cap another notch before resuming, so it
self-converges on a hot day. Only gives up if it still trips at the freq floor.

RESUME. Checkpoints every --save-steps to the host-mounted out dir. Relaunch
passes --resume only when a checkpoint exists. Guardian exits on completion.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

CONT = "jevtrain"
POLL_S = 20
GPU_JUNCTION_DANGER = 106.0    # crit 110
GPU_MEM_DANGER = 104.0         # crit 108
CPU_DANGER = 99.0              # Tjmax 95; >99 = past ceiling
COOL_GPU_JUNCTION = 80.0
COOL_CPU = 80.0
COOL_TIMEOUT_S = 300
GPU_POWER_CAP_W = 198          # card floor
CPU_FREQ_START_KHZ = 4200000   # 4.2 GHz
CPU_FREQ_STEP_KHZ = 400000     # lower 0.4 GHz per repeated trip
CPU_FREQ_FLOOR_KHZ = 3000000   # 3.0 GHz floor
FLOOR_TRIP_LIMIT = 3           # trips AT the floor before giving up
ENV = "OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive HSA_ENABLE_INTERRUPT=1"
GPU_HWMON = "/sys/class/drm/card1/device/hwmon/hwmon1/power1_cap"


def now():
    return datetime.now().strftime("%H:%M:%S")


def glog(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open("results/guardian.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def set_cpu_freq(khz):
    sh(f"for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq; do "
       f"echo {khz} | sudo tee $c >/dev/null; done")
    glog(f"CPU max freq cap -> {khz//1000} MHz")


def set_gpu_power(watts):
    sh(f"echo {watts*1000000} | sudo tee {GPU_HWMON} >/dev/null")
    glog(f"GPU power cap -> {watts} W")


GPU_HWMON_DIR = "/sys/class/drm/card1/device/hwmon/hwmon1"


def set_gpu_fan_max():
    sh(f"echo 1 | sudo tee {GPU_HWMON_DIR}/pwm1_enable >/dev/null; "
       f"echo 255 | sudo tee {GPU_HWMON_DIR}/pwm1 >/dev/null")
    glog("GPU fan -> max (manual)")


PERF_LEVEL = "/sys/class/drm/card1/device/power_dpm_force_performance_level"
SCLK_MASK = "/sys/class/drm/card1/device/pp_dpm_sclk"
SCLK_LEVEL = "2"  # DPM level 2 (~1927MHz) vs level 1 max (~2506MHz): ~23% lower


def ensure_clock_capped():
    """Keep the GPU at full-clock 'auto' (fast) while preventing the DPM 'low'
    stall. Set GUARDIAN_GPU_DESKTOP=1 to instead pin a reduced manual clock that
    leaves headroom for an interactive desktop (slower, no frame-skipping)."""
    import os
    desktop = os.environ.get("GUARDIAN_GPU_DESKTOP") == "1"
    cur = sh(f"cat {PERF_LEVEL}").stdout.strip()
    if desktop:
        if cur != "manual":
            sh(f"echo manual | sudo tee {PERF_LEVEL} >/dev/null")
            sh(f"echo {SCLK_LEVEL} | sudo tee {SCLK_MASK} >/dev/null")
            glog(f"GPU clock capped: manual sclk level {SCLK_LEVEL} (~1927MHz) for desktop headroom")
    elif cur and cur != "auto":
        sh(f"echo auto | sudo tee {PERF_LEVEL} >/dev/null")
        glog(f"GPU perf level was '{cur}' -> auto (full clock)")


def read_temps():
    r = sh("sensors -j", timeout=15)
    out = {"gpu_junction": None, "gpu_mem": None, "gpu_edge": None, "cpu": None}
    try:
        d = json.loads(r.stdout)
    except Exception:
        return out
    for chip, v in d.items():
        if not isinstance(v, dict):
            continue
        if chip.startswith("amdgpu") and any("junction" in f for f in v):
            for f, fv in v.items():
                if not isinstance(fv, dict):
                    continue
                key = "junction" if "junction" in f else ("mem" if "mem" in f else ("edge" if "edge" in f else None))
                if key:
                    val = next((x for k, x in fv.items() if k.endswith("_input")), None)
                    if val is not None:
                        out["gpu_" + key] = float(val)
        elif chip.startswith("k10temp"):
            hot = None
            for f, fv in v.items():
                if isinstance(fv, dict) and (f.startswith("Tctl") or f.startswith("Tccd")):
                    val = next((x for k, x in fv.items() if k.endswith("_input")), None)
                    if val is not None:
                        hot = val if hot is None else max(hot, val)
            out["cpu"] = hot
    return out


def container_up():
    return sh(f"sudo podman inspect -f '{{{{.State.Running}}}}' {CONT}").stdout.strip() == "true"


def has_checkpoint(out):
    return sh(f"sudo podman exec {CONT} bash -c 'ls -d {out}/checkpoint-* 2>/dev/null | head -1'").stdout.strip() != ""


def training_running():
    r = sh(f"sudo podman exec {CONT} pgrep -f qlora_train.py")
    return r.returncode == 0 and r.stdout.strip() != ""


def training_complete(out):
    return sh(f"sudo podman exec {CONT} test -f {out}/train_args.json").returncode == 0


def ensure_container():
    if not container_up():
        glog(f"container {CONT} down -> starting")
        sh(f"sudo podman start {CONT}")
        time.sleep(5)


def launch(base, out, corpus, log):
    ensure_container()
    resume = "--resume" if has_checkpoint(out) else ""
    cmd = (f"{ENV} python harness/qlora_train.py --base {base} --corpus {corpus} "
           f"--out {out} --epochs 1 --grad-accum 16 --save-steps 50 --max-len 1536 "
           f"{resume} > {log} 2>&1")
    glog(f"launch ({'resume' if resume else 'fresh'}): {base} -> {out}")
    sh(f"sudo podman exec -d {CONT} bash -c '{cmd}'")
    time.sleep(8)


def kill_training():
    sh(f"sudo podman exec {CONT} pkill -f qlora_train.py")
    time.sleep(5)


def gpu_danger(t):
    return (t["gpu_junction"] or 0) >= GPU_JUNCTION_DANGER or (t["gpu_mem"] or 0) >= GPU_MEM_DANGER


def cpu_danger(t):
    return (t["cpu"] or 0) >= CPU_DANGER


def danger(t):
    return gpu_danger(t) or cpu_danger(t)


def cooled(t):
    return ((t["gpu_junction"] or 0) <= COOL_GPU_JUNCTION and (t["cpu"] or 0) <= COOL_CPU)


def cooldown():
    glog("cooldown: training stopped")
    t0 = time.time()
    while time.time() - t0 < COOL_TIMEOUT_S:
        time.sleep(15)
        t = read_temps()
        glog(f"  cooling gpu_j={t['gpu_junction']} gpu_mem={t['gpu_mem']} cpu={t['cpu']}")
        if cooled(t):
            return
    glog("  cooldown timeout, proceeding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default="results/jevlike4b_lora")
    ap.add_argument("--corpus", default="data/mixed/train.jsonl")
    ap.add_argument("--log", default="results/train4b.log")
    args = ap.parse_args()

    glog(f"guardian start base={args.base} out={args.out}")
    cpu_freq = CPU_FREQ_START_KHZ
    set_cpu_freq(cpu_freq)
    set_gpu_power(GPU_POWER_CAP_W)
    ensure_clock_capped()
    floor_trips = 0

    if not training_running() and not training_complete(args.out):
        launch(args.base, args.out, args.corpus, args.log)

    while True:
        if training_complete(args.out):
            glog("training complete - guardian exiting")
            return 0
        ensure_clock_capped()
        t = read_temps()
        run = training_running()
        glog(f"gpu_j={t['gpu_junction']} gpu_mem={t['gpu_mem']} gpu_edge={t['gpu_edge']} "
             f"cpu={t['cpu']} cpu_cap={cpu_freq//1000}MHz training={'up' if run else 'DOWN'}")
        if danger(t):
            glog(f"*** DANGER {t} -> stopping ***")
            kill_training()
            if gpu_danger(t):
                set_gpu_fan_max()  # correct lever for GPU temp; power already at floor
            if cpu_danger(t):
                if cpu_freq > CPU_FREQ_FLOOR_KHZ:
                    cpu_freq = max(CPU_FREQ_FLOOR_KHZ, cpu_freq - CPU_FREQ_STEP_KHZ)
                    set_cpu_freq(cpu_freq)
                else:
                    floor_trips += 1
                    glog(f"already at freq floor; floor trip {floor_trips}/{FLOOR_TRIP_LIMIT}")
                    if floor_trips >= FLOOR_TRIP_LIMIT:
                        glog("*** persistent CPU overheat at freq floor: STOPPING for a human. "
                             f"Checkpoints intact at {args.out}. ***")
                        return 2
            cooldown()
            launch(args.base, args.out, args.corpus, args.log)
        elif not run and not training_complete(args.out):
            glog("training died (temps OK) -> resuming")
            launch(args.base, args.out, args.corpus, args.log)
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
