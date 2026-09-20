#!/usr/bin/env bash
# Launch the ROCm container with the RX 7900 GRE passed through and the project
# mounted at /workspace. gfx1100 is officially supported (no HSA override needed).
# renderD128 is the discrete card; the Raphael iGPU (renderD129) is hidden so HIP
# never picks it. Usage: harness/rocm_shell.sh [command...]
set -euo pipefail
IMG=docker.io/rocm/pytorch:latest
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec podman run --rm -it \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add keep-groups \
  --ipc=host --shm-size 8g \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -e HF_HOME=/workspace/.hf_cache \
  -v "$PROJ":/workspace -w /workspace \
  "$IMG" "$@"
