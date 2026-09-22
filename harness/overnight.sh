#!/usr/bin/env bash
# Overnight definitive-model run. Runs INSIDE the jevtrain container.
#
# PATH (decided from diagnosis): the only remaining SNI->Jev gap is NLI, and it
# is a FORMAT gap - the held-out SNI NLI tasks are "select which of 3 candidates
# has relation R", not single-pair label classification. data/mixed_final adds
# format-matched select-of-3 NLI (build_nli_select.py) on top of the v3 recipe
# (absence-augmentation + upweight). This run trains the definitive 4B and the
# 0.6B production model on that corpus, at a proper ~1.25 epochs (prior runs were
# 400 steps ~= 0.3 epoch - undertrained), and benchmarks both on the full board.
#
# SAFETY: batch 4 x seq 384 caps VRAM ~7-11 GB (leaves >=5 GB for the desktop, so
# no compositor starvation / artifacts even if someone returns to the machine).
# Every phase is sentinel-gated and every training checkpoints (--save-steps) and
# resumes (--resume) so an interruption loses at most the steps since the last
# checkpoint (AGENTS.md rule 9).
set -u
cd /workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
LOG=results/overnight_log.txt
CORPUS=data/mixed_final/train.jsonl
B06=Qwen/Qwen3-0.6B
B4B=Qwen/Qwen3-4B-Instruct-2507
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

train(){ # base out max_steps
  local base=$1 out=$2 steps=$3
  if [ -f "$out/train_args.json" ]; then say "SKIP train $out (done)"; return 0; fi
  local resume=""; ls "$out"/checkpoint-* >/dev/null 2>&1 && resume="--resume"
  say "TRAIN $out  base=$base steps=$steps $resume"
  python harness/qlora_train.py --base "$base" --corpus "$CORPUS" --out "$out" \
    --batch-size 4 --grad-accum 6 --max-len 384 --max-steps "$steps" \
    --save-steps 400 --epochs 2 $resume >>"$LOG" 2>&1 \
    && say "TRAIN $out DONE" || say "TRAIN $out FAILED (rc=$?)"
}

evalb(){ # base adapter out
  local base=$1 adapter=$2 out=$3
  if [ -f "$out" ]; then say "SKIP eval $out (done)"; return 0; fi
  say "EVAL $adapter -> $out"
  python harness/hf_score.py --base "$base" --adapter "$adapter" \
    --datasets sni reflex bfcl bfcl_irr --out "$out" >>"$LOG" 2>&1 \
    && say "EVAL $out DONE" || say "EVAL $out FAILED (rc=$?)"
  grep -E "^\[(sni|reflex|bfcl|bfcl_irr)\]" "$LOG" | tail -4 | tee -a "$LOG"
}

say "===== OVERNIGHT START  corpus=$CORPUS ($(wc -l <"$CORPUS") rows) ====="

# Phase 1: definitive 4B (production quality), ~1.25 epoch on 38k @ eff-batch 24
train "$B4B" results/q4b_final_lora 2000
# Phase 2: full board for the 4B
evalb "$B4B" results/q4b_final_lora results/q4b_final_eval.json
# Phase 3: 0.6B production model (fast-inference ship), same corpus/length
train "$B06" results/q06_final_lora 2000
# Phase 4: full board for the 0.6B
evalb "$B06" results/q06_final_lora results/q06_final_eval.json

say "===== OVERNIGHT DONE ====="
say "4B  : $(grep -E '^\[' "$LOG" | grep -A4 q4b_final 2>/dev/null | tail -4 | tr '\n' ' ')"
python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import json
for tag,p in [("4B final","results/q4b_final_eval.json"),("0.6B final","results/q06_final_eval.json")]:
    try:
        d=json.load(open(p))["datasets"]
        print(tag, {k:round(v["acc_all"],3) for k,v in d.items()})
    except Exception as e:
        print(tag,"(pending)",e)
PY
