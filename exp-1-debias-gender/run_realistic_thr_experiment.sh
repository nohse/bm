#!/usr/bin/env bash
# ============================================================================
# Realistic-face-loss face/no-face threshold experiment.
#   1) accelerate (2 procs on GPU 2,3) generates images from checkpoint-200,
#      labels each with get_face, measures the realistic-face SDS loss + attn
#      hard-mask fraction, writes per-rank shards. No training.
#   2) realistic_thr_analyze.py turns the shards into plots / tables / summary.
#
# Usage:
#   ./run_realistic_thr_experiment.sh                 # full run (defaults below)
#   RTE_TARGET=3 RTE_POOL_CAP=48 RTE_R=2 ./run_realistic_thr_experiment.sh   # smoke
# ============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

RUN="outputs/gender_aaai/20260707-1144_gender_aaai_region-attn_skip-50pct_wImg-4_wRealFace-4.0_Th-0.2_lr-5e-05"
CKPT="$RUN/ckpts/checkpoint-200"
OUT="${RTE_OUT:-$RUN/realistic_thr_experiment_ckpt200_skip0}"

# knobs (env-overridable for smoke tests)
TARGET="${RTE_TARGET:-25}"
POOL_CAP="${RTE_POOL_CAP:-2400}"
R="${RTE_R:-5}"
STEPS="${RTE_STEPS:-25}"
SKIP_PCT="${RTE_SKIP_PCT:-0}"
BATCH="${RTE_BATCH:-8}"
GPUS="${RTE_GPUS:-2,3}"
NPROC="${RTE_NPROC:-2}"
PORT="${RTE_PORT:-29777}"

export WANDB_MODE=offline
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false

echo "[RTE] CKPT=$CKPT"
echo "[RTE] OUT=$OUT  target/class=$TARGET pool_cap=$POOL_CAP R=$R steps=$STEPS skip=$SKIP_PCT gpus=$GPUS nproc=$NPROC"

accelerate launch \
  --num_processes "$NPROC" --num_machines 1 --mixed_precision fp16 \
  --gpu_ids "$GPUS" --main_process_port "$PORT" \
  chekc_SCRclip_attmap_grad.py \
  --resume_from_checkpoint "$CKPT" \
  --realistic_thr_experiment \
  --rte_out_dir "$OUT" \
  --rte_target_per_class "$TARGET" \
  --rte_pool_cap "$POOL_CAP" \
  --rte_realistic_repeats "$R" \
  --rte_num_denoise "$STEPS" \
  --rte_skip_pct "$SKIP_PCT" \
  --rte_batch "$BATCH" \
  --rte_weights ema \
  --rte_seed 1234
rc=$?
echo "[RTE] accelerate exit code: $rc"
if [ $rc -ne 0 ]; then
  echo "[RTE] generation failed; skipping analysis."
  exit $rc
fi

echo "[RTE] running analysis on $OUT"
python realistic_thr_analyze.py "$OUT"
echo "[RTE] done. See $OUT/analysis/"
