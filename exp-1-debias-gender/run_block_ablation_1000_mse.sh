#!/usr/bin/env bash
# ==========================================================================
# Block-ablation experiments for the h-space (image-semantics preserving) loss.
#   --- MSE + standard-norm variant, 1000 training steps ---
#
#   3 runs total:
#     1) mid+up   block combo  +  realistic-face loss 4
#     2) mid+down block combo  +  realistic-face loss 4
#     3) mid+down block combo  plain (realistic-face loss 0)
#     loss config : MSE + standard-norm   (--h_loss_form raw  --h_block_norm std)
#
#   Same MSE+std setup as group 1 of run_block_ablation_160.sh, but:
#     - only the mid+up and mid+down block combos
#     - trained to 1000 steps (not 160)
#     - image-semantics loss weight fixed to 8 (--weight_loss_img 8)
#
#   block tokens "up,mid" / "down,mid" match the earlier 160-step MSE runs
#   (2/8 and 3/8), so the output folder tags (_hBlk-up+mid / _hBlk-down+mid)
#   line up for direct comparison.
#
# ---- the 1000-step cap ----
#   parse_args() applies the --config YAML *after* the CLI flags, so a CLI
#   --max_train_steps would be silently overridden by the YAML. The cap is
#   therefore enforced via configs/debias-text-encoder3_1000.yaml
#   (max_train_steps: 1000). That config intentionally OMITS weight_loss_img
#   so $WIMG below (CLI) takes effect.
#
# ---- checkpoints / eval at 1000 steps ----
#   CKPT_LONG=200 -> permanent checkpoints at 200/400/600/800/1000
#   (checkpoint-1000 is the final model; earlier ones let you pick the best).
#   Set CKPT_LONG=1000 to keep ONLY the final checkpoint (less disk).
#   EVAL_EVERY=200 -> in-run eval at 200/400/600/800/1000. Raise it (e.g. 1000)
#   to eval only at the end.
# ==========================================================================

set -u
cd "$(dirname "$0")"

ACC_CFG="configs/accelerate_config3.yaml"        # 3 GPUs (num_processes: 3)
PY="1-main-gender-sgd_dmscr_h_gen_check_block_ablation.py"
CFG="configs/debias-text-encoder3_1000.yaml"     # 3-GPU batch sizes, max_train_steps: 1000

# ---- knobs shared by all runs (tweak freely) ----
WIMG=8            # weight_loss_img (config omits it, so this CLI value wins)
SKIP_PCT=0       # skip_final_steps_pct
EVAL_EVERY=200    # eval cadence -> in-run eval at 200/400/600/800/1000
CKPT_LONG=200     # permanent checkpoints at 200/400/600/800/1000 (incl. final)

run () {
  local tag="$1"; local blocks="$2"; local form="$3"; local norm="$4"; local face="$5"
  echo "==================================================================="
  echo ">>> [$tag]  h_block_names=$blocks  h_loss_form=$form  h_block_norm=$norm  faceRealistic=$face  (max_train_steps=1000, wImg=$WIMG)"
  echo "==================================================================="
  accelerate launch --config_file "$ACC_CFG" \
    "$PY" --config "$CFG" \
    --region_mask_mode attn \
    --skip_final_steps_pct "$SKIP_PCT" \
    --weight_loss_face_realistic "$face" \
    --weight_loss_img "$WIMG" \
    --h_block_names "$blocks" \
    --h_loss_form "$form" \
    --h_block_norm "$norm" \
    --evaluate_every_n_iter "$EVAL_EVERY" \
    --checkpointing_steps_long "$CKPT_LONG" \
    --save_attmaps
}

# ---- MSE + standard-norm  (raw / std), 1000 steps, wImg=8 ----
run "1/3  MSE+std  mid+up    faceReal=4" "up,mid"   raw std 4
run "2/3  MSE+std  mid+down  faceReal=4" "down,mid" raw std 4
run "3/3  MSE+std  mid+down  faceReal=0" "down,mid" raw std 0

echo "All 3 MSE+std (1000-step, wImg=8) block-ablation runs finished."
