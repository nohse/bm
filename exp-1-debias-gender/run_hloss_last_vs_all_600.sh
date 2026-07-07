#!/usr/bin/env bash
# ==========================================================================
# Controlled comparison: h-space (image-preservation) loss on the MID block,
# computed with 8x image-loss weight, over 600 training steps.
#
#   RUN 1  "last"  : check_last_cos.py   --h_loss_form mse
#                    -> MSE(mid_block) at ONLY the LAST executed denoising step
#   RUN 2  "all"   : 1-main-...-block_ablation.py
#                    --h_loss_form raw --h_block_names mid --h_block_norm none
#                    -> mean over ALL executed steps of the SAME per-step MSE(mid_block)
#
#   The two per-step MSE terms are computed by identical code on the identical
#   module (unet.mid_block); "raw" (block_ablation) == "mse" (check_last_cos).
#   So the ONLY intended difference is: last step  vs  mean over all ~19-23 steps.
#   (num_denoising_steps is randomly 19-23 per iter in BOTH scripts.)
#
# ---- why every shared flag is pinned explicitly (default drift is a trap) ----
#   The two scripts have DIFFERENT argparse defaults; unless pinned they would
#   silently diverge on more than the h-loss:
#     skip_final_steps_pct : check_last_cos=0    block_ablation=50   -> pinned 0
#     region_mask_mode     : check_last_cos=attn block_ablation=none -> pinned attn
#     save_attmaps         : check_last_cos=off  block_ablation=on   -> pinned on
#     h_loss_form default  : check_last_cos=cos  block_ablation=raw  -> set per run
#   None of these touch the h-loss math, but they change the SDS region weighting
#   / IO, so we hold them identical to keep the comparison clean.
#
# ---- the 600-step cap + wImg=8 ----
#   parse_args() applies the --config YAML *after* the CLI flags, so YAML keys
#   OVERRIDE matching --flags. The cap and the image-loss weight are therefore
#   enforced in configs/debias-text-encoder2_600_wimg8.yaml
#   (max_train_steps: 600, weight_loss_img: 8). The CLI --weight_loss_img 8 below
#   is redundant-but-consistent (config wins, and it is also 8).
#
# ---- GOTCHA: do NOT pass "--h_loss_form mse" to the block_ablation script ----
#   block_ablation's choices are {raw, cos} (no "mse"); "raw" IS the MSE form.
#   check_last_cos's choices are {cos, mse, raw} (raw is a legacy alias of mse).
#
# ---- checkpoints / eval at 600 steps ----
#   CKPT_LONG=200 -> permanent checkpoints at 200/400/600 (600 = final model).
#   EVAL_EVERY=200 -> in-run eval at 200/400/600.
# ==========================================================================

set -u
cd "$(dirname "$0")"

ACC_CFG="configs/accelerate_config2.yaml"            # GPUs 2,3 (num_processes: 2)
CFG="configs/debias-text-encoder2_600_wimg8.yaml"    # max_train_steps: 600, weight_loss_img: 8

PY_LAST="check_last_cos.py"
PY_ALL="1-main-gender-sgd_dmscr_h_gen_check_block_ablation.py"

# ---- knobs held IDENTICAL across both runs ----
WIMG=8            # image-loss weight (config also pins 8; kept here for readability)
SKIP_PCT=0        # skip_final_steps_pct (pinned; defaults differ 0 vs 50)
REGION=attn       # region_mask_mode   (pinned; defaults differ attn vs none)
FACE=4            # weight_loss_face_realistic
EVAL_EVERY=200    # in-run eval at 200/400/600
CKPT_LONG=200     # permanent checkpoints at 200/400/600 (incl. final)

run () {
  local tag="$1"; local py="$2"; shift 2
  local hflags=("$@")   # h-loss-specific flags that DEFINE the difference
  echo "==================================================================="
  echo ">>> [$tag]  py=$py  hflags=(${hflags[*]})  (max_train_steps=600, wImg=$WIMG)"
  echo "==================================================================="
  accelerate launch --config_file "$ACC_CFG" \
    "$py" --config "$CFG" \
    --region_mask_mode "$REGION" \
    --skip_final_steps_pct "$SKIP_PCT" \
    --weight_loss_face_realistic "$FACE" \
    --weight_loss_img "$WIMG" \
    --evaluate_every_n_iter "$EVAL_EVERY" \
    --checkpointing_steps_long "$CKPT_LONG" \
    --save_attmaps \
    "${hflags[@]}"
}

# RUN 1: LAST executed step only, MID block, MSE
run "1/2  last-step  mid  MSE" "$PY_LAST" --h_loss_form mse

# RUN 2: ALL executed steps (mean), MID block, MSE  (raw == mse; mid == mid_block; no norm)
run "2/2  all-steps  mid  MSE" "$PY_ALL" --h_loss_form raw --h_block_names mid --h_block_norm none

echo "Both runs (last-step vs all-steps h-loss, 600 steps, wImg=8) finished."
