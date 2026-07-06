#!/usr/bin/env bash
# ==========================================================================
# Block-ablation experiments for the h-space (image-semantics preserving) loss.
#
#   8 runs total = 4 block combos  x  2 loss configs
#     block combos : up+mid+down | up+mid | down+mid | up+down
#     loss configs : MSE + standard-norm   (--h_loss_form raw  --h_block_norm std)
#                    cosine, no norm        (--h_loss_form cos  --h_block_norm none)
#
#   Each run is capped at 160 training steps.
#   Runs sequentially; each writes to its own folder (tagged _hBlk-/_hForm-/_hNorm-).
#
# ---- gotcha 1: the 160-step cap ----
#   parse_args() applies the --config YAML *after* the CLI flags, so a CLI
#   --max_train_steps would be silently overridden by the YAML. The cap is
#   therefore enforced via configs/debias-text-encoder3_160.yaml (max_train_steps: 160).
#   That config intentionally OMITS weight_loss_img so $WIMG below (CLI) takes effect.
#
# ---- gotcha 2: getting a usable checkpoint at 160 steps ----
#   default checkpointing_steps_long=200 > 160, so a naive short run saves NO
#   permanent checkpoint. CKPT_LONG=160 below saves checkpoint-160 at the end.
#   EVAL_EVERY=200 (= default) means NO in-run eval within 160 steps; generate
#   images afterwards from checkpoint-160. Lower EVAL_EVERY (e.g. 40) for in-run eval.
# ==========================================================================

set -u
cd "$(dirname "$0")"

ACC_CFG="configs/accelerate_config3.yaml"       # 3 GPUs (num_processes: 3)
PY="1-main-gender-sgd_dmscr_h_gen_check_block_ablation.py"
CFG="configs/debias-text-encoder3_160.yaml"     # 3-GPU batch sizes, max_train_steps: 160

# ---- knobs shared by all 8 runs (tweak freely) ----
WIMG=2            # weight_loss_img (config omits it, so this CLI value wins)
SKIP_PCT=0       # skip_final_steps_pct
EVAL_EVERY=200    # eval cadence; =default, so with 160 steps there is NO in-run eval
CKPT_LONG=160     # permanent checkpoint-160 saved at the end (default 200 never fires)

run () {
  local tag="$1"; local blocks="$2"; local form="$3"; local norm="$4"
  echo "==================================================================="
  echo ">>> [$tag]  h_block_names=$blocks  h_loss_form=$form  h_block_norm=$norm  (max_train_steps=160)"
  echo "==================================================================="
  accelerate launch --config_file "$ACC_CFG" \
    "$PY" --config "$CFG" \
    --region_mask_mode attn \
    --skip_final_steps_pct "$SKIP_PCT" \
    --weight_loss_face_realistic 0 \
    --weight_loss_img "$WIMG" \
    --h_block_names "$blocks" \
    --h_loss_form "$form" \
    --h_block_norm "$norm" \
    --evaluate_every_n_iter "$EVAL_EVERY" \
    --checkpointing_steps_long "$CKPT_LONG" \
    --save_attmaps
}

# ---- Group 1: MSE + standard-norm  (raw / std) ----
run "1/8  MSE+std  up+mid+down" "up,mid,down" raw std
run "2/8  MSE+std  up+mid"      "up,mid"      raw std
run "3/8  MSE+std  down+mid"    "down,mid"    raw std
run "4/8  MSE+std  up+down"     "up,down"     raw std

# ---- Group 2: cosine, no norm  (cos / none) ----
run "5/8  cos      up+mid+down" "up,mid,down" cos none
run "6/8  cos      up+mid"      "up,mid"      cos none
run "7/8  cos      down+mid"    "down,mid"    cos none
run "8/8  cos      up+down"     "up,down"     cos none

echo "All 8 block-ablation runs finished."
