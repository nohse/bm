#!/usr/bin/env bash
# ==========================================================================
# Block-ablation experiments for the h-space (image-semantics preserving) loss.
#   --- COSINE, NO-NORM variant ---
#
#   4 runs total = 4 block combos  x  1 loss config
#     block combos : up+mid+down | up+mid | down+mid | up+down
#     loss config  : cosine, no norm   (--h_loss_form cos  --h_block_norm none)
#
#   This is the cos-only counterpart of run_block_ablation_160.sh, which also
#   ran an MSE + standard-norm group (--h_loss_form raw --h_block_norm std).
#   Here we drop MSE entirely and ablate ONLY the cosine-similarity distance
#   computed on raw (un-normalized) block features:
#       block_loss = 1 - cosine_similarity(h_fine.flatten, h_ref.flatten)
#   Cosine is inherently scale-invariant, so --h_block_norm none keeps the
#   features raw (no per-block standardization is applied).
#
#   The cos/raw branch is selected purely from the CLI via --h_loss_form, which
#   already exists in the python (choices: raw|cos); no code change is needed.
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

# ---- knobs shared by all runs (tweak freely) ----
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

# ---- cosine, no norm  (cos / none) ----
run "1/4  cos  up+mid+down" "up,mid,down" cos none
run "2/4  cos  up+mid"      "up,mid"      cos none
run "3/4  cos  down+mid"    "down,mid"    cos none
run "4/4  cos  up+down"     "up,down"     cos none

echo "All 4 cos (no-norm) block-ablation runs finished."
