#!/usr/bin/env bash
# Run the debias-gender training 4 times, each capped at 600 iterations.
#   - 2x with --skip_final_steps_pct 50
#   - 2x with --skip_final_steps_pct 0
#
# 600-iter cap:
#   parse_args() applies the --config YAML *after* the CLI flags, so any key in the
#   YAML OVERRIDES the matching command-line flag. The base configs/debias-text-encoder4.yaml
#   sets `max_train_steps: 10000`, so a CLI `--max_train_steps 600` would be silently
#   ignored. The cap is therefore enforced via a dedicated config that is identical to
#   debias-text-encoder4.yaml except `max_train_steps: 600`.

set -u
cd "$(dirname "$0")"

ACC_CFG="configs/accelerate_config4.yaml"
PY="1-main-gender-sgd_dmscr_h_gen_check.py"
CFG="configs/debias-text-encoder4_600.yaml"   # copy of debias-text-encoder4.yaml, max_train_steps: 600

run () {
  local tag="$1"; local skip_pct="$2"
  echo "==================================================================="
  echo ">>> [$tag] skip_final_steps_pct=${skip_pct}  (max_train_steps=600)"
  echo "==================================================================="
  accelerate launch --config_file "$ACC_CFG" \
    "$PY" --config "$CFG" \
    --region_mask_mode attn \
    --skip_final_steps_pct "$skip_pct" \
    --weight_loss_face_realistic 0 \
    --weight_loss_img 2 \
    --save_attmaps
}

run "run 1/4  skip50" 50
run "run 2/4  skip50" 50
run "run 3/4  skip0"  0
run "run 4/4  skip0"  0

echo "All 4 runs finished."
