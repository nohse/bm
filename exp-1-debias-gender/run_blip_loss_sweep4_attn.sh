#!/usr/bin/env bash
# ============================================================================
# BLIP eval sweep (ATTN region-mask variant): 4 experiments x 2 reps = 8 SEQUENTIAL runs.
#
# IDENTICAL to run_blip_loss_sweep4.sh EXCEPT every generated per-run YAML also sets
#   region_mask_mode: attn
# so SDS is spatially weighted by the woman/man cross-attention maps
# (attmap_mean = 0.5*(att_woman + att_man)) instead of the unmasked default ('none').
#
#   Exp 1 : weight_loss_img = 0 , weight_loss_face_realistic = 0
#   Exp 2 : weight_loss_img = 2 , weight_loss_face_realistic = 4
#   Exp 3 : weight_loss_img = 4 , weight_loss_face_realistic = 4
#   Exp 4 : weight_loss_img = 8 , weight_loss_face_realistic = 4
#
# Fixed for every run (the rest of the parsers come from the base YAML):
#   - max_train_steps = 1000
#   - eval_at_step0   = True     -> eval at steps 0 / 200 / 400 / 600 / 800 / 1000
#   - region_mask_mode= attn     <-- ONLY difference vs run_blip_loss_sweep4.sh
#   - script          = 1-main-gender-sgd_dmscr_h_gen_check_blip.py  (BLIP eval)
#   - accelerate cfg  = configs/accelerate_config4.yaml
#   - base param yaml = configs/debias-text-encoder4.yaml
#
# IMPORTANT: this script's parse_args lets the YAML OVERRIDE CLI args and SILENTLY
# IGNORES unknown keys, so every per-run override must go through a generated YAML
# using the EXACT argparse names. region_mask_mode is a registered str arg, so it is
# injected into the generated YAML here (NOT as a CLI flag, which would be ignored).
#
# Output per run is fully separated by a UNIQUE proj_name (note the -attn tag) so it
# does NOT mix with the baseline ('none') sweep:
#   outputs/gender-blip-attn_<tag>/<auto-folder>/{imgs, ckpts, eval_results.json}
#
# Runs are SEQUENTIAL. One run crashing does NOT stop the rest.
#
# To pin to specific free GPUs (recommended when the machine is shared), launch with:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_blip_loss_sweep4_attn.sh
# accelerate's gpu_ids:"all" then only sees those GPUs.
# ============================================================================

# cd to this script's own directory (works regardless of /root vs /workspace).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Guard against accidental double-launch (distinct lock from the baseline sweep).
LOCK="/tmp/run_blip_loss_sweep4_attn.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[ABORT] run_blip_loss_sweep4_attn.sh is ALREADY running (lock: $LOCK). Not starting a second copy."
  exit 1
fi

SCRIPT="1-main-gender-sgd_dmscr_h_gen_check_blip.py"
MASTER_PORT=29561                              # distinct rendezvous port from baseline sweep (29557)
ACC_CFG="configs/accelerate_config4.yaml"
BASE_YAML="configs/debias-text-encoder4.yaml"
MAX_STEPS=1000
REPS=2

# Parallel arrays: experiment id / weight_loss_img / weight_loss_face_realistic
EXP_IDS=(1 2 3 4)
WIMGS=(0 2 4 8)
REALS=(0 4 4 4)

mkdir -p configs/_sweep logs_sweep

TOTAL=$(( ${#EXP_IDS[@]} * REPS ))
run_idx=0

for i in "${!EXP_IDS[@]}"; do
  exp="${EXP_IDS[$i]}"
  wimg="${WIMGS[$i]}"
  real="${REALS[$i]}"
  for rep in $(seq 1 "$REPS"); do
    run_idx=$((run_idx + 1))

    # Unique tag -> unique proj_name -> fully separated output folder (note: -attn).
    tag="exp${exp}-wimg${wimg}-real${real}-attn_rep${rep}"
    proj="gender-blip-attn_${tag}"
    run_yaml="configs/_sweep/${tag}.yaml"
    log="logs_sweep/${tag}.log"

    # Build the per-run YAML from the base, overriding only the needed keys
    # (region_mask_mode: attn is the ONLY change vs the baseline sweep).
    python3 - "$BASE_YAML" "$run_yaml" "$proj" "$wimg" "$real" "$MAX_STEPS" <<'PY'
import sys, yaml
base, out, proj, wimg, real, steps = sys.argv[1:7]
d = yaml.safe_load(open(base)) or {}
d["proj_name"]                  = proj
d["weight_loss_img"]            = int(wimg)
d["weight_loss_face_realistic"] = float(real)
d["max_train_steps"]            = int(steps)
d["eval_at_step0"]              = True
d["region_mask_mode"]           = "attn"   # <-- the only difference from run_blip_loss_sweep4.sh
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
PY

    echo "=========================================================="
    echo "[$(date '+%F %T')] RUN ${run_idx}/${TOTAL}  exp=${exp}  wImg=${wimg}  realistic=${real}  rep=${rep}  region_mask_mode=attn"
    echo "  max_steps=${MAX_STEPS}  eval@steps=0,200,400,600,800,1000"
    echo "  config   = ${run_yaml}"
    echo "  proj_name= ${proj}"
    echo "  outputs  -> outputs/${proj}/<auto-folder>/eval_results.json"
    echo "  log      = ${log}"
    echo "=========================================================="

    # Stream live to the terminal AND save the full log to file (tee).
    accelerate launch --config_file "$ACC_CFG" --main_process_port "$MASTER_PORT" \
        "$SCRIPT" --config "$run_yaml" 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[WARN] RUN ${run_idx} (${tag}) exited non-zero -- continuing to next run."
    fi

    sleep 15   # let GPU memory free before the next run
  done
done

echo "[$(date '+%F %T')] ALL ${TOTAL} RUNS COMPLETE."
echo "Per-run metrics JSON: outputs/gender-blip-attn_exp*/*/eval_results.json"
