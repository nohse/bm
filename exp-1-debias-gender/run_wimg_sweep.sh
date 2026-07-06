#!/usr/bin/env bash
# ------------------------------------------------------------------
# Sweep weight_loss_img in {2,4,8}, 2 reps each = 6 runs.
# Each run is capped at 400 training steps (eval at 200 & 400).
# Runs are SEQUENTIAL (each run uses both GPUs).
# One run crashing does NOT stop the remaining runs.
#
# NOTE on config: this script's parse_args lets the YAML override CLI
# args, so weight_loss_img / max_train_steps CANNOT be passed on the
# command line. Instead we generate a per-run YAML (copied from the
# base YAML, with only weight_loss_img + max_train_steps changed).
# ------------------------------------------------------------------

cd /root/finetune-fair-diffusion/exp-1-debias-gender || exit 1

# Guard against accidental double-launch. Running this script twice makes the two
# accelerate jobs collide on the distributed rendezvous port -> ConnectionError.
LOCK="/tmp/run_wimg_sweep.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[ABORT] run_wimg_sweep.sh is ALREADY running (lock: $LOCK). Not starting a second copy."
  exit 1
fi

SCRIPT="1-main-gender-sgd_dmscr_h_gen_check.py"   # FR-fallback kept (use 1-main-gender-sgd_dmscr_h_gen.py for the 60-min NCCL-timeout fix)
MASTER_PORT=29555                                  # fixed rendezvous port (avoids default-port clashes)
ACC_CFG="configs/accelerate_config.yaml"
BASE_YAML="configs/debias-text-encoder.yaml"
MAX_STEPS=400

mkdir -p configs/_sweep logs_sweep

# weight_loss_img -> number of repetitions (wImg=2 once; wImg=4 & 8 twice = 5 runs total)
declare -A REPS=( [2]=1 [4]=2 [8]=2 )
ORDER=(2 4 8)
TOTAL=0; for w in "${ORDER[@]}"; do TOTAL=$((TOTAL + REPS[$w])); done

run_idx=0
for wimg in "${ORDER[@]}"; do
  for rep in $(seq 1 "${REPS[$wimg]}"); do
    run_idx=$((run_idx + 1))
    run_yaml="configs/_sweep/wimg-${wimg}_rep-${rep}.yaml"
    log="logs_sweep/wimg-${wimg}_rep-${rep}.log"

    # Build per-run YAML from the base, overriding only weight_loss_img + max_train_steps.
    python3 - "$BASE_YAML" "$run_yaml" "$wimg" "$MAX_STEPS" <<'PY'
import sys, yaml
base, out, wimg, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = yaml.safe_load(open(base)) or {}
d["weight_loss_img"] = wimg
d["max_train_steps"] = steps
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
PY

    echo "=========================================================="
    echo "[$(date '+%F %T')] RUN ${run_idx}/${TOTAL}  wImg=${wimg} rep=${rep}  (max_steps=${MAX_STEPS})"
    echo "  config=${run_yaml}"
    echo "  log=${log}"
    echo "=========================================================="

    # Stream live to the terminal AND save the full log to file (tee).
    accelerate launch --config_file "$ACC_CFG" --main_process_port "$MASTER_PORT" "$SCRIPT" --config "$run_yaml" 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[WARN] RUN ${run_idx} (wImg=${wimg} rep=${rep}) exited non-zero -- continuing to next run."
    fi

    sleep 15   # let GPU memory free before the next run
  done
done

echo "[$(date '+%F %T')] ALL ${TOTAL} RUNS COMPLETE."
