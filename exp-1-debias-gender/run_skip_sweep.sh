#!/usr/bin/env bash
# ------------------------------------------------------------------
# Sweep skip_final_steps_pct in {0, 25, 50, 75} %, 2 reps each = 8 runs.
#   - weight_loss_img = 2
#   - max 1000 training steps (eval at 200/400/600/800/1000)
#   - NO images uploaded to wandb (metrics still logged)
# Runs are SEQUENTIAL (each run uses all GPUs).
# One run crashing does NOT stop the remaining runs.
#
# Output folders/files are tagged with the skip value (folder name gets
# "_skip-{N}_", per-run yaml/log are "skip-{N}_rep-{R}").
# ------------------------------------------------------------------

cd /root/finetune-fair-diffusion/exp-1-debias-gender || exit 1

# Guard against accidental double-launch (two accelerate jobs would collide on the port).
LOCK="/tmp/run_skip_sweep.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[ABORT] run_skip_sweep.sh is ALREADY running (lock: $LOCK). Not starting a second copy."
  exit 1
fi

SCRIPT="1-main-gender-sgd_dmscr_h_gen_check.py"   # use 1-main-gender-sgd_dmscr_h_gen.py for the 60-min NCCL-timeout fix
MASTER_PORT=29556                                  # fixed rendezvous port
ACC_CFG="configs/accelerate_config.yaml"
BASE_YAML="configs/debias-text-encoder.yaml"
WIMG=2
MAX_STEPS=1000
SKIPS=(0 25 50 75)
REPS=2

mkdir -p configs/_sweep logs_sweep

TOTAL=$(( ${#SKIPS[@]} * REPS ))
run_idx=0
for skip in "${SKIPS[@]}"; do
  for rep in $(seq 1 "$REPS"); do
    run_idx=$((run_idx + 1))
    run_yaml="configs/_sweep/skip-${skip}_rep-${rep}.yaml"
    log="logs_sweep/skip-${skip}_rep-${rep}.log"

    # Build per-run YAML: base + weight_loss_img / max_train_steps / skip_final_steps_pct / no wandb images.
    python3 - "$BASE_YAML" "$run_yaml" "$WIMG" "$MAX_STEPS" "$skip" <<'PY'
import sys, yaml
base, out, wimg, steps, skip = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
d = yaml.safe_load(open(base)) or {}
d["weight_loss_img"]      = wimg
d["max_train_steps"]      = steps
d["skip_final_steps_pct"] = skip
d["log_wandb_images"]     = False
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
PY

    echo "=========================================================="
    echo "[$(date '+%F %T')] RUN ${run_idx}/${TOTAL}  skip=${skip}% rep=${rep}  (wImg=${WIMG}, max_steps=${MAX_STEPS}, no-wandb-images)"
    echo "  config=${run_yaml}"
    echo "  log=${log}"
    echo "=========================================================="

    # Stream live to the terminal AND save the full log to file (tee).
    accelerate launch --config_file "$ACC_CFG" --main_process_port "$MASTER_PORT" "$SCRIPT" --config "$run_yaml" 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[WARN] RUN ${run_idx} (skip=${skip} rep=${rep}) exited non-zero -- continuing to next run."
    fi

    sleep 15   # let GPU memory free before the next run
  done
done

echo "[$(date '+%F %T')] ALL ${TOTAL} RUNS COMPLETE."
