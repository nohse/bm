#!/usr/bin/env bash
# ============================================================================
# 06291-main-gender ATTN run: max_train_steps=600 을 3번 SEQUENTIAL 반복.
#
# 원래 의도한 명령:
#   accelerate launch --config_file configs/accelerate_config3.yaml \
#       06291-main-gender-sgd_dmscr_h_gen_check_att.py --config configs/debias-text-encoder3.yaml \
#       --region_mask_mode attn --skip_final_steps_pct 50 \
#       --weight_loss_face_realistic 4 --weight_loss_img 2
#
# 주의 2가지 (이 스크립트가 처리해 줌):
#   1) 이 파이썬 스크립트는 parse_args 후 --config YAML 이 CLI 인자를 "덮어쓴다".
#      base YAML(debias-text-encoder3.yaml) 에 max_train_steps:10000 이 있어서
#      CLI 로 --max_train_steps 600 을 줘도 무시된다. -> per-run YAML 을 생성해
#      600 을 직접 박아 넣는다 (모든 run-defining knob 도 YAML 로 통일).
#   2) 원래 명령의 --weight-main-gender_loss_face_realistic 는 존재하지 않는 인자다.
#      올바른 이름은 weight_loss_face_realistic. 여기서는 YAML 키로 넣는다.
#
# 출력은 rep 마다 proj_name 이 달라 완전히 분리된다:
#   outputs/<proj_name>/<timestamp_...>/{imgs, ckpts, eval_results.json}
#
# 사용법:
#   ./run_main_gender_600_x3.sh
#   CUDA_VISIBLE_DEVICES=0,1,2 ./run_main_gender_600_x3.sh   # GPU 고정 시
# ============================================================================
set -u

# 이 스크립트가 있는 디렉터리로 이동 (/root vs /workspace 무관하게 동작)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 더블 런 방지 락
LOCK="/tmp/run_main_gender_600_x3.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[ABORT] run_main_gender_600_x3.sh 가 이미 실행 중입니다 (lock: $LOCK)."
  exit 1
fi

SCRIPT="06291-main-gender-sgd_dmscr_h_gen_check_att.py"
ACC_CFG="configs/accelerate_config3.yaml"
BASE_YAML="configs/debias-text-encoder3.yaml"
MASTER_PORT=29563          # 다른 sweep 과 겹치지 않는 rendezvous 포트

# --- 이번 run 의 knob (원래 명령에서 가져옴) -------------------------------
MAX_STEPS=600
REGION_MASK_MODE="attn"
SKIP_FINAL_STEPS_PCT=50
WEIGHT_LOSS_IMG=2
WEIGHT_LOSS_FACE_REALISTIC=4
REPS=3
# ---------------------------------------------------------------------------

mkdir -p configs/_sweep logs_sweep

for rep in $(seq 1 "$REPS"); do
  tag="600x3_rep${rep}"
  proj="gender_aaai_600_rep${rep}"
  run_yaml="configs/_sweep/${tag}.yaml"
  log="logs_sweep/${tag}.log"

  # base YAML 에서 per-run YAML 생성: run-defining knob 들을 직접 박아 넣는다.
  python3 - "$BASE_YAML" "$run_yaml" "$proj" "$MAX_STEPS" \
      "$REGION_MASK_MODE" "$SKIP_FINAL_STEPS_PCT" \
      "$WEIGHT_LOSS_IMG" "$WEIGHT_LOSS_FACE_REALISTIC" <<'PY'
import sys, yaml
base, out, proj, steps, rmm, skip, wimg, wreal = sys.argv[1:9]
d = yaml.safe_load(open(base)) or {}
d["proj_name"]                  = proj
d["max_train_steps"]            = int(steps)
d["region_mask_mode"]           = rmm
d["skip_final_steps_pct"]       = float(skip)
d["weight_loss_img"]            = float(wimg)
d["weight_loss_face_realistic"] = float(wreal)
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
PY

  echo "=========================================================="
  echo "[$(date '+%F %T')] RUN ${rep}/${REPS}"
  echo "  max_steps=${MAX_STEPS}  region_mask_mode=${REGION_MASK_MODE}  skip=${SKIP_FINAL_STEPS_PCT}pct"
  echo "  wImg=${WEIGHT_LOSS_IMG}  wRealFace=${WEIGHT_LOSS_FACE_REALISTIC}"
  echo "  config   = ${run_yaml}"
  echo "  proj_name= ${proj}"
  echo "  log      = ${log}"
  echo "=========================================================="

  # 터미널에 실시간 출력 + 로그 파일 저장 (tee)
  accelerate launch --config_file "$ACC_CFG" --main_process_port "$MASTER_PORT" \
      "$SCRIPT" --config "$run_yaml" 2>&1 | tee "$log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[WARN] RUN ${rep} (${tag}) 가 비정상 종료됨 -- 다음 run 으로 계속 진행."
  fi

  sleep 15   # 다음 run 전에 GPU 메모리 정리 시간
done

echo "[$(date '+%F %T')] 전체 ${REPS} 회 RUN 완료."
echo "결과: outputs/gender_aaai_600_rep*/<timestamp_...>/"
