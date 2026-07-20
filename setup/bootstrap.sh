#!/usr/bin/env bash
# =============================================================================
#  새 VM에서 이 프로젝트 환경을 "한 번에" 재현한다.
#    파이썬(3.11.10) -> PyTorch 2.9.1+cu128 -> 전체 pip 패키지 -> 모든 모델 가중치
#
#  전제: NVIDIA 드라이버가 설치돼 있고 GPU가 보임 (nvidia-smi 동작).
#        CUDA 툴킷은 필요 없음 — torch 휠이 CUDA 12.8 런타임을 포함해서 받음.
#
#  사용 (저장소 클론 후):
#     bash setup/bootstrap.sh
#  환경 이름 바꾸려면:
#     BM_ENV_NAME=myenv bash setup/bootstrap.sh
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${BM_ENV_NAME:-bm}"
PY_VER="3.11.10"
CU_INDEX="https://download.pytorch.org/whl/cu128"
REQ="$SCRIPT_DIR/requirements_lock.txt"

log(){ printf '\n\033[1;36m%s\033[0m\n' "$*"; }

log "[0/4] GPU/드라이버 점검"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
else
  echo "  경고: nvidia-smi 없음. GPU 학습을 하려면 NVIDIA 드라이버가 필요합니다. (설치는 계속 진행)"
fi

log "[1/4] conda 확보 (없으면 Miniconda 설치 → 파이썬 제공)"
find_conda() {
  command -v conda >/dev/null 2>&1 && return 0
  for c in "$HOME/miniconda3" "/opt/conda" "$HOME/anaconda3" "$HOME/miniforge3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then . "$c/etc/profile.d/conda.sh"; return 0; fi
  done
  return 1
}
if ! find_conda; then
  echo "  → Miniconda 설치 중..."
  MC="$HOME/miniconda3"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o /tmp/miniconda_bm.sh
  bash /tmp/miniconda_bm.sh -b -p "$MC"
  rm -f /tmp/miniconda_bm.sh
  . "$MC/etc/profile.d/conda.sh"
fi
eval "$(conda shell.bash hook)"

log "[2/4] 환경 '$ENV_NAME' 생성 (python=$PY_VER)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  환경 이미 존재 → 재사용"
else
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"
echo -n "  파이썬: "; python --version

log "[3/4] pip 패키지 정확 재현 (torch 2.9.1+cu128 포함, 전체 lock)"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r "$REQ" --extra-index-url "$CU_INDEX"

log "[4/4] 모델 가중치 다운로드 (~24GB, 회선에 따라 오래 걸릴 수 있음)"
bash "$SCRIPT_DIR/download_weights.sh"

log "설치 검증"
python - <<'PY'
import torch, torchvision
print("  torch       :", torch.__version__)
print("  torchvision :", torchvision.__version__)
print("  cuda build  :", torch.version.cuda)
print("  gpu 사용가능:", torch.cuda.is_available(), "| gpu 수:", torch.cuda.device_count())
PY

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cat <<EOF

================================================================
 ✅ 완료. 이제 이 환경에서 코드를 돌리면 됩니다.

 1) 새 터미널에서 환경 활성화:
      conda activate ${ENV_NAME}

 2) 학습 재다운로드 레이스 방지(멀티 GPU면 권장):
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

 3) 실행 예시:
      cd ${REPO_ROOT}/exp-1-debias-gender
      accelerate launch --config_file configs/accelerate_config6.yaml \\
        "1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector.py" \\
        --config configs/debias-text-encoder6.yaml
================================================================
EOF
