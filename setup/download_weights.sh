#!/usr/bin/env bash
# =============================================================================
#  학습/평가에 필요한 모든 가중치를 웹에서 로컬 기본 캐시(~/.cache)로 받는다.
#  코드가 실제로 from_pretrained/torch.hub/insightface 로 부르는 것만 받음. (총 ~24GB)
#
#  받는 목록 (전부 코드에서 참조됨):
#    1) SD1.5              : runwayml 원본이 HF에서 삭제 -> 커뮤니티 미러 + alias 로 그대로 동작
#    2) CLIP-ViT-H-14      : laion/CLIP-ViT-H-14-laion2B-s32B-b79K
#    3) CLIP-ViT-bigG-14   : laion/CLIP-ViT-bigG-14-laion2B-39B-b160k (open_clip weight)
#    4) openai CLIP-large  : openai/clip-vit-large-patch14
#    5) DINOv2             : torch.hub facebookresearch/dinov2 (vitb14 + vitg14)
#    6) insightface        : buffalo_l (detection)
#
#  ※ bootstrap.sh 가 자동으로 호출한다. 단독 실행도 가능:
#       bash setup/download_weights.sh
#     연결이 끊겨도 이어받게 tmux 안에서 돌리는 걸 권장:
#       tmux new -s dl ; bash setup/download_weights.sh   (Ctrl+b d 로 detach)
# =============================================================================
set -uo pipefail

# 기본 캐시(~/.cache)로 받도록 NAS/오프라인 env 제거
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME 2>/dev/null || true
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
HUB="$HOME/.cache/huggingface/hub"
mkdir -p "$HUB"

hf_dl () {  # $1=repo  $2..=allow_patterns (없으면 전체)
  local repo="$1"; shift
  python - "$repo" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, pats = sys.argv[1], sys.argv[2:] or None
p = snapshot_download(repo_id=repo, allow_patterns=pats, resume_download=True, max_workers=8)
print("  OK:", repo, "->", p)
PY
}

echo "=== [1/6] SD1.5 (mirror, diffusers 컴포넌트만 ~4GB) ==="
hf_dl stable-diffusion-v1-5/stable-diffusion-v1-5 \
  "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
  "vae/config.json"  "vae/diffusion_pytorch_model.safetensors" \
  "text_encoder/config.json" "text_encoder/model.safetensors" \
  "tokenizer/*" "scheduler/scheduler_config.json" "model_index.json"
# 코드가 부르는 'runwayml/...' 이름으로도 캐시에서 찾게 alias (코드/설정 수정 불필요)
rm -rf "$HUB/models--runwayml--stable-diffusion-v1-5"
ln -sfn "$HUB/models--stable-diffusion-v1-5--stable-diffusion-v1-5" \
        "$HUB/models--runwayml--stable-diffusion-v1-5"
echo "  alias: models--runwayml--stable-diffusion-v1-5 -> mirror"

echo "=== [2/6] CLIP-ViT-H-14 (transformers, ~3.7GB) ==="
hf_dl laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  "config.json" "model.safetensors" "preprocessor_config.json"

echo "=== [3/6] CLIP-ViT-bigG-14 (open_clip weight, ~9.5GB) ==="
hf_dl laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
  "open_clip_model.safetensors" "open_clip_config.json"

echo "=== [4/6] openai/clip-vit-large-patch14 (~1.6GB) ==="
hf_dl openai/clip-vit-large-patch14

echo "=== [5/6] DINOv2 (torch.hub: vitb14 + vitg14, ~4.6GB) ==="
[ -f "$HOME/.cache/torch/hub/facebookresearch_dinov2_main/hubconf.py" ] || \
  rm -rf "$HOME/.cache/torch/hub/facebookresearch_dinov2_main"
python - <<'PY'
import torch
for name in ["dinov2_vitb14", "dinov2_vitg14"]:
    torch.hub.load('facebookresearch/dinov2', name)
    print("  OK: dinov2", name)
PY

echo "=== [6/6] insightface buffalo_l (~0.28GB) ==="
[ -f "$HOME/.insightface/models/buffalo_l/det_10g.onnx" ] || rm -rf "$HOME/.insightface"
python - <<'PY'
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_l", allowed_modules=['detection'])
print("  OK: buffalo_l")
PY

echo
echo "================================================================"
echo " 다운로드 완료. 로컬 캐시 크기:"
du -sh "$HOME/.cache/huggingface" "$HOME/.cache/torch" "$HOME/.insightface" 2>/dev/null
echo
echo " 학습 시엔 8-rank 재다운로드 레이스 방지로 offline 켜고 실행 권장:"
echo "   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
echo "================================================================"
