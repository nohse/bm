# 새 VM 환경 재현 (한 줄 설치)

이 환경과 **완전히 동일하게** — 파이썬 · PyTorch · 모든 패키지 · 모든 모델 가중치를
한 번에 설치한다. 새 VM에서 아래 **한 줄**만 실행하면 된다.

## ✅ 한 줄 명령

```bash
git clone -b exp-1-errorDAL-SCR-Face https://github.com/nohse/bm.git && bash bm/setup/bootstrap.sh
```

이 한 줄이 순서대로 수행하는 것:

1. **파이썬 3.11.10** — conda가 없으면 Miniconda 자동 설치 후 `bm` 환경 생성
2. **PyTorch 2.9.1 + CUDA 12.8** — `torch/torchvision/torchaudio` (cu128 휠)
3. **전체 pip 패키지** — [`requirements_lock.txt`](requirements_lock.txt) 140개 정확 고정
4. **모델 가중치 ~24GB** — 아래 6종을 웹에서 로컬 캐시(`~/.cache`)로 다운로드

설치가 끝나면 마지막에 실행 방법이 출력된다.

## 전제 조건

- Linux x86_64, NVIDIA GPU + **드라이버 설치됨** (`nvidia-smi` 동작).
  - CUDA 툴킷은 **불필요** — torch 휠이 CUDA 12.8 런타임을 포함한다.
  - Blackwell(RTX PRO 6000) 등 최신 GPU는 드라이버 570 이상 권장.
- 인터넷 연결(가중치 다운로드). 회선에 따라 가중치 단계는 수십 분 걸릴 수 있으니
  가능하면 `tmux`/`screen` 안에서 돌리는 걸 권장.

## 설치 후 실행

```bash
conda activate bm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1        # 멀티 GPU 재다운로드 레이스 방지

cd bm/exp-1-debias-gender
accelerate launch --config_file configs/accelerate_config6.yaml \
  "1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector.py" \
  --config configs/debias-text-encoder6.yaml
```

## 받는 가중치 (코드에서 실제로 참조하는 것만)

| # | 모델 | repo / 소스 | 용도 |
|---|------|-------------|------|
| 1 | Stable Diffusion v1.5 | `runwayml/stable-diffusion-v1-5` (원본 삭제됨 → 커뮤니티 미러 + alias) | 베이스 확산모델 |
| 2 | CLIP-ViT-H-14 | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 학습 image/vision |
| 3 | CLIP-ViT-bigG-14 | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | 평가 CLIP-T/I |
| 4 | CLIP-large | `openai/clip-vit-large-patch14` | 텍스트/이미지 인코더 |
| 5 | DINOv2 (vitb14, vitg14) | `torch.hub: facebookresearch/dinov2` | 특징 추출 |
| 6 | insightface buffalo_l | insightface 자동 다운로드 | 얼굴 검출 |

> `runwayml/stable-diffusion-v1-5`는 HuggingFace에서 삭제되어, 다운로드 스크립트가
> 커뮤니티 미러(`stable-diffusion-v1-5/stable-diffusion-v1-5`)를 받은 뒤 캐시 폴더
> alias를 만들어 코드의 `from_pretrained("runwayml/...")`가 수정 없이 그대로 동작한다.

가중치만 따로 다시 받으려면:
```bash
bash bm/setup/download_weights.sh
```

## 옵션: 개별 단계

- 이미 conda/파이썬이 있으면 `bootstrap.sh`가 알아서 감지해서 재사용한다.
- 환경 이름을 바꾸려면: `BM_ENV_NAME=myenv bash bm/setup/bootstrap.sh`

## 옵션: Docker로 더 엄격하게 동일 재현

바닥부터 완전히 동일한 OS/시스템 라이브러리까지 맞추고 싶으면, 이 프로젝트가
검증한 베이스 이미지를 쓰는 방법도 있다:

```bash
docker run -d --name bm --gpus all --ipc=host --shm-size=32g \
  -v "$PWD/bm":/workspace -w /workspace \
  pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel sleep infinity
docker exec bm bash -lc "pip install --no-cache-dir -r setup/requirements_lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 && bash setup/download_weights.sh"
```

이 이미지는 파이썬 3.11 + torch 2.9.1+cu128을 이미 포함하므로 `bootstrap.sh`의
conda/파이썬 설치 단계가 필요 없다.
