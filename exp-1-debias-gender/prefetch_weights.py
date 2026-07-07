#!/usr/bin/env python
"""
Single-process prefetch / validate for chekc_SCRclip_attmap_grad.py.

Run this ONCE (plain `python`, NOT accelerate) before the 3-rank launch.
It downloads + extracts + parses every weight the training run needs, so the
3 accelerate ranks all read from a warm, valid cache and never race — which is
what corrupted the buffalo_l landmark ONNX (google.protobuf DecodeError).

Every identifier below is copied verbatim from chekc_SCRclip_attmap_grad.py
and its config defaults, so the cache keys match exactly.
"""
import os
import sys
import glob

# import torch BEFORE insightface (same ordering the training script requires)
import torch

FAIL = []


def step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        print(f"OK  {name}", flush=True)
    except Exception as e:
        print(f"FAIL {name} -> {type(e).__name__}: {e}", flush=True)
        FAIL.append(name)


# ---------------------------------------------------------------- HuggingFace hub
def hf_sd15():
    from diffusers import (
        AutoencoderKL, UNet2DConditionModel,
        DDPMScheduler, DPMSolverMultistepScheduler,
    )
    from transformers import CLIPTokenizer, CLIPTextModel
    m = "runwayml/stable-diffusion-v1-5"
    CLIPTokenizer.from_pretrained(m, subfolder="tokenizer")
    CLIPTextModel.from_pretrained(m, subfolder="text_encoder")
    AutoencoderKL.from_pretrained(m, subfolder="vae")
    UNet2DConditionModel.from_pretrained(m, subfolder="unet")
    DPMSolverMultistepScheduler.from_config(m, subfolder="scheduler")
    DDPMScheduler.from_pretrained(m, subfolder="scheduler")


def hf_zeroshot_clip():
    from transformers import CLIPModel, CLIPProcessor
    m = "openai/clip-vit-large-patch14"          # args.zeroshot_model default
    CLIPModel.from_pretrained(m)
    CLIPProcessor.from_pretrained(m)


def hf_clip_vision_h14():
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    m = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    CLIPImageProcessor.from_pretrained(m)
    CLIPVisionModelWithProjection.from_pretrained(m)


# --------------------------------------------------------------------- open_clip
def openclip_bigg():
    import open_clip
    open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k",
        precision="fp32", device="cpu",
    )
    open_clip.get_tokenizer("ViT-bigG-14")


# --------------------------------------------------------------------- torch.hub
def hub_dinov2():
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")   # img-preservation
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")   # eval DINO-I


# ------------------------------------------------------------------- torchvision
def tv_mobilenet():
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)


# -------------------------------------------------------------------- insightface
def insightface_buffalo():
    from insightface.app import FaceAnalysis
    # CPU provider is enough to force download+extract; we only need a warm cache.
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    # Explicitly re-parse every onnx — this is the exact op that crashed rank2.
    import onnx
    root = os.path.expanduser("~/.insightface/models/buffalo_l")
    for f in sorted(glob.glob(os.path.join(root, "*.onnx"))):
        onnx.load(f)
        print(f"    parsed {os.path.basename(f)}", flush=True)


# ------------------------------------------------------------- local (not a DL)
def check_local_classifier():
    p = "../data/5-trained-test-classifiers/CelebA-MobileNetLarge-Gender-09191318/epoch=19-step=25320_MobileNetLarge.pt"
    if not os.path.exists(p):
        raise FileNotFoundError(f"classifier weight missing (must be provided, not downloadable): {p}")
    print(f"    found {p} ({os.path.getsize(p)} bytes)", flush=True)


if __name__ == "__main__":
    step("SD1.5 (runwayml/stable-diffusion-v1-5)", hf_sd15)
    step("zero-shot CLIP (openai/clip-vit-large-patch14)", hf_zeroshot_clip)
    step("CLIP-ViT-H-14 vision (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)", hf_clip_vision_h14)
    step("open_clip ViT-bigG-14 (laion2b_s39b_b160k)", openclip_bigg)
    step("DINOv2 vitb14 + vitg14 (torch.hub)", hub_dinov2)
    step("torchvision mobilenet_v3_large weights", tv_mobilenet)
    step("insightface buffalo_l (+ onnx parse check)", insightface_buffalo)
    step("local gender classifier checkpoint", check_local_classifier)

    print("\n" + "=" * 50)
    if FAIL:
        print("PREFETCH INCOMPLETE — fix these before launching:")
        for n in FAIL:
            print("  - " + n)
        sys.exit(1)
    print("ALL WEIGHTS READY — safe to accelerate launch.")
