"""Prove my pipeline generation == original code's generate_image_no_gradient.

Runs both paths from the SAME initial noise and reports pixel difference.
"""
import numpy as np
import torch
from diffusers import (StableDiffusionPipeline, DPMSolverMultistepScheduler,
                       UNet2DConditionModel, AutoencoderKL)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

MODEL = "/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DEV = "cuda"
DT = torch.float16
DT_HI = torch.float32
PROMPT = "a photo of a realistic face"
STEPS = 25
GUID = 7.5

# --- shared components (identical weights for both paths) ---
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEV, DT)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(DEV, DT)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(DEV, DT)
sched = DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler")


# ---------- replica of original generate_image_no_gradient (with flags to isolate cause) ----------
@torch.no_grad()
def orig_generate(prompt, noises, dyn_pad=True, use_attn_mask=True, cfg_hi=True):
    N = noises.shape[0]
    prompts = [prompt] * N
    if dyn_pad:  # ORIGINAL: dynamic padding + pass attention_mask
        tok = tokenizer(prompts, return_tensors="pt", padding=True)
    else:        # PIPELINE style: pad prompt to max_length 77
        tok = tokenizer(prompts, return_tensors="pt", padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True)
    ids = tok["input_ids"].to(DEV); am = tok["attention_mask"].to(DEV)
    prompt_embeds = text_encoder(ids, am if use_attn_mask else None)[0]
    bs = prompt_embeds.shape[0]
    unc = tokenizer([""] * bs, padding="max_length",
                    max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt")
    uids = unc["input_ids"].to(DEV); uam = unc["attention_mask"].to(DEV)
    neg = text_encoder(uids, uam if use_attn_mask else None)[0]
    emb = torch.cat([neg, prompt_embeds]).to(DT)
    dt_acc = DT_HI if cfg_hi else DT

    sched.set_timesteps(STEPS)
    latents = noises
    for t in sched.timesteps:
        lmi = torch.cat([latents.to(DT)] * 2)
        lmi = sched.scale_model_input(lmi, t)
        np_ = unet(lmi, t, encoder_hidden_states=emb).sample.to(dt_acc)
        u, c = np_.chunk(2)
        np_ = u + GUID * (c - u)
        latents = sched.step(np_, t, latents).prev_sample
    latents = 1 / vae.config.scaling_factor * latents
    img = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)
    return (img / 2 + 0.5).clamp(0, 1)  # [0,1]


# ---------- my pipeline path (same weights, same initial noise) ----------
pipe = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    scheduler=DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler"),
    safety_checker=None, feature_extractor=None, requires_safety_checker=False,
)
pipe.set_progress_bar_config(disable=True)


def to_np(img01):  # [1,3,H,W] -> HWC uint8
    a = (img01[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return a


def dnp(x, y):
    d = np.abs(to_np(x).astype(int) - to_np(y).astype(int))
    return f"max|Δ|={d.max():3d} mean|Δ|={d.mean():6.3f} identical%={100*(d==0).mean():4.1f}"


seed = 0
g = torch.Generator(device=DEV).manual_seed(seed)
noise = torch.randn((1, 4, 64, 64), generator=g, device=DEV, dtype=DT)

# reference = EXACT original
ref = orig_generate(PROMPT, noise.clone(), dyn_pad=True, use_attn_mask=True, cfg_hi=True)
# turn OFF original-specific choices one at a time toward pipeline behavior
v_noattn = orig_generate(PROMPT, noise.clone(), dyn_pad=True, use_attn_mask=False, cfg_hi=True)
v_maxpad = orig_generate(PROMPT, noise.clone(), dyn_pad=False, use_attn_mask=False, cfg_hi=True)
v_fp16 = orig_generate(PROMPT, noise.clone(), dyn_pad=False, use_attn_mask=False, cfg_hi=False)
# my actual pipeline
out = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=GUID,
           height=512, width=512, latents=noise.clone(), output_type="pt").images
mine = (out if out.ndim == 4 else out.unsqueeze(0)).clamp(0, 1)

print("ISOLATION (each row vs the EXACT original), seed 0:")
print(f"  original            (ref)          : {dnp(ref, ref)}")
print(f"  - drop attention_mask              : {dnp(ref, v_noattn)}")
print(f"  - drop attn + maxlen-pad prompt    : {dnp(ref, v_maxpad)}")
print(f"  - + fp16 CFG (= pipeline settings) : {dnp(ref, v_fp16)}")
print(f"  MY PIPELINE                        : {dnp(ref, mine)}")
print(f"  [check] pipeline vs fp16-manual    : {dnp(mine, v_fp16)}")
Image.fromarray(np.concatenate([to_np(ref), to_np(mine)], axis=1)).save(
    "/workspace/verify_orig_vs_mine.png")
