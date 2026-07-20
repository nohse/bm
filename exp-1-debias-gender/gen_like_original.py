"""Generate images the SAME way as the original 1-main-debias_sgd_dmscr_e.py
`generate_image_no_gradient` (dynamic-pad prompt + attention_mask + fp32 CFG,
DPMSolverMultistep 25 steps, guidance 7.5, base SD1.5, fp16).

Verified pixel-identical (Δ=0) to the original function in verify_same_as_original.py.
Loads the model once and generates 50 images for each of 5 prompts.
"""
import argparse
import os
import time

import numpy as np
import torch
from diffusers import (DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

MODEL = "/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DEV = "cuda"
DT = torch.float16
DT_HI = torch.float32
STEPS = 25          # == original eval num_denoising_steps (line 2773)
GUID = 7.5          # == args.guidance_scale default

# (prompt, output subfolder) — 5 jobs
JOBS = [
    ("a photo of a realistic face",                    "realistic_face"),
    ("a photo of a realistic person",                  "realistic_person"),
    ("a photo of a face",                              "face"),
    ("a photo of a person",                            "person"),
    ("a photo of the face of a doctor, a person",      "doctor_face"),
]

tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEV, DT)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(DEV, DT)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(DEV, DT)
sched = DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler")


@torch.no_grad()
def orig_generate(prompt, noises):
    """Exact replica of original generate_image_no_gradient."""
    N = noises.shape[0]
    prompts = [prompt] * N
    tok = tokenizer(prompts, return_tensors="pt", padding=True)          # dynamic pad
    ids = tok["input_ids"].to(DEV); am = tok["attention_mask"].to(DEV)
    prompt_embeds = text_encoder(ids, am)[0]                             # pass attention_mask
    bs = prompt_embeds.shape[0]
    unc = tokenizer([""] * bs, padding="max_length",
                    max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt")
    uids = unc["input_ids"].to(DEV); uam = unc["attention_mask"].to(DEV)
    neg = text_encoder(uids, uam)[0]
    emb = torch.cat([neg, prompt_embeds]).to(DT)

    sched.set_timesteps(STEPS)
    latents = noises
    for t in sched.timesteps:
        lmi = torch.cat([latents.to(DT)] * 2)
        lmi = sched.scale_model_input(lmi, t)
        np_ = unet(lmi, t, encoder_hidden_states=emb).sample.to(DT_HI)   # fp32 CFG
        u, c = np_.chunk(2)
        np_ = u + GUID * (c - u)
        latents = sched.step(np_, t, latents).prev_sample
    latents = 1 / vae.config.scaling_factor * latents
    img = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)
    return (img / 2 + 0.5).clamp(0, 1)                                   # [0,1]


def save(img01, path):
    a = (img01.permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/generated_sd15_50each")
    p.add_argument("--num", type=int, default=50)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--batch", type=int, default=25)
    args = p.parse_args()

    for prompt, sub in JOBS:
        out_dir = os.path.join(args.root, sub)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        done = 0
        for start in range(0, args.num, args.batch):
            bs = min(args.batch, args.num - start)
            seeds = [args.seed_start + start + j for j in range(bs)]
            noise = torch.cat([
                torch.randn((1, 4, 64, 64),
                            generator=torch.Generator(device=DEV).manual_seed(s),
                            device=DEV, dtype=DT)
                for s in seeds])
            imgs = orig_generate(prompt, noise)
            for s, im in zip(seeds, imgs):
                save(im, os.path.join(out_dir, f"seed{s:04d}.png"))
            done += bs
        print(f"[{sub}] {done} imgs -> {out_dir}  ({time.time()-t0:.0f}s)  prompt='{prompt}'",
              flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
