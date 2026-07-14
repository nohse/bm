#!/usr/bin/env python
# coding=utf-8
"""One GPU worker: generate SD1.5 images for a SHARD of (occupation_prompt, seed)
jobs using the user's exact template "A photo of the face of a {occupation}, a person",
run the insightface detector, and record whether a face was detected. No negative
prompt -> the NATURAL distribution, so 'no-face' images are genuine SD failures to
render a detectable face (the hard, realistic setting)."""
import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image

from insightface.app import FaceAnalysis
from diffusers import (StableDiffusionPipeline, DPMSolverMultistepScheduler,
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "runwayml/stable-diffusion-v1-5"


def build_pipe(device):
    dt = torch.float16
    tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=dt)
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=dt)
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=dt)
    sched = DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
                                   scheduler=sched, safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_face_app():
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"],
                       providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def detect(app, pil):
    rgb = np.array(pil.convert("RGB"))
    faces = app.get(rgb[:, :, ::-1])
    if not faces:
        return 0, 0.0
    return len(faces), float(max(f.det_score for f in faces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cand_dir", default=os.path.join(HERE, "occ_candidates"))
    args = ap.parse_args()
    device = "cuda"
    os.makedirs(args.cand_dir, exist_ok=True)

    prompts = json.load(open(os.path.join(HERE, "prompts_occ.json")))
    jobs = [(pi, s) for pi in range(len(prompts)) for s in range(args.seeds)]
    jobs = jobs[args.shard::args.nshards]

    app = build_face_app()
    pipe = build_pipe(device)

    recs = []
    for k, (pi, s) in enumerate(jobs):
        prompt = prompts[pi]
        seed = 10000 * (s + 1) + pi
        g = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5,
                   generator=g, height=512, width=512).images[0]
        n_faces, score = detect(app, img)
        fname = f"img_{pi:02d}_{s}.png"
        img.save(os.path.join(args.cand_dir, fname))
        recs.append(dict(file=os.path.join("occ_candidates", fname), prompt_idx=pi,
                         prompt=prompt, seed=seed, n_faces=n_faces,
                         det_score=round(score, 4),
                         label=("face" if n_faces >= 1 else "noface")))
        if k % 10 == 0:
            print(f"[shard {args.shard}] {k+1}/{len(jobs)}", flush=True)

    json.dump(recs, open(os.path.join(args.cand_dir, f"shard_{args.shard}.json"), "w"), indent=2)
    nf = sum(1 for r in recs if r["label"] == "face")
    print(f"[shard {args.shard}] done: {len(recs)} imgs, {nf} face / {len(recs)-nf} noface", flush=True)


if __name__ == "__main__":
    main()
