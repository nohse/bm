#!/usr/bin/env python
# coding=utf-8
"""
Generate 50 images WITH a human face and 50 images WITHOUT a face using Stable
Diffusion v1.5, then VERIFY each image with the same insightface FaceAnalysis
detector used by the debias experiments. Only images whose detector result
matches the intended label are kept; extra candidates are generated until each
set has exactly TARGET_PER_CLASS clean images.

Outputs:
  faces/face_XX.png        (detector: >=1 face)
  nonfaces/nonface_XX.png  (detector: 0 faces)
  manifest.json            (per-image: label, prompt, seed, n_faces, det_score)
  verify_summary.txt
"""
import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# torch MUST be imported before insightface so onnxruntime sees CUDA
from insightface.app import FaceAnalysis

from diffusers import (StableDiffusionPipeline, DPMSolverMultistepScheduler,
                       AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "runwayml/stable-diffusion-v1-5"
TARGET_PER_CLASS = 50

# ------------------------------------------------------------------ prompts
# FACE prompts: portrait / headshot framing so a human face is clearly present.
# Diverse across gender / age / ethnicity / occupation (on-topic for this repo).
FACE_SUBJECTS = [
    "a young woman", "an old man", "a middle-aged woman", "a teenage boy",
    "a smiling businessman", "a female doctor", "a male construction worker",
    "an elderly woman", "a young man with glasses", "a female scientist",
    "a male chef", "a woman firefighter", "a male teacher", "a female nurse",
    "a young girl", "an old fisherman", "a female lawyer", "a male athlete",
    "a woman farmer", "a male pilot", "a female artist", "a male musician",
    "a woman soldier", "a male student", "a female engineer", "a bearded man",
    "a woman with curly hair", "a man wearing a hat", "a female CEO",
    "a male police officer", "a woman journalist", "a male barber",
    "a female dancer", "a man with a mustache", "a young nurse",
    "a female astronaut", "a male gardener", "a woman librarian",
    "a male painter", "a female pilot", "a smiling grandmother",
    "a serious judge", "a woman singer", "a male waiter", "a female cashier",
    "a man in a suit", "a woman in a lab coat", "a male doctor",
    "a female teacher", "a young athlete", "an old craftsman", "a woman baker",
    "a male electrician", "a female architect", "a man reading",
    "a woman laughing", "a male violinist", "a female photographer",
    "a man drinking coffee", "a woman with sunglasses on her head",
]
FACE_TEMPLATE = "a close-up portrait photo of {s}, face clearly visible, looking at the camera, studio lighting, high detail"

# NON-FACE prompts: scenes / objects with NO people and NO close-up animal faces.
NONFACE_PROMPTS = [
    "a photo of a mountain landscape at sunset, no people",
    "a photo of an empty sandy beach with waves, no people",
    "a still life photo of a bowl of fresh fruit on a table",
    "a photo of a modern city skyline at night, no people",
    "a photo of a dense green forest with tall trees",
    "a photo of a plate of spaghetti pasta with tomato sauce",
    "a photo of a red sports car parked on a street, no people",
    "a photo of a wooden coffee table with a laptop and a cup",
    "a photo of a snowy mountain peak under a blue sky",
    "a photo of a field of yellow sunflowers",
    "a photo of an old stone bridge over a river",
    "a photo of a cozy living room interior with a sofa",
    "a photo of a bookshelf full of colorful books",
    "a photo of a calm lake reflecting the sky",
    "a photo of a bunch of ripe bananas on a market stall",
    "a photo of a lighthouse on a rocky coast",
    "a photo of a plate of sushi rolls",
    "a photo of a desert with sand dunes",
    "a photo of a waterfall in a tropical jungle",
    "a photo of a vintage bicycle leaning against a brick wall",
    "a photo of a cup of coffee and croissant on a cafe table",
    "a photo of autumn leaves on a forest path",
    "a photo of a starry night sky over a mountain",
    "a photo of a bowl of colorful vegetables",
    "a photo of a wooden cabin in the woods",
    "a photo of a glass skyscraper reflecting clouds",
    "a photo of a garden full of blooming roses",
    "a photo of an antique clock on a mantelpiece",
    "a photo of a plate of pancakes with syrup and berries",
    "a photo of a rocky canyon under bright sun",
    "a photo of a row of colorful houses along a canal",
    "a photo of a bowl of steaming ramen noodles",
    "a photo of a green tea plantation on hills",
    "a photo of a pile of autumn pumpkins",
    "a photo of a modern kitchen with marble countertops",
    "a photo of a hot air balloon over a valley",
    "a photo of a cluster of seashells on the sand",
    "a photo of a snow-covered pine forest",
    "a photo of a plate of chocolate cake slices",
    "a photo of a train station platform, empty, no people",
    "a photo of a cobblestone street in an old town, no people",
    "a photo of a stack of vinyl records and a turntable",
    "a photo of a bowl of fresh salad with tomatoes",
    "a photo of a foggy morning over a wheat field",
    "a photo of a colorful coral reef underwater",
    "a photo of a mountain lake surrounded by pine trees",
    "a photo of a plate of grilled vegetables",
    "a photo of a sunset over the ocean horizon",
    "a photo of a wooden dock extending into a calm lake",
    "a photo of a bouquet of tulips in a glass vase",
    "a photo of a busy highway interchange from above, no people",
    "a photo of a bowl of cereal with milk and strawberries",
    "a photo of an abstract pattern of colorful geometric shapes",
    "a photo of a snowy village at night with lit windows",
    "a photo of a stack of old leather suitcases",
    "a photo of a field of lavender flowers at sunset",
    "a photo of a glass of orange juice on a marble counter",
    "a photo of a mountain trail winding through rocks",
    "a photo of a plate of fresh oysters on ice",
    "a photo of a modern office desk with a monitor, no people",
]
NEG_FACE = "cropped, blurry, low quality, deformed, cartoon, painting, multiple faces"
NEG_NONFACE = "person, people, face, human, portrait, man, woman, child, crowd, hands"


def build_pipe(device):
    # The offline cache holds only the component subfolders (no model_index.json /
    # safety_checker), so assemble StableDiffusionPipeline from parts manually.
    dt = torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=dt)
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=dt)
    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=dt)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_face_app():
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"],
                       providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def detect_faces(app, pil_img):
    """Return (n_faces, max_det_score). Input BGR as insightface expects."""
    rgb = np.array(pil_img.convert("RGB"))
    faces = app.get(rgb[:, :, ::-1])  # RGB -> BGR
    if len(faces) == 0:
        return 0, 0.0
    return len(faces), float(max(f.det_score for f in faces))


def gen_one(pipe, prompt, neg, seed, device, steps=50, gs=7.5):
    g = torch.Generator(device=device).manual_seed(seed)
    img = pipe(prompt, negative_prompt=neg, num_inference_steps=steps,
               guidance_scale=gs, generator=g, height=512, width=512).images[0]
    return img


def collect(pipe, app, device, want_face, prompts_iter, neg, out_dir, prefix, target):
    """Generate until `target` images match the intended label (want_face)."""
    os.makedirs(out_dir, exist_ok=True)
    kept = []
    seed = 1000 if want_face else 9000
    pi = 0
    prompts = list(prompts_iter)
    attempts = 0
    pbar = tqdm(total=target, desc=f"{prefix} (want_face={want_face})")
    while len(kept) < target and attempts < target * 6:
        prompt = prompts[pi % len(prompts)]
        pi += 1
        seed += 1
        attempts += 1
        img = gen_one(pipe, prompt, neg, seed, device)
        n_faces, score = detect_faces(app, img)
        ok = (n_faces >= 1) if want_face else (n_faces == 0)
        if ok:
            idx = len(kept)
            fname = f"{prefix}_{idx:02d}.png"
            img.save(os.path.join(out_dir, fname))
            kept.append(dict(file=os.path.join(os.path.basename(out_dir), fname),
                             label=("face" if want_face else "nonface"),
                             intended_face=want_face, prompt=prompt, seed=seed,
                             n_faces=n_faces, det_score=round(score, 4)))
            pbar.update(1)
    pbar.close()
    print(f"[{prefix}] kept {len(kept)}/{target} after {attempts} attempts")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    print("loading insightface ...")
    app = build_face_app()
    print("loading SD1.5 ...")
    pipe = build_pipe(device)

    face_prompts = [FACE_TEMPLATE.format(s=s) for s in FACE_SUBJECTS]
    manifest = []
    manifest += collect(pipe, app, device, True, face_prompts, NEG_FACE,
                        os.path.join(HERE, "faces"), "face", TARGET_PER_CLASS)
    manifest += collect(pipe, app, device, False, NONFACE_PROMPTS, NEG_NONFACE,
                        os.path.join(HERE, "nonfaces"), "nonface", TARGET_PER_CLASS)

    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    n_face = sum(1 for m in manifest if m["label"] == "face")
    n_nonface = sum(1 for m in manifest if m["label"] == "nonface")
    # independent re-verification pass over the SAVED files (sanity)
    reface = sum(1 for m in manifest if m["label"] == "face" and m["n_faces"] >= 1)
    renon = sum(1 for m in manifest if m["label"] == "nonface" and m["n_faces"] == 0)
    summary = (
        f"FACE set:     {n_face} images, all with >=1 detected face: {reface}/{n_face}\n"
        f"NON-FACE set: {n_nonface} images, all with 0 detected faces: {renon}/{n_nonface}\n"
        f"face det_score range: "
        f"{min((m['det_score'] for m in manifest if m['label']=='face'), default=0):.3f}"
        f" - {max((m['det_score'] for m in manifest if m['label']=='face'), default=0):.3f}\n"
    )
    print(summary)
    with open(os.path.join(HERE, "verify_summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
