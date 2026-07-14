#!/usr/bin/env python
# coding=utf-8
"""One GPU worker: generate GENUINE no-face-but-person images from the user's 50
occupations (back view / hands / truncated / workplace) and keep only candidates
where insightface AND dlib-HOG both detect ZERO faces (final CNN check happens at
selection). This builds a TRUE face-absent set with person/occupation context --
the hard, realistic negative class (vs. easy landscapes)."""
import os, json, argparse
import numpy as np
import torch
from PIL import Image
import face_recognition
from gen_occ_worker import build_pipe, build_face_app, detect

HERE = os.path.dirname(os.path.abspath(__file__))

NOFACE_TEMPLATES = [
    "A photo of a {occ} seen from behind, back of the head, no face visible",
    "A photo of a {occ} working, viewed from behind, back turned to the camera",
    "A close-up photo of the hands of a {occ} at work, no face",
    "A photo of a {occ} facing away from the camera, back turned",
    "A wide photo of the empty workplace of a {occ}, tools only, no people",
]


def occ_of(prompt):
    return prompt.split("face of a ")[1].split(", a person")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cand_dir", default=os.path.join(HERE, "noface_candidates"))
    args = ap.parse_args()
    os.makedirs(args.cand_dir, exist_ok=True)
    device = "cuda"

    base_prompts = json.load(open(os.path.join(HERE, "prompts_occ.json")))
    occs = [occ_of(p) for p in base_prompts]
    jobs = [(oi, ti, s) for oi in range(len(occs))
            for ti in range(len(NOFACE_TEMPLATES)) for s in range(args.seeds)]
    jobs = jobs[args.shard::args.nshards]

    app = build_face_app()
    pipe = build_pipe(device)

    recs = []
    kept = 0
    for k, (oi, ti, s) in enumerate(jobs):
        occ = occs[oi]
        prompt = NOFACE_TEMPLATES[ti].format(occ=occ)
        seed = 700000 + 1000 * ti + 7 * s + oi
        g = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=7.5,
                   generator=g, height=512, width=512).images[0]
        n_ins, sc = detect(app, img)
        rgb = np.array(img)
        n_hog = len(face_recognition.face_locations(rgb, model="hog")) if n_ins == 0 else 1
        if n_ins == 0 and n_hog == 0:   # both agree faceless -> keep candidate
            fname = f"nf_{oi:02d}_{ti}_{s}.png"
            img.save(os.path.join(args.cand_dir, fname))
            recs.append(dict(file=os.path.join("noface_candidates", fname), occ_idx=oi,
                             occupation=occ, template=ti, seed=seed,
                             insight_n=n_ins, hog_n=n_hog))
            kept += 1
        if k % 15 == 0:
            print(f"[nf shard {args.shard}] {k+1}/{len(jobs)} kept={kept}", flush=True)

    json.dump(recs, open(os.path.join(args.cand_dir, f"shard_{args.shard}.json"), "w"), indent=2)
    print(f"[nf shard {args.shard}] done: kept {kept} genuine-faceless of {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
