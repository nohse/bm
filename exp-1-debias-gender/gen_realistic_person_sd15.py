"""Generate images of "a photo of a realistic person" with SD 1.5, one per seed.

Runs on CPU (this session has no CUDA). Deterministic: image i uses seed = seed_start + i.
"""
import argparse
import os
import time

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", default="a photo of a realistic person")
    p.add_argument("--out_dir", default="/workspace/generated_realistic_person_sd15")
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    # DPM-Solver++ gives good quality in few steps -> much faster
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    done = 0
    t0 = time.time()
    for start in range(0, args.num, args.batch):
        bs = min(args.batch, args.num - start)
        seeds = [args.seed_start + start + j for j in range(bs)]
        # per-image generator so each image has its own independent noise
        gens = [torch.Generator(device=device).manual_seed(s) for s in seeds]
        images = pipe(
            [args.prompt] * bs,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=gens,
        ).images
        for s, img in zip(seeds, images):
            img.save(os.path.join(args.out_dir, f"person_seed{s:04d}.png"))
        done += bs
        el = time.time() - t0
        print(f"[{done}/{args.num}] {el:.0f}s elapsed | {el/done:.1f}s/img | "
              f"ETA {el/done*(args.num-done)/60:.1f} min", flush=True)

    print(f"DONE {done} images -> {args.out_dir} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
