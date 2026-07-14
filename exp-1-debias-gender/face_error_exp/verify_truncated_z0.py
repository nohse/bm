#!/usr/bin/env python
"""Reproduce the nodetector training failure: the errFD classifier was ablated on FULLY
DENOISED latents, but the truncated_hspace training file feeds it a TRUNCATED
(skip_denoise_frac=0.5) x0 estimate. This script generates the SAME images both ways
(full vs truncated, replicating generate_image_no_gradient's jump formula exactly),
runs the K=8 t50-950 face/faceless classifier on both z0s, and uses insightface on the
DECODED truncated image as ground truth (what the original detector actually saw)."""
import os, math
import numpy as np
import torch
from insightface.app import FaceAnalysis
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = "runwayml/stable-diffusion-v1-5"
DEV, DT = "cuda", torch.float16
GUID = 7.5
NUM_STEPS = 21              # training samples from range(19,24)
FRAC = 0.5                  # args.skip_denoise_frac default in the truncated files
K, T_LO, T_HI = 8, 50, 950  # errFD settings in the nodetector file

FACE_PROMPTS = [f"A photo of the face of a {o}, a person" for o in
                ["doctor", "butcher", "senator", "cosmetologist", "geologist",
                 "lifeguard", "narrator", "researcher"]]
NOFACE_PROMPTS = [f"A photo of a {o} seen from behind, back of the head, no face visible"
                  for o in ["doctor", "butcher", "senator", "lifeguard"]]

torch.cuda.set_per_process_memory_fraction(0.15, 0)

tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=DT).to(DEV).eval()
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=DT).to(DEV).eval()
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=DT).to(DEV).eval()
sched = DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")


def embed(text):
    ids = tok([text], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        return te(ids)[0]


E_FACE = embed("a photo of a face")
E_FACELESS = embed("a faceless photo")
E_UNCOND = embed("")


@torch.no_grad()
def gen_z0(prompt, seed, truncated):
    """Replicates generate_image_no_gradient: DPMSolver loop w/ CFG; if truncated, run only
    n_run=round((1-frac)*n) steps and jump x0=(z_t - sqrt(1-abar)*eps)/sqrt(abar)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    lat = torch.randn((1, 4, 64, 64), generator=g, device=DEV, dtype=DT)
    c = torch.cat([E_UNCOND, embed(prompt)])
    sched.set_timesteps(NUM_STEPS, device=DEV)
    n_total = len(sched.timesteps)
    n_run = n_total if not truncated else max(1, int(round((1 - FRAC) * n_total)))
    for i, t in enumerate(sched.timesteps):
        inp = sched.scale_model_input(torch.cat([lat] * 2), t)
        eps = unet(inp, t, encoder_hidden_states=c).sample
        eu, ec = eps.chunk(2)
        eps = eu + GUID * (ec - eu)
        if truncated and i == n_run - 1:
            abar = sched.alphas_cumprod[t].to(device=DEV, dtype=eps.dtype)
            lat = (lat - (1 - abar).sqrt() * eps) / abar.sqrt()
            break
        lat = sched.step(eps, t, lat).prev_sample
    sched.set_timesteps(NUM_STEPS, device=DEV)  # reset state
    return lat  # scheduler-scale z0, matches training


@torch.no_grad()
def errfd(z0):
    """The exact residual_face_indicators estimator: K=8, t in [50,950], shared eps."""
    n = z0.shape[0]
    ts = torch.linspace(T_LO, T_HI, steps=K, device=DEV).round().long()
    eps_list, zt_list = [], []
    for t in ts:
        e = torch.randn_like(z0)
        zt_list.append(sched.add_noise(z0, e, t.repeat(n)))
        eps_list.append(e)
    eps_all = torch.stack(eps_list, 1).reshape(n * K, *z0.shape[1:])
    zt_all = torch.stack(zt_list, 1).reshape(n * K, *z0.shape[1:]).to(DT)
    t_all = ts.repeat(n)
    E = {}
    for cls, emb in (("face", E_FACE), ("faceless", E_FACELESS)):
        p = unet(zt_all, t_all, encoder_hidden_states=emb.expand(n * K, -1, -1)).sample
        E[cls] = (p.float() - eps_all.float()).pow(2).mean(dim=(1, 2, 3)).view(n, K).mean(1)
    return (E["face"] < E["faceless"]).item(), (E["faceless"] - E["face"]).item()


app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


@torch.no_grad()
def decode_detect(z0):
    img = vae.decode((z0 / vae.config.scaling_factor).to(DT)).sample.clamp(-1, 1)
    arr = ((img[0].permute(1, 2, 0).float().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    return len(app.get(arr[:, :, ::-1])) > 0


rows = []
for label, prompts in (("face", FACE_PROMPTS), ("noface", NOFACE_PROMPTS)):
    for pi, prompt in enumerate(prompts):
        seed = 4242 + pi
        zf = gen_z0(prompt, seed, truncated=False)
        zt = gen_z0(prompt, seed, truncated=True)
        pf, mf = errfd(zf)
        pt, mt = errfd(zt)
        gt_f = decode_detect(zf)
        gt_t = decode_detect(zt)
        rows.append((label, pf, pt, mf, mt, gt_f, gt_t))
        print(f"[{label:6s}] full: pred={'F' if pf else 'N'} m={mf:+.2e} det={gt_f} | "
              f"trunc: pred={'F' if pt else 'N'} m={mt:+.2e} det={gt_t}", flush=True)

fr = [r for r in rows if r[0] == "face"]
nr = [r for r in rows if r[0] == "noface"]
print("\n================= SUMMARY =================")
print(f"FACE prompts   (n={len(fr)}):  errFD says face on FULL z0: {sum(r[1] for r in fr)}/{len(fr)}"
      f"   on TRUNCATED z0: {sum(r[2] for r in fr)}/{len(fr)}")
print(f"                insightface on decoded: full {sum(r[5] for r in fr)}/{len(fr)}, trunc {sum(r[6] for r in fr)}/{len(fr)}")
print(f"NOFACE prompts (n={len(nr)}):  errFD says face on FULL z0: {sum(r[1] for r in nr)}/{len(nr)}"
      f"   on TRUNCATED z0: {sum(r[2] for r in nr)}/{len(nr)}")
print(f"margin (E_faceless - E_face) mean: FACE full {np.mean([r[3] for r in fr]):+.3e} -> trunc {np.mean([r[4] for r in fr]):+.3e}")
print(f"                                   NOFACE full {np.mean([r[3] for r in nr]):+.3e} -> trunc {np.mean([r[4] for r in nr]):+.3e}")
