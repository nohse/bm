#!/usr/bin/env python
"""v2: fully faithful reproduction of the nodetector training conditions + fix sweep.
Differences vs v1 (which got 6/8 on truncated z0 while real training shows ~0/24):
  * generation uses padding=True SHORT-context embeddings (training's tokenizer call),
    not 77-token padding -- this changes the generated z0 distribution;
  * scoring runs under BOTH default SDPA and the vanilla get_attention_scores path
    (what the installed CrossAttnCaptureProcessor uses in training);
  * then a fix sweep on the cached TRUNCATED z0s: prompt pairs x t-ranges at K=8.
"""
import os
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = "runwayml/stable-diffusion-v1-5"
DEV, DT = "cuda", torch.float16
GUID, NUM_STEPS, FRAC = 7.5, 21, 0.5
K = 8

OCC_FACE = ["doctor", "butcher", "senator", "cosmetologist", "geologist", "lifeguard",
            "narrator", "researcher", "custodian", "administrator", "economist", "bartender"]
OCC_NOFACE = ["doctor", "butcher", "senator", "lifeguard", "custodian", "economist"]
FACE_GEN = [f"A photo of the face of a {o}, a person" for o in OCC_FACE]
NOFACE_GEN = [f"A photo of a {o} seen from behind, back of the head, no face visible" for o in OCC_NOFACE]

torch.cuda.set_per_process_memory_fraction(0.2, 0)
tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=DT).to(DEV).eval()
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=DT).to(DEV).eval()
sched = DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")


@torch.no_grad()
def embed_train_style(texts, pad_to=None):
    """training generation path: padding=True (batch-longest), NOT max_length=77"""
    kw = dict(return_tensors="pt", padding=True) if pad_to is None else \
         dict(return_tensors="pt", padding="max_length", max_length=pad_to, truncation=True)
    tt = tok(texts, **kw)
    return te(tt["input_ids"].to(DEV), tt["attention_mask"].to(DEV))[0]


@torch.no_grad()
def embed77(text):
    ids = tok([text], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(DEV)
    return te(ids)[0]


@torch.no_grad()
def gen_z0_train_style(prompt, seed, truncated):
    """generate_image_no_gradient replica INCLUDING padding=True short-context embeds."""
    pe = embed_train_style([prompt])                       # [1, L, D], L = prompt length
    ue = embed_train_style([""], pad_to=pe.shape[1])       # uncond padded to same L
    c = torch.cat([ue, pe])
    g = torch.Generator(device=DEV).manual_seed(seed)
    lat = torch.randn((1, 4, 64, 64), generator=g, device=DEV, dtype=DT)
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
    return lat


PROMPT_EMB = {p: embed77(p) for p in
              ["a photo of a face", "a faceless photo", "a portrait photo of a person",
               "a photo of a person", "a photo of an empty scene", "a photo with no person"]}


def set_cross_attn_processor(proc_cls):
    """like the training file: install on attn2 (cross) ONLY; self-attn keeps SDPA."""
    procs = {}
    for name in unet.attn_processors:
        procs[name] = proc_cls() if name.endswith("attn2.processor") else AttnProcessor2_0()
    unet.set_attn_processor(procs)


@torch.no_grad()
def E_of(z0, text, t_lo, t_hi, k, seed=777, chunk=4):
    n = z0.shape[0]
    ts = torch.linspace(t_lo, t_hi, steps=k, device=DEV).round().long()
    out = torch.zeros(n, device=DEV)
    for s in range(0, n, chunk):
        z0c = z0[s:s + chunk]
        nc = z0c.shape[0]
        g = torch.Generator(device=DEV).manual_seed(seed + s)
        eps_list, zt_list = [], []
        for t in ts:
            e = torch.randn(z0c.shape, generator=g, device=DEV, dtype=z0c.dtype)
            zt_list.append(sched.add_noise(z0c, e, t.repeat(nc)))
            eps_list.append(e)
        eps_all = torch.stack(eps_list, 1).reshape(nc * k, *z0c.shape[1:])
        zt_all = torch.stack(zt_list, 1).reshape(nc * k, *z0c.shape[1:]).to(DT)
        t_all = ts.repeat(nc)
        emb = PROMPT_EMB[text].expand(nc * k, -1, -1)
        p = unet(zt_all, t_all, encoder_hidden_states=emb).sample
        out[s:s + nc] = (p.float() - eps_all.float()).pow(2).mean(dim=(1, 2, 3)).view(nc, k).mean(1)
    return out


def acc_report(zs_face, zs_noface, fp, np_, t_lo, t_hi, tag):
    Ef_f = E_of(zs_face, fp, t_lo, t_hi, K); En_f = E_of(zs_face, np_, t_lo, t_hi, K)
    Ef_n = E_of(zs_noface, fp, t_lo, t_hi, K); En_n = E_of(zs_noface, np_, t_lo, t_hi, K)
    face_ok = int((Ef_f < En_f).sum()); nf_ok = int((Ef_n >= En_n).sum())
    mf = (En_f - Ef_f).mean().item(); mn = (En_n - Ef_n).mean().item()
    print(f"  [{tag:36s}] face {face_ok}/{len(zs_face)}  noface {nf_ok}/{len(zs_noface)}"
          f"  | margin face {mf:+.2e}  noface {mn:+.2e}")
    return face_ok, nf_ok


print("=== generating (train-style padding=True), full + truncated ===", flush=True)
ZF_full, ZF_tr = [], []
for i, p in enumerate(FACE_GEN):
    ZF_full.append(gen_z0_train_style(p, 4242 + i, False))
    ZF_tr.append(gen_z0_train_style(p, 4242 + i, True))
ZN_full, ZN_tr = [], []
for i, p in enumerate(NOFACE_GEN):
    ZN_full.append(gen_z0_train_style(p, 9242 + i, False))
    ZN_tr.append(gen_z0_train_style(p, 9242 + i, True))
ZF_full = torch.cat(ZF_full); ZF_tr = torch.cat(ZF_tr)
ZN_full = torch.cat(ZN_full); ZN_tr = torch.cat(ZN_tr)

print("\n=== A) processor path: SDPA (ablation) vs get_attention_scores (training capture proc) ===")
print(" -- current errFD config: face vs faceless, t50-950, K=8 --")
for proc_cls, pname in ((AttnProcessor2_0, "SDPA"), (AttnProcessor, "get_attention_scores(attn2-only)")):
    set_cross_attn_processor(proc_cls)
    print(f" processor = {pname}")
    acc_report(ZF_full, ZN_full, "a photo of a face", "a faceless photo", 50, 950, "FULL z0")
    acc_report(ZF_tr, ZN_tr, "a photo of a face", "a faceless photo", 50, 950, "TRUNCATED z0")

print("\n=== B) fix sweep on TRUNCATED z0 (training distribution), K=8, capture-path processor ===")
set_cross_attn_processor(AttnProcessor)    # match training numerics (attn2 only)
for fp, np_ in [("a photo of a face", "a faceless photo"),
                ("a portrait photo of a person", "a faceless photo"),
                ("a photo of a person", "a photo with no person"),
                ("a portrait photo of a person", "a photo of an empty scene")]:
    for (t_lo, t_hi) in [(50, 950), (300, 950), (400, 800), (500, 950), (600, 950)]:
        acc_report(ZF_tr, ZN_tr, fp, np_, t_lo, t_hi, f"{fp[:20]} | {np_[:18]} | t{t_lo}-{t_hi}")
print("done")
