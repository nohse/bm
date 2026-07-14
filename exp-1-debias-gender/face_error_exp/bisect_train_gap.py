#!/usr/bin/env python
"""Bisect why errFD works in the offline ablation (~88%) but fails in the actual
training run (~0.5/24) despite skip_denoise_frac=0.0 (full generation confirmed).

Replicates the training generation VERBATIM (fp32 latent trajectory, padding=True
short-context embeds, batch-3 per prompt, 19-23 DPM steps, capture-path attn2), with
the RUN'S OWN occupations, then scores errFD (face/faceless, K=8, t50-950) under:
  A  raw generation z0 (fp32)          <- what training feeds the classifier
  A2 raw z0 cast fp16
  B  VAE ROUND-TRIP z0: decode -> uint8 image -> re-encode -> mean*sf
     <- what the OFFLINE ABLATION actually scored (images saved to PNG!)
  D  raw z0 (fp32) with unet.train()+gradient checkpointing (exact training mode)
"""
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = "runwayml/stable-diffusion-v1-5"
DEV = "cuda"
FP16, FP32 = torch.float16, torch.float32
GUID, K, T_LO, T_HI = 7.5, 8, 50, 950

# occupations from the run's own first logged steps (0/24 or 4/24 there)
OCCS = ["envoy", "dental laboratory technician", "grounds maintenance worker",
        "rail yard engineer", "detective", "choreographer", "travel clerk",
        "medical equipment preparer"]
STEPS = [21, 19, 23, 20, 22, 21, 19, 23]     # training samples range(19,24)

torch.cuda.set_per_process_memory_fraction(0.25, 0)
tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=FP16).to(DEV).eval()
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=FP16).to(DEV).eval()
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=FP16).to(DEV).eval()
sched = DPMSolverMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")

# capture-path processors on attn2 only (training installs these on the scoring unet == gen unet)
procs = {n: (AttnProcessor() if n.endswith("attn2.processor") else AttnProcessor2_0())
         for n in unet.attn_processors}
unet.set_attn_processor(procs)


@torch.no_grad()
def enc_train_style(texts, pad_to=None):
    kw = dict(return_tensors="pt", padding=True) if pad_to is None else \
         dict(return_tensors="pt", padding="max_length", max_length=pad_to, truncation=True)
    t = tok(texts, **kw)
    return te(t["input_ids"].to(DEV), t["attention_mask"].to(DEV))[0]


@torch.no_grad()
def gen_batch3(prompt, seed, nsteps):
    """generate_image_no_gradient VERBATIM: fp32 latents, fp16 UNet inputs, batch 3."""
    pe = enc_train_style([prompt] * 3)
    ue = enc_train_style([""] * 3, pad_to=pe.shape[1])
    c = torch.cat([ue, pe]).to(FP16)
    g = torch.Generator(device=DEV).manual_seed(seed)
    latents = torch.randn((3, 4, 64, 64), generator=g, device=DEV, dtype=FP32)  # fp32 like training
    sched.set_timesteps(nsteps)
    for t in sched.timesteps:
        inp = torch.cat([latents.detach().to(FP16)] * 2)
        inp = sched.scale_model_input(inp, t)
        pred = unet(inp, t, encoder_hidden_states=c).sample.to(FP32)
        pu, pt = pred.chunk(2)
        pred = pu + GUID * (pt - pu)
        latents = sched.step(pred, t, latents).prev_sample
    return latents                                            # fp32 raw z0 (training path)


E_FACE = None
E_FACELESS = None


@torch.no_grad()
def embed77(text):
    ids = tok([text], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(DEV)
    return te(ids)[0]


@torch.no_grad()
def errfd_margin(z0, seed=777):
    """exact residual_face_indicators; returns margin E_faceless - E_face per image."""
    n = z0.shape[0]
    ts = torch.linspace(T_LO, T_HI, steps=K, device=DEV).round().long()
    g = torch.Generator(device=DEV).manual_seed(seed)
    eps_l, zt_l = [], []
    for t in ts:
        e = torch.randn(z0.shape, generator=g, device=DEV, dtype=z0.dtype)
        zt_l.append(sched.add_noise(z0, e, t.repeat(n)))
        eps_l.append(e)
    eps_all = torch.stack(eps_l, 1).reshape(n * K, *z0.shape[1:])
    zt_all = torch.stack(zt_l, 1).reshape(n * K, *z0.shape[1:]).to(FP16)
    t_all = ts.repeat(n)
    E = {}
    for cls, emb in (("face", E_FACE), ("faceless", E_FACELESS)):
        # chunk rows to keep memory low
        outs = []
        for s in range(0, n * K, 24):
            p = unet(zt_all[s:s+24], t_all[s:s+24],
                     encoder_hidden_states=emb.expand(min(24, n*K-s), -1, -1)).sample
            outs.append((p.float() - eps_all[s:s+24].float()).pow(2).mean(dim=(1, 2, 3)))
        E[cls] = torch.cat(outs).view(n, K).mean(1)
    return (E["faceless"] - E["face"]).cpu().numpy()


@torch.no_grad()
def roundtrip(z0):
    """decode -> clamp -> uint8 image -> re-encode (the OFFLINE ABLATION's z0 path)."""
    img = vae.decode((z0.to(FP16) / vae.config.scaling_factor)).sample.clamp(-1, 1)
    arr = ((img.permute(0, 2, 3, 1).float().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    x = torch.from_numpy(arr).float().div(127.5).sub(1.0).permute(0, 3, 1, 2).to(DEV, FP16)
    return vae.encode(x).latent_dist.mean * vae.config.scaling_factor


E_FACE = embed77("a photo of a face")
E_FACELESS = embed77("a faceless photo")

print("generating 8 occupations x 3 imgs (training-verbatim, fp32 z0) ...", flush=True)
Z_raw = []
for i, (occ, ns) in enumerate(zip(OCCS, STEPS)):
    Z_raw.append(gen_batch3(f"A photo of the face of a {occ}, a person", 5991 + i, ns))
Z_raw = torch.cat(Z_raw)                       # [24,4,64,64] fp32
Z_rt = roundtrip(Z_raw)                        # ablation-style z0

conds = {}
conds["A  raw z0 fp32 (training path)"] = Z_raw
conds["A2 raw z0 cast fp16"] = Z_raw.to(FP16)
conds["B  VAE round-trip z0 (ablation path)"] = Z_rt

for name, z in conds.items():
    ms = np.stack([errfd_margin(z, seed=sd) for sd in (777, 1234)])   # [2,24]
    m = ms.mean(0)
    print(f"[{name:38s}] faces {(m>0).sum():>2}/24   margin mean {m.mean():+.3e}  "
          f"min {m.min():+.2e}  max {m.max():+.2e}", flush=True)

# D: exact training-mode scoring on raw z0
unet.train()
unet.enable_gradient_checkpointing()
ms = np.stack([errfd_margin(Z_raw, seed=sd) for sd in (777, 1234)])
m = ms.mean(0)
print(f"[{'D  raw z0 + unet.train()+ckpt':38s}] faces {(m>0).sum():>2}/24   margin mean {m.mean():+.3e}")
unet.disable_gradient_checkpointing(); unet.eval()
print("done")
