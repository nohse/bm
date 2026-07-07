"""
Attmap diagnostics for the SDS fair loss.
Reuses grad_decomp.py's loaded SD1.5 + AST-extracted loss fns.
Tests:
  (1) how face-concentrated is the attmap actually used as the SDS weight (IPR, top-k mass)
  (2) does attmap-weighting distort the gender signal g=sds_m-sds_f vs full-image (region=none)?
  (3) saves image+attmap overlays so we can see face-vs-hair coverage.
"""
import os, sys, math
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image

import grad_decomp as G   # loads SD1.5 + extracted fns on cuda:0
dev = G.DEVICE
OUT = "/tmp/claude-0/-workspace/7ab3f00c-b547-48e3-b662-81d50dddb7bc/scratchpad/attmap_out"
os.makedirs(OUT, exist_ok=True)

PROMPT = "A photo of the face of a doctor, a person"
N = 8
STEPS = 21
G.args.use_attn_weight = True

g = torch.Generator(device=dev).manual_seed(1234)
noises = torch.randn([N, 4, 64, 64], generator=g, device=dev, dtype=G.weight_dtype_high_precision)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
    imgs, _ = G.generate_image_no_gradient(PROMPT, noises, STEPS,
                which_text_encoder=G.text_encoder, which_unet=G.unet,
                skip_final_steps=0, skip_final_steps_pct=0.0)

    # --- attn mode: get attmap + weighted sds ---
    G.args.region_mask_mode = "attn"
    outA = G.sds_logits_from_images(imgs, tau=1e-4, t_min=400, t_max=800, num_t=15, num_eps=1,
                gate_grad=False, sds_text_encoder=G.eval_text_encoder, sds_unet=G.unet,
                sds_scheduler=G.ddpm_forward, compute_realistic_sds=False, return_attmap=True)
    preds_a, probs_a, logits_a, sds_f_a, sds_m_a, _sr, attmap = outA   # attmap [N,H,W]

    # --- none mode: full-image sds ---
    G.args.region_mask_mode = "none"
    outN = G.sds_logits_from_images(imgs, tau=1e-4, t_min=400, t_max=800, num_t=15, num_eps=1,
                gate_grad=False, sds_text_encoder=G.eval_text_encoder, sds_unet=G.unet,
                sds_scheduler=G.ddpm_forward, compute_realistic_sds=False)
    preds_n, probs_n, logits_n, sds_f_n, sds_m_n, _ = outN

# ---------------- (1) attmap concentration ----------------
att = attmap.float()                      # [N,H,W]
H, Wd = att.shape[-2], att.shape[-1]; P = H*Wd
w = att.clamp_min(0).flatten(1)
w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)   # per-sample sum-to-one (as loss does)
ipr = 1.0 / (w.pow(2).sum(dim=1))          # effective #pixels (inverse participation ratio)
# top-k mass
def topk_mass(w, frac):
    k = max(1, int(P*frac))
    tk = torch.topk(w, k, dim=1).values.sum(dim=1)
    return tk
print("="*70)
print(f"ATTMAP CONCENTRATION (latent {H}x{Wd} = {P} px), N={N} images, prompt='{PROMPT}'")
print(f"  effective area (IPR): mean={ipr.mean():.0f}px  ({100*ipr.mean()/P:.1f}% of image)  [min={ipr.min():.0f} max={ipr.max():.0f}]")
print(f"  mass in top  5% px: {topk_mass(w,0.05).mean():.2f}")
print(f"  mass in top 10% px: {topk_mass(w,0.10).mean():.2f}")
print(f"  mass in top 20% px: {topk_mass(w,0.20).mean():.2f}")
print(f"  mass in bottom 50% px: {(1-topk_mass(w,0.50)).mean():.3f}")

# ---------------- (2) signal: attn-weighted vs full-image ----------------
g_a = (sds_m_a - sds_f_a).float()   # gender signal, attn-weighted
g_n = (sds_m_n - sds_f_n).float()   # gender signal, full image
def stats(x): return f"mean={x.mean():+.3e} std={x.std():.3e} |mean|/std={abs(x.mean().item())/ (x.std().item()+1e-20):.2f}"
print("\nGENDER SIGNAL g = sds_m - sds_f  (per sample, N=%d)"%N)
print(f"  attn-weighted g: {stats(g_a)}")
print(f"  full-image   g: {stats(g_n)}")
# do they agree on sign (which gender)?
agree = ((g_a>0)==(g_n>0)).float().mean()
print(f"  sign agreement (same gender call) attn vs none: {agree:.2f}")
# correlation across samples
if N>2:
    ga=g_a-g_a.mean(); gn=g_n-g_n.mean()
    corr=(ga*gn).sum()/((ga.norm()*gn.norm())+1e-20)
    print(f"  correlation(g_attn, g_none) across samples: {corr:+.2f}")
print(f"  per-sample g_attn: {[round(x,5) for x in g_a.tolist()]}")
print(f"  per-sample g_none: {[round(x,5) for x in g_n.tolist()]}")
print(f"  SDS pred(attn) male-prob: {[round(x,2) for x in probs_a[:,1].tolist()]}")
print(f"  SDS pred(none) male-prob: {[round(x,2) for x in probs_n[:,1].tolist()]}")

# ---------------- (3) save image + attmap overlay ----------------
imgs_disp = ((imgs.float()*0.5+0.5).clamp(0,1)*255).byte().cpu()   # [N,3,512,512]
att_up = F.interpolate(att.unsqueeze(1), size=(imgs.shape[-2], imgs.shape[-1]), mode="bilinear", align_corners=False).squeeze(1)
# per-sample min-max for viz
def colorize(a):
    a=(a-a.min())/(a.max()-a.min()+1e-8)
    a=a.pow(0.5)  # same gamma as the code's viz
    # simple hot colormap
    r=(a*3).clamp(0,1); gg=(a*3-1).clamp(0,1); b=(a*3-2).clamp(0,1)
    return torch.stack([r,gg,b],0)
tiles=[]
for i in range(min(N,8)):
    im = imgs_disp[i].float()/255
    ov = colorize(att_up[i].cpu())
    blend = (0.55*im + 0.45*ov).clamp(0,1)
    row = torch.cat([im, blend], dim=2)  # image | overlay
    tiles.append(row)
grid = torch.cat(tiles, dim=1)
Image.fromarray((grid.permute(1,2,0).numpy()*255).astype('uint8')).save(os.path.join(OUT,"attmap_overlay.png"))
print(f"\nsaved overlay grid -> {OUT}/attmap_overlay.png  (left=image, right=attmap overlay)")
