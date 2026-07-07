"""
Recompute realistic_SDS on the ALREADY-SAVED clean images (images_ori/, images_ft/)
using two alternative "realistic" prompts, keeping every other knob identical to
repro_clean.py (base SD-1.5 UNet + base TE, unmasked; K=15 t in [400,800], E=4 eps,
DDPM alphas, fp16).

Prompts compared (all evaluated on the SAME reloaded images + SAME noise per image, so
they are directly comparable to each other):
    orig : "a photo of a realistic face"    (what clean_results.json used)
    p1   : "a photorealistic face"
    p2   : "a realistic photo of a face"

NOTE: the noise here is freshly (seeded) sampled, so the "orig" column will differ
slightly from clean_results.json's realistic_SDS (that run used un-seeded noise). Use
the recomputed "orig" column -- not the old json -- as the baseline for the two new
prompts.

Output: realistic_prompts_results.json  (aggregates + per-image rows for each prompt).
"""
import os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"; DT = torch.float16; MODEL = "runwayml/stable-diffusion-v1-5"
SDS_TMIN, SDS_TMAX, SDS_NT = 400, 800, 15
NUM_EPS = 4
SEED = 1234

PROMPTS = {
    "orig": "a photo of a realistic face",
    "p1":   "a photorealistic face",
    "p2":   "a realistic photo of a face",
}

def log(*a): print(*a, flush=True)

# ---------------- models (only what SDS needs: VAE + UNet + base TE) ----------------
log("[load] SD-1.5 pipeline (vae+unet) ...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=DT,
        safety_checker=None, requires_safety_checker=False)
pipe.to(DEVICE); pipe.set_progress_bar_config(disable=True)
tokenizer = pipe.tokenizer; vae = pipe.vae; unet = pipe.unet

te_base = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEVICE, DT).eval().requires_grad_(False)
ddpm = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
alphas = ddpm.alphas_cumprod.to(DEVICE)

@torch.no_grad()
def embed_one(prompt):
    # single [1,77,768] embedding for a constant prompt (expanded per-chunk below)
    tok = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    return te_base(tok.input_ids.to(DEVICE), tok.attention_mask.to(DEVICE))[0].to(DT)

PE = {k: embed_one(v) for k, v in PROMPTS.items()}

def unet_chunked(zt, tv, pe1, chunk=48):
    outs = []
    for s in range(0, zt.shape[0], chunk):
        z = zt[s:s+chunk]
        pe = pe1.expand(z.shape[0], -1, -1)
        outs.append(unet(z, tv[s:s+chunk], encoder_hidden_states=pe).sample)
    return torch.cat(outs, 0)

@torch.no_grad()
def sds_real(images_m1, keys):
    """images_m1: [B,3,512,512] in [-1,1]. Returns {key: tensor[B]} of SDS_realistic."""
    B = images_m1.shape[0]
    lat = vae.encode(images_m1.to(DT)).latent_dist.sample() * vae.config.scaling_factor
    t_idx = torch.linspace(SDS_TMIN, SDS_TMAX, SDS_NT, device=DEVICE).round().long()
    K = t_idx.shape[0]; E = NUM_EPS
    lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(B, K, E, *lat.shape[1:]).contiguous().view(B*K*E, *lat.shape[1:])
    tv = t_idx.view(1, K, 1).expand(B, K, E).reshape(-1)
    eps = torch.randn_like(lat_exp)
    ab = alphas[tv].to(lat_exp.dtype)
    zt = ab.sqrt().view(-1, 1, 1, 1) * lat_exp + (1 - ab).sqrt().view(-1, 1, 1, 1) * eps
    out = {}
    for k in keys:
        ep = unet_chunked(zt, tv, PE[k])
        err = ((ep.float() - eps.float()) ** 2).mean(1)
        out[k] = err.flatten(1).mean(1).view(B, K, E).mean((1, 2)).cpu()
    return out

# ---------------- iterate over saved images ----------------
with open(os.path.join(HERE, "clean_results.json")) as f:
    clean = json.load(f)

def load_batch(imgdir, files):
    ims = []
    for fn in files:
        im = Image.open(os.path.join(HERE, imgdir, fn + ".png")).convert("RGB")
        a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        ims.append(a * 2 - 1)                       # [-1,1], matches repro_clean img_m1
    return torch.stack(ims).to(DEVICE)

def run(tag, imgdir, rows_meta):
    # group rows by occupation-block (o00..) preserving order; batch of n_per per block
    blocks = {}
    for r in rows_meta:
        blocks.setdefault(r["file"][:3], []).append(r)
    out_rows = []
    for bi, (key, rs) in enumerate(sorted(blocks.items())):
        files = [r["file"] for r in rs]
        imgs = load_batch(imgdir, files)
        res = sds_real(imgs, list(PROMPTS.keys()))
        for i, r in enumerate(rs):
            out_rows.append(dict(occ=r["occ"], idx=r["idx"], file=r["file"],
                sds_real_orig=float(res["orig"][i]),
                sds_real_p1=float(res["p1"][i]),
                sds_real_p2=float(res["p2"][i])))
        log(f"  [{tag}] block {bi+1}/{len(blocks)} {rs[0]['occ'][:24]:24s} "
            f"orig={res['orig'].mean():.4f} p1={res['p1'].mean():.4f} p2={res['p2'].mean():.4f}")
    return out_rows

def agg(rows):
    return {
        "n": len(rows),
        "realistic_SDS_orig": float(np.mean([r["sds_real_orig"] for r in rows])),
        "realistic_SDS_p1":   float(np.mean([r["sds_real_p1"]   for r in rows])),
        "realistic_SDS_p2":   float(np.mean([r["sds_real_p2"]   for r in rows])),
    }

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
log("\n=== ori (plain SD-1.5 images) ===")
rows_ori = run("ori", "images_ori", clean["rows_ori"])
log("\n=== ft (EMA TE-LoRA images) ===")
rows_ft = run("ft", "images_ft", clean["rows_ft"])

A_o, A_f = agg(rows_ori), agg(rows_ft)
log("\n=============== realistic_SDS by prompt (recomputed on saved images) ===============")
log(f"{'prompt':38s} {'ori':>10s} {'ft':>10s}")
log("-" * 60)
for k, name in [("orig", PROMPTS['orig']), ("p1", PROMPTS['p1']), ("p2", PROMPTS['p2'])]:
    log(f"{name:38s} {A_o['realistic_SDS_'+k]:10.4f} {A_f['realistic_SDS_'+k]:10.4f}")

out = dict(prompts=PROMPTS, config=dict(t_min=SDS_TMIN, t_max=SDS_TMAX, n_t=SDS_NT,
           num_eps=NUM_EPS, seed=SEED, note="recomputed on saved PNGs; 'orig' baseline "
           "differs from clean_results.json due to re-seeded noise"),
           ori=A_o, ft=A_f, rows_ori=rows_ori, rows_ft=rows_ft)
with open(os.path.join(HERE, "realistic_prompts_results.json"), "w") as f:
    json.dump(out, f, indent=2)
log(f"\n[done] wrote {os.path.join(HERE, 'realistic_prompts_results.json')}")
