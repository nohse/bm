"""
realistic_SDS (prompt "a photo of a realistic face") on the saved clean images,
broken down over the FULL t range: t = 0,50,...,950  (stride 50 -> 20 timesteps).
For each timestep: mean over ori images, mean over ft images, and (ori - ft).
Also the overall average (mean over all 20 timesteps) and its ori-ft gap.
Everything else identical to repro_clean.py (base SD-1.5 UNet + base TE, unmasked, fp16).
"""
import os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"; DT = torch.float16; MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT = "a photo of a realistic face"
T_LIST = list(range(0, 1000, 50))          # 0,50,...,950  -> 20 timesteps
NUM_EPS = 8
SEED = 1234

def log(*a): print(*a, flush=True)

log("[load] SD-1.5 (vae+unet) + base TE ...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=DT,
        safety_checker=None, requires_safety_checker=False)
pipe.to(DEVICE); pipe.set_progress_bar_config(disable=True)
tokenizer = pipe.tokenizer; vae = pipe.vae; unet = pipe.unet
te_base = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEVICE, DT).eval().requires_grad_(False)
alphas = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler").alphas_cumprod.to(DEVICE)

@torch.no_grad()
def embed_one(prompt):
    tok = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    return te_base(tok.input_ids.to(DEVICE), tok.attention_mask.to(DEVICE))[0].to(DT)
PE = embed_one(PROMPT)

def unet_chunked(zt, tv, chunk=64):
    outs = []
    for s in range(0, zt.shape[0], chunk):
        z = zt[s:s+chunk]
        outs.append(unet(z, tv[s:s+chunk], encoder_hidden_states=PE.expand(z.shape[0], -1, -1)).sample)
    return torch.cat(outs, 0)

@torch.no_grad()
def sds_per_t(images_m1):
    """returns [B, K] : realistic eps-MSE per image, per timestep (mean over E eps)."""
    B = images_m1.shape[0]
    lat = vae.encode(images_m1.to(DT)).latent_dist.sample() * vae.config.scaling_factor
    t_idx = torch.tensor(T_LIST, device=DEVICE, dtype=torch.long)
    K = t_idx.shape[0]; E = NUM_EPS
    lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(B, K, E, *lat.shape[1:]).contiguous().view(B*K*E, *lat.shape[1:])
    tv = t_idx.view(1, K, 1).expand(B, K, E).reshape(-1)
    eps = torch.randn_like(lat_exp)
    ab = alphas[tv].to(lat_exp.dtype)
    zt = ab.sqrt().view(-1, 1, 1, 1) * lat_exp + (1 - ab).sqrt().view(-1, 1, 1, 1) * eps
    ep = unet_chunked(zt, tv)
    err = ((ep.float() - eps.float()) ** 2).mean(1)          # mean over 4 latent channels
    return err.flatten(1).mean(1).view(B, K, E).mean(2).cpu()  # [B,K]

def load_batch(imgdir, files):
    ims = []
    for fn in files:
        im = Image.open(os.path.join(HERE, imgdir, fn + ".png")).convert("RGB")
        a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        ims.append(a * 2 - 1)
    return torch.stack(ims).to(DEVICE)

clean = json.load(open(os.path.join(HERE, "clean_results.json")))

def run(tag, imgdir, rows_meta):
    blocks = {}
    for r in rows_meta:
        blocks.setdefault(r["file"][:3], []).append(r)
    per = []                                   # list of [B,K]
    for key, rs in sorted(blocks.items()):
        per.append(sds_per_t(load_batch(imgdir, [r["file"] for r in rs])))
    M = torch.cat(per, 0).numpy()              # [100, 20]
    log(f"  [{tag}] {M.shape[0]} imgs x {M.shape[1]} timesteps")
    return M

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
log("\n=== ori images ===");  Mo = run("ori", "images_ori", clean["rows_ori"])
log("=== ft images ===");     Mf = run("ft",  "images_ft",  clean["rows_ft"])

ori_t = Mo.mean(0); ft_t = Mf.mean(0); diff_t = ori_t - ft_t
# overall = mean over all 20 timesteps (= mean over all image-timestep samples)
ori_all = float(Mo.mean()); ft_all = float(Mf.mean()); diff_all = ori_all - ft_all

log("\n================= realistic_SDS by timestep (t=0..950, stride 50) =================")
log(f"{'t':>5s} {'ori':>10s} {'ft':>10s} {'ori-ft':>10s} {'rel%':>8s}")
log("-" * 48)
for i, t in enumerate(T_LIST):
    rel = 100 * diff_t[i] / ori_t[i] if ori_t[i] else 0.0
    log(f"{t:5d} {ori_t[i]:10.5f} {ft_t[i]:10.5f} {diff_t[i]:+10.5f} {rel:7.1f}%")
log("-" * 48)
log(f"{'ALL':>5s} {ori_all:10.5f} {ft_all:10.5f} {diff_all:+10.5f} {100*diff_all/ori_all:7.1f}%")

# coarse 4-band rollup
log("\n--- coarse bands (mean of per-t means) ---")
bands = [(0,250),(250,500),(500,750),(750,1000)]
band_rows = []
for lo, hi in bands:
    idx = [i for i, t in enumerate(T_LIST) if lo <= t < hi]
    o = float(ori_t[idx].mean()); f = float(ft_t[idx].mean())
    band_rows.append(dict(band=f"{lo}-{hi}", ori=o, ft=f, diff=o-f))
    log(f"  t[{lo:4d},{hi:4d})  ori={o:.5f}  ft={f:.5f}  ori-ft={o-f:+.5f}  ({100*(o-f)/o:.1f}%)")

out = dict(prompt=PROMPT, t_list=T_LIST, num_eps=NUM_EPS, seed=SEED,
    per_timestep=[dict(t=int(t), ori=float(ori_t[i]), ft=float(ft_t[i]),
                       diff=float(diff_t[i]), rel_pct=float(100*diff_t[i]/ori_t[i]))
                  for i, t in enumerate(T_LIST)],
    overall=dict(ori=ori_all, ft=ft_all, diff=diff_all, rel_pct=100*diff_all/ori_all),
    bands=band_rows)
json.dump(out, open(os.path.join(HERE, "realistic_by_timestep_results.json"), "w"), indent=2)
log(f"\n[done] wrote realistic_by_timestep_results.json")
