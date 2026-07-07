"""
Reproduce eval of checkpoint-1800 (TE-only LoRA on SD-1.5) and probe:
  Q1: realistic-SDS loss for ori (sharp base) vs finetune (LoRA) on SAME noise.
      -> does the "realistic face" SDS loss go DOWN for blurry images?
  Q3: SDS gender classification (s_f, s_m, argmax) for ori vs finetune.
      -> does fair-loss-down actually produce a real 50/50 split, or do s_f~=s_m collapse?
  + objective blur metrics (Laplacian variance, high-freq FFT energy) to confirm blur.

Faithful to 1-main-debias-DAL_SCRnorm.py:
  ori      = base text_encoder + base unet          (eval line 3002)
  finetune = EMA-LoRA text_encoder + base unet       (eval line 3097)
  generation: DPMSolverMultistep, 25 steps, CFG 7.5  (num_denoising_steps=25)
  realistic/gender SDS: base unet + base text_encoder, t in [400,800], num_t=15, tau=1e-4
"""
import os, sys, json, math, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (AutoencoderKL, UNet2DConditionModel,
                       DPMSolverMultistepScheduler, DDPMScheduler)
from diffusers.loaders import LoraLoaderMixin

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n_prompts", type=int, default=10)
p.add_argument("--n_per", type=int, default=5)
p.add_argument("--num_eps", type=int, default=4)   # training used 1; >1 lowers variance
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--also_skip", action="store_true")  # also gen skip-50% (the training regime)
p.add_argument("--save_imgs", action="store_true")   # save every individual ori/ft image + pair
args = p.parse_args()

DEVICE = "cuda"
DT = torch.float16
MODEL = "runwayml/stable-diffusion-v1-5"
RANK = 50
GUIDANCE = 7.5
N_STEPS = 25
SDS_T_MIN, SDS_T_MAX, SDS_NUM_T = 400, 800, 15
TAU = 1e-4
os.makedirs(args.out, exist_ok=True)

def log(*a):
    print(*a, flush=True)

torch.manual_seed(args.seed); np.random.seed(args.seed)

# ----------------------- load models -----------------------
log("[load] models ...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
# finetune TE (gets LoRA); base TE for ori + for SDS conditioning
te_lora = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder")
te_base = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder")
vae  = AutoencoderKL.from_pretrained(MODEL, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet")
sched = DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler")
ddpm_forward = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")

# build LoRA on te_lora EXACTLY like training, then load EMA shadow params
lora_params = LoraLoaderMixin._modify_text_encoder(te_lora, dtype=torch.float32, rank=RANK, patch_mlp=True)
ema = torch.load(os.path.join(args.ckpt, "custom_checkpoint_0.pkl"), map_location="cpu")
shadow = ema["shadow_params"]
assert len(shadow) == len(lora_params), (len(shadow), len(lora_params))
with torch.no_grad():
    for q, s in zip(lora_params, shadow):
        q.data.copy_(s.data.to(q.dtype))
log(f"[load] applied EMA LoRA: {len(shadow)} params, ema.optimization_step={ema.get('optimization_step')}")

# precisions: LoRA TE in fp32 (lora math), others fp16. embeds cast to fp16 for unet.
te_lora.to(DEVICE, dtype=torch.float32).eval().requires_grad_(False)
te_base.to(DEVICE, dtype=DT).eval().requires_grad_(False)
vae.to(DEVICE, dtype=DT).eval().requires_grad_(False)
unet.to(DEVICE, dtype=DT).eval().requires_grad_(False)
alphas_cumprod = ddpm_forward.alphas_cumprod.to(DEVICE)

# ----------------------- helpers -----------------------
@torch.no_grad()
def embeds(prompts, te, dtype):
    tok = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    ids = tok.input_ids.to(DEVICE); am = tok.attention_mask.to(DEVICE)
    return te(ids, am)[0].to(dtype)

@torch.no_grad()
def gen(prompt, noises, te, skip_pct=0.0):
    """faithful generate_image_no_gradient: DPMSolver, CFG 7.5, optional final-step skip via x0-jump."""
    N = noises.shape[0]
    # prompt embeds with padding=True (match eval path) -> but to share uncond max_length we use max_length padding
    pe = embeds([prompt]*N, te, DT)
    ue = embeds([""]*N, te, DT)
    cat = torch.cat([ue, pe]).to(DT)
    sched.set_timesteps(N_STEPS)
    ts = sched.timesteps
    total = len(ts)
    skip = int(math.floor(total*skip_pct/100.0 + 0.5)) if skip_pct > 0 else 0
    skip = min(skip, total-1)
    run = total - skip
    lat = noises.clone()
    for i, t in enumerate(ts[:run]):
        mi = torch.cat([lat]*2).to(DT)
        mi = sched.scale_model_input(mi, t)
        ep = unet(mi, t, encoder_hidden_states=cat).sample.float()
        eu, ec = ep.chunk(2)
        ep = eu + GUIDANCE*(ec-eu)
        lat = sched.step(ep, t, lat).prev_sample
    if skip > 0:
        t_cur = ts[run]
        mi = torch.cat([lat]*2).to(DT); mi = sched.scale_model_input(mi, t_cur)
        ep = unet(mi, t_cur, encoder_hidden_states=cat).sample.float()
        eu, ec = ep.chunk(2); ep = eu + GUIDANCE*(ec-eu)
        so = sched.step(ep, t_cur, lat)
        lat = so.pred_original_sample if getattr(so, "pred_original_sample", None) is not None else lat
    img = vae.decode((lat/vae.config.scaling_factor).to(DT)).sample.clamp(-1,1)
    return img  # [N,3,512,512] in [-1,1]

def unet_chunked(zt, t_vec, pe, chunk=64):
    outs = []
    for s in range(0, zt.shape[0], chunk):
        outs.append(unet(zt[s:s+chunk], t_vec[s:s+chunk],
                         encoder_hidden_states=pe[s:s+chunk]).sample)
    return torch.cat(outs, 0)

@torch.no_grad()
def sds_eval(images, num_eps):
    """unmasked SDS. returns dict of per-image [B] tensors: s_f, s_m, s_real, pred(0=woman,1=man)."""
    B = images.shape[0]
    lat = vae.encode(images.to(DT)).latent_dist.sample() * vae.config.scaling_factor  # [B,4,64,64]
    t_idx = torch.linspace(SDS_T_MIN, SDS_T_MAX, SDS_NUM_T, device=DEVICE).round().long()
    K = t_idx.shape[0]; E = num_eps
    lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(B,K,E,*lat.shape[1:]).contiguous().view(B*K*E,*lat.shape[1:])
    t_vec = t_idx.view(1,K,1).expand(B,K,E).reshape(-1)
    BKE = lat_exp.shape[0]
    eps = torch.randn_like(lat_exp)
    ab = alphas_cumprod[t_vec].to(lat_exp.dtype)
    zt = ab.sqrt().view(-1,1,1,1)*lat_exp + (1-ab).sqrt().view(-1,1,1,1)*eps
    pe_f = embeds(["a photo of a woman"]*BKE, te_base, DT)
    pe_m = embeds(["a photo of a man"]*BKE, te_base, DT)
    pe_r = embeds(["a photo of a realistic face"]*BKE, te_base, DT)
    epf = unet_chunked(zt, t_vec, pe_f); epm = unet_chunked(zt, t_vec, pe_m); epr = unet_chunked(zt, t_vec, pe_r)
    def per(ep):
        err = ((ep.float()-eps.float())**2).mean(dim=1)  # [BKE,H,W]
        return err.flatten(1).mean(dim=1).view(B,K,E).mean(dim=(1,2))  # [B]
    s_f, s_m, s_r = per(epf), per(epm), per(epr)
    logits = torch.stack([-s_f/TAU, -s_m/TAU], 1)
    pred = logits.argmax(1)
    return dict(s_f=s_f.cpu(), s_m=s_m.cpu(), s_real=s_r.cpu(), pred=pred.cpu())

def blur_metrics(images):
    """images [B,3,512,512] in [-1,1]. Laplacian variance + high-freq FFT energy fraction (per image)."""
    x = (images*0.5+0.5).clamp(0,1).float()
    g = (0.299*x[:,0]+0.587*x[:,1]+0.114*x[:,2])*255.0  # [B,H,W] gray on 0-255 scale
    # Laplacian variance (standard blur metric on 0-255 images)
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=g.dtype, device=g.device).view(1,1,3,3)
    lap = F.conv2d(g.unsqueeze(1), k, padding=1).squeeze(1)
    lapvar = lap.flatten(1).var(dim=1)
    # high-freq energy fraction via FFT (fp32!)
    G = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2,-1))
    mag = G.abs()**2
    B,H,W = g.shape
    yy, xx = torch.meshgrid(torch.arange(H,device=g.device), torch.arange(W,device=g.device), indexing="ij")
    r = torch.sqrt(((yy-H/2)**2 + (xx-W/2)**2)).unsqueeze(0)
    rmax = math.sqrt((H/2)**2+(W/2)**2)
    hi = (r > 0.25*rmax).float()  # high-freq ring
    hf_frac = (mag*hi).flatten(1).sum(1) / (mag.flatten(1).sum(1)+1e-8)
    return lapvar.cpu(), hf_frac.cpu()

# ----------------------- prompts -----------------------
with open("../data/1-prompts/occupation.json") as f:
    pdata = json.load(f)
tmpl = pdata["prompt_templates_test"][0]
occs = pdata["occupations_test_set"][:args.n_prompts]
prompts = [tmpl.format(occupation=o) for o in occs]
log(f"[prompts] template={tmpl!r}; {len(prompts)} prompts; {args.n_per} imgs each; eps={args.num_eps}")

# ----------------------- run -----------------------
def run_mode(skip_pct, tag):
    acc = {k: [] for k in ["s_f","s_m","s_real","pred"]}
    acc_o = {k: [] for k in ["s_f","s_m","s_real","pred"]}
    lap_o,lap_f,hf_o,hf_f = [],[],[],[]
    saved = []
    manifest = []
    from torchvision.utils import save_image as _save
    def _slug(s): return "".join(c if c.isalnum() else "_" for c in s)[:48]
    imgdir = None
    if args.save_imgs:
        imgdir = os.path.join(args.out, f"images_{tag}"); os.makedirs(imgdir, exist_ok=True)
    def _w(t): return (t*0.5+0.5).clamp(0,1).float().cpu()
    for pi, prompt in enumerate(prompts):
        gen_seed = args.seed + pi
        g = torch.Generator(device=DEVICE).manual_seed(gen_seed)
        noises = torch.randn([args.n_per,4,64,64], device=DEVICE, generator=g, dtype=torch.float32)
        img_o = gen(prompt, noises, te_base, skip_pct=skip_pct)
        img_f = gen(prompt, noises, te_lora, skip_pct=skip_pct)
        ro, rf = sds_eval(img_o, args.num_eps), sds_eval(img_f, args.num_eps)
        for k in acc: acc[k].append(rf[k]); acc_o[k].append(ro[k])
        lo,ho = blur_metrics(img_o); lf,hf = blur_metrics(img_f)
        lap_o.append(lo); lap_f.append(lf); hf_o.append(ho); hf_f.append(hf)
        if pi < 6: saved.append((prompt, img_o[0], img_f[0]))
        if args.save_imgs:
            occ = _slug(prompt.replace("A photo of the face of a ","").replace(", a person",""))
            for ni in range(img_o.shape[0]):
                base = f"p{pi:02d}_n{ni}_{occ}"
                _save(_w(img_o[ni]), os.path.join(imgdir, base+"_ori.png"))
                _save(_w(img_f[ni]), os.path.join(imgdir, base+"_ft.png"))
                _save(_w(torch.cat([img_o[ni], img_f[ni]], dim=2)), os.path.join(imgdir, base+"_pair.png"))
                pm = 'man' if int(rf['pred'][ni])==1 else 'woman'; om = 'man' if int(ro['pred'][ni])==1 else 'woman'
                manifest.append(dict(file=base, prompt=prompt, noise_seed=gen_seed, noise_idx=ni,
                    ori_realistic=float(ro['s_real'][ni]), ft_realistic=float(rf['s_real'][ni]),
                    ori_lapvar=float(lo[ni]), ft_lapvar=float(lf[ni]),
                    ori_s_f=float(ro['s_f'][ni]), ori_s_m=float(ro['s_m'][ni]),
                    ft_s_f=float(rf['s_f'][ni]), ft_s_m=float(rf['s_m'][ni]),
                    ori_sds_gender=om, ft_sds_gender=pm))
        log(f"  [{tag}] {pi+1}/{len(prompts)} {prompt[:40]!r} "
            f"real ori={ro['s_real'].mean():.4f} ft={rf['s_real'].mean():.4f} | "
            f"lapvar ori={lo.mean():.1f} ft={lf.mean():.1f}")
    if args.save_imgs:
        with open(os.path.join(imgdir,"manifest.json"),"w") as mf: json.dump(manifest, mf, indent=2)
        log(f"[save] {len(manifest)} ori + {len(manifest)} ft images -> {imgdir} (+manifest.json)")
    A = {k: torch.cat(v) for k,v in acc.items()}
    O = {k: torch.cat(v) for k,v in acc_o.items()}
    lap_o=torch.cat(lap_o); lap_f=torch.cat(lap_f); hf_o=torch.cat(hf_o); hf_f=torch.cat(hf_f)

    def split(P):
        n=P.numel(); m=(P==1).float().mean().item(); return m, 1-m
    mo, wo = split(O["pred"]); mf, wf = split(A["pred"])
    res = dict(
        tag=tag, n=int(A["pred"].numel()),
        realistic_SDS_ori=float(O["s_real"].mean()), realistic_SDS_ft=float(A["s_real"].mean()),
        s_f_ori=float(O["s_f"].mean()), s_m_ori=float(O["s_m"].mean()),
        s_f_ft=float(A["s_f"].mean()), s_m_ft=float(A["s_m"].mean()),
        gender_gap_ori=abs(mo-wo), gender_gap_ft=abs(mf-wf),
        male_ratio_ori=mo, male_ratio_ft=mf,
        # |s_f - s_m| normalized by their mean: how separable are the two genders (collapse if ~0)
        sds_sep_ori=float(((O["s_f"]-O["s_m"]).abs()/((O["s_f"]+O["s_m"])/2+1e-8)).mean()),
        sds_sep_ft=float(((A["s_f"]-A["s_m"]).abs()/((A["s_f"]+A["s_m"])/2+1e-8)).mean()),
        lapvar_ori=float(lap_o.mean()), lapvar_ft=float(lap_f.mean()),
        hf_frac_ori=float(hf_o.mean()), hf_frac_ft=float(hf_f.mean()),
    )
    log(f"\n===== RESULT [{tag}] (n={res['n']}) =====")
    for k,v in res.items():
        if k not in ("tag","n"): log(f"  {k:22s}: {v:.4f}")
    # save montage
    try:
        from torchvision.utils import save_image
        rows=[]
        for prompt,io,iff in saved:
            rows.append(torch.cat([io,iff],dim=2))  # side by side: ori | ft
        grid=torch.cat(rows,dim=1)
        save_image((grid*0.5+0.5).clamp(0,1), os.path.join(args.out,f"montage_{tag}.png"))
        log(f"[save] montage_{tag}.png (left=ori, right=finetune)")
    except Exception as e:
        log("montage save failed:", e)
    return res

with torch.cuda.device(0):
    results = {}
    results["full25"] = run_mode(0.0, "full25")
    if args.also_skip:
        results["skip50"] = run_mode(50.0, "skip50")

with open(os.path.join(args.out,"results.json"),"w") as f:
    json.dump(results, f, indent=2)
log("\n[done] wrote", os.path.join(args.out,"results.json"))
