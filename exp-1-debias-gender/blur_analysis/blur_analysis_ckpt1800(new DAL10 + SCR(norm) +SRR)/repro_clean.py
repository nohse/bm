"""
CLEAN reproduction (addresses the 'ori looked broken' concern):
  ori      = plain Stable-Diffusion-v1.5 (standard StableDiffusionPipeline, NO attention_mask)
  finetune = same pipeline + EMA TE-LoRA exported by 2-export-checkpoint.py (text_encoder_lora_EMA.pth)
Same noise per (occupation, idx). Per-image metrics: CLIP-T (ViT-bigG-14, = eval metric),
SDS_f / SDS_m / SDS_realistic (base UNet+TE, unmasked), Laplacian variance, high-freq energy.

Goal the user asked for:
  - see images degrade while CLIP-T barely moves
  - see images degrade while SDS_realistic DROPS
"""
import os, json, math, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.loaders import LoraLoaderMixin
import open_clip

p = argparse.ArgumentParser()
p.add_argument("--ema", required=True, help="path to text_encoder_lora_EMA.pth (from 2-export-checkpoint.py)")
p.add_argument("--out", required=True)
p.add_argument("--n_occ", type=int, default=10)
p.add_argument("--n_per", type=int, default=10)
p.add_argument("--num_eps", type=int, default=4)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--save_imgs", action="store_true")
args = p.parse_args()

DEVICE="cuda"; DT=torch.float16; MODEL="runwayml/stable-diffusion-v1-5"; RANK=50
GUID=7.5; STEPS=25
SDS_TMIN,SDS_TMAX,SDS_NT=400,800,15; TAU=1e-4
os.makedirs(args.out, exist_ok=True)
def log(*a): print(*a, flush=True)

# ---------------- pipeline (ori = plain SD-1.5) ----------------
log("[load] SD-1.5 pipeline ...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=DT,
        safety_checker=None, requires_safety_checker=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(DEVICE); pipe.set_progress_bar_config(disable=True)
tokenizer = pipe.tokenizer; vae = pipe.vae; unet = pipe.unet

# base TE kept clean for SDS conditioning (independent of the pipe TE which we later LoRA-fy)
te_base = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEVICE, DT).eval().requires_grad_(False)
ddpm = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
alphas = ddpm.alphas_cumprod.to(DEVICE)

# ---------------- CLIP-T (ViT-bigG-14, same as eval Clip-T) ----------------
log("[load] open_clip ViT-bigG-14 ...")
clip_model,_,clip_pre = open_clip.create_model_and_transforms("ViT-bigG-14",
        pretrained="laion2b_s39b_b160k", device=DEVICE)
clip_model.eval().requires_grad_(False)
clip_tok = open_clip.get_tokenizer("ViT-bigG-14")

@torch.no_grad()
def clipT(pil_images, prompt):
    imgs = torch.stack([clip_pre(im) for im in pil_images]).to(DEVICE)
    f = F.normalize(clip_model.encode_image(imgs), dim=-1)
    t = F.normalize(clip_model.encode_text(clip_tok([prompt]*len(pil_images)).to(DEVICE)), dim=-1)
    return (f*t).sum(-1).float().cpu()

# ---------------- SDS (unmasked) using base TE + base UNet ----------------
@torch.no_grad()
def embeds(prompts, te):
    tok = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    return te(tok.input_ids.to(DEVICE), tok.attention_mask.to(DEVICE))[0].to(DT)

def unet_chunked(zt, tv, pe, chunk=48):
    outs=[]
    for s in range(0, zt.shape[0], chunk):
        outs.append(unet(zt[s:s+chunk], tv[s:s+chunk], encoder_hidden_states=pe[s:s+chunk]).sample)
    return torch.cat(outs,0)

@torch.no_grad()
def sds_eval(images_m1, num_eps):
    B=images_m1.shape[0]
    lat = vae.encode(images_m1.to(DT)).latent_dist.sample()*vae.config.scaling_factor
    t_idx = torch.linspace(SDS_TMIN,SDS_TMAX,SDS_NT,device=DEVICE).round().long()
    K=t_idx.shape[0]; E=num_eps
    lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(B,K,E,*lat.shape[1:]).contiguous().view(B*K*E,*lat.shape[1:])
    tv = t_idx.view(1,K,1).expand(B,K,E).reshape(-1)
    eps = torch.randn_like(lat_exp)
    ab = alphas[tv].to(lat_exp.dtype)
    zt = ab.sqrt().view(-1,1,1,1)*lat_exp + (1-ab).sqrt().view(-1,1,1,1)*eps
    pe_f=embeds(["a photo of a woman"]*zt.shape[0], te_base)
    pe_m=embeds(["a photo of a man"]*zt.shape[0], te_base)
    pe_r=embeds(["a photo of a realistic face"]*zt.shape[0], te_base)
    def per(pe):
        ep=unet_chunked(zt,tv,pe)
        err=((ep.float()-eps.float())**2).mean(1)
        return err.flatten(1).mean(1).view(B,K,E).mean((1,2))
    s_f,s_m,s_r=per(pe_f),per(pe_m),per(pe_r)
    pred=torch.stack([-s_f/TAU,-s_m/TAU],1).argmax(1)
    return dict(s_f=s_f.cpu(),s_m=s_m.cpu(),s_real=s_r.cpu(),pred=pred.cpu())

def blur_metrics(images_m1):
    x=(images_m1*0.5+0.5).clamp(0,1).float()
    g=(0.299*x[:,0]+0.587*x[:,1]+0.114*x[:,2])*255.0
    k=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=g.dtype,device=g.device).view(1,1,3,3)
    lap=F.conv2d(g.unsqueeze(1),k,padding=1).squeeze(1)
    lapvar=lap.flatten(1).var(1)
    G=torch.fft.fftshift(torch.fft.fft2(g),dim=(-2,-1)); mag=G.abs()**2
    B,H,W=g.shape
    yy,xx=torch.meshgrid(torch.arange(H,device=g.device),torch.arange(W,device=g.device),indexing="ij")
    r=torch.sqrt(((yy-H/2)**2+(xx-W/2)**2)).unsqueeze(0); rmax=math.sqrt((H/2)**2+(W/2)**2)
    hi=(r>0.25*rmax).float()
    hf=(mag*hi).flatten(1).sum(1)/(mag.flatten(1).sum(1)+1e-8)
    return lapvar.cpu(), hf.cpu()

# ---------------- prompts ----------------
with open("../data/1-prompts/occupation.json") as f: pdata=json.load(f)
tmpl=pdata["prompt_templates_test"][0]
occs=pdata["occupations_test_set"][:args.n_occ]
log(f"[prompts] template={tmpl!r}; {len(occs)} occupations x {args.n_per} imgs; CLIP=ViT-bigG-14; eps={args.num_eps}")

def gen(prompt, gens):
    out = pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUID,
               generator=gens, output_type="pt", num_images_per_prompt=len(gens))
    img01 = out.images.to(DEVICE)          # [N,3,512,512] in [0,1]
    return img01

def to_pil(img01):
    return [Image.fromarray((im.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)) for im in img01]

def run(tag, lora_on):
    imgdir=os.path.join(args.out,f"images_{tag}");
    if args.save_imgs: os.makedirs(imgdir, exist_ok=True)
    def slug(s): return "".join(c if c.isalnum() else "_" for c in s)[:40]
    rows=[]; from torchvision.utils import save_image
    for oi,occ in enumerate(occs):
        prompt=tmpl.format(occupation=occ)
        gens=[torch.Generator(device=DEVICE).manual_seed(args.seed+oi*1000+i) for i in range(args.n_per)]
        img01=gen(prompt,gens)
        img_m1=img01*2-1
        pil=to_pil(img01)
        ct=clipT(pil,prompt)
        sd=sds_eval(img_m1,args.num_eps)
        lv,hf=blur_metrics(img_m1)
        for i in range(args.n_per):
            base=f"o{oi:02d}_n{i:02d}_{slug(occ)}"
            if args.save_imgs: save_image(img01[i].cpu(), os.path.join(imgdir,base+".png"))
            rows.append(dict(occ=occ, idx=i, prompt=prompt, file=base,
                clipT=float(ct[i]), sds_f=float(sd["s_f"][i]), sds_m=float(sd["s_m"][i]),
                sds_real=float(sd["s_real"][i]), lapvar=float(lv[i]), hf_frac=float(hf[i]),
                sds_gender=("man" if int(sd["pred"][i])==1 else "woman")))
        log(f"  [{tag}] occ {oi+1}/{len(occs)} {occ[:26]:26s} clipT={ct.mean():.3f} real={sd['s_real'].mean():.4f} lapvar={lv.mean():.0f}")
    return rows

# Phase 1: ori (plain SD-1.5)
log("\n=== PHASE 1: ori (plain SD-1.5) ===")
rows_ori = run("ori", lora_on=False)

# Phase 2: apply EMA LoRA to the pipe text encoder, regenerate
log("\n=== PHASE 2: finetune (EMA TE-LoRA) ===")
lora_params = LoraLoaderMixin._modify_text_encoder(pipe.text_encoder, dtype=torch.float32, rank=RANK, patch_mlp=True)
ema = torch.load(args.ema, map_location="cpu")
res = pipe.text_encoder.load_state_dict(ema, strict=False)
n_loaded = len(ema); n_missing_lora = sum("lora" in k for k in res.missing_keys)
log(f"[lora] loaded {n_loaded} EMA params; missing_lora_keys={n_missing_lora}; unexpected={len(res.unexpected_keys)}")
assert n_missing_lora==0 and len(res.unexpected_keys)==0, "LoRA name mismatch!"
pipe.text_encoder.to(DEVICE)   # keep lora fp32 sublayers; base fp16
rows_ft = run("ft", lora_on=True)

# ---------------- aggregate table ----------------
def agg(rows):
    import statistics as st
    sf=np.array([r["sds_f"] for r in rows]); sm=np.array([r["sds_m"] for r in rows])
    male=np.mean([r["sds_gender"]=="man" for r in rows])
    sep=np.mean(np.abs(sf-sm)/((sf+sm)/2+1e-8))
    return dict(n=len(rows),
        clipT=float(np.mean([r["clipT"] for r in rows])),
        realistic_SDS=float(np.mean([r["sds_real"] for r in rows])),
        lapvar=float(np.mean([r["lapvar"] for r in rows])),
        hf_frac=float(np.mean([r["hf_frac"] for r in rows])),
        s_f=float(sf.mean()), s_m=float(sm.mean()),
        sep=float(sep), male_ratio=float(male), gender_gap=float(abs(2*male-1)))
A_o, A_f = agg(rows_ori), agg(rows_ft)
log("\n================= TABLE (clean, plain SD-1.5 vs EMA-LoRA) =================")
hdr=f"{'metric':22s} {'ori(SD1.5)':>14s} {'finetune(EMA)':>14s}"
log(hdr); log("-"*len(hdr))
for k in ["clipT","realistic_SDS","lapvar","hf_frac","s_f","s_m","sep","male_ratio","gender_gap"]:
    log(f"{k:22s} {A_o[k]:14.4f} {A_f[k]:14.4f}")

with open(os.path.join(args.out,"clean_results.json"),"w") as f:
    json.dump(dict(ori=A_o, ft=A_f, rows_ori=rows_ori, rows_ft=rows_ft), f, indent=2)
log(f"\n[done] wrote {os.path.join(args.out,'clean_results.json')}")
