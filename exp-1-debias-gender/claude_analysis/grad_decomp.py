"""
Faithful per-loss gradient decomposition for the TE-LoRA debias training step.
Extracts the EXACT loss functions (generate_image_w_gradient, sds_logits_from_images, etc.)
from the training script via AST, runs one real training-step batch on a single GPU,
and reports how much each of {fair, img(hspace), face(realistic)} contributes to the
gradient on the text-encoder LoRA params (norm + direction/cosine + cancellation).
"""
import os, sys, math, random, itertools, ast, textwrap, types
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import scipy.stats
from typing import List, Optional

torch.backends.cuda.matmul.allow_tf32 = True
SRC = "/workspace/finetune-fair-diffusion/exp-1-debias-gender/1-main-gender-sgd_dmscr_h_gen_check.py"
DEVICE = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES pins the physical GPU
MODEL = "runwayml/stable-diffusion-v1-5"

# ---------------- extract needed defs from the source (no transcription drift) ----------------
source = open(SRC).read()
tree = ast.parse(source)
WANT_TOP = {"_AttnStore", "CrossAttnCapture", "MidBlockCapture", "make_grad_hook"}
WANT_NESTED = {"generate_image_no_gradient", "generate_image_w_gradient", "_text_embeds",
               "find_token_positions", "sds_logits_from_images",
               "generate_dynamic_targets", "gen_dynamic_weights_sds"}
segments = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANT_TOP:
        segments[node.name] = ast.get_source_segment(source, node)
mainfn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
for sub in mainfn.body:
    if isinstance(sub, (ast.FunctionDef, ast.ClassDef)) and sub.name in WANT_NESTED:
        segments[sub.name] = textwrap.dedent(ast.get_source_segment(source, sub))
missing = (WANT_TOP | WANT_NESTED) - set(segments)
assert not missing, f"missing defs: {missing}"
print("[extract] pulled defs:", sorted(segments))

# ---------------- runtime globals the closures expect ----------------
class Acc:
    device = DEVICE; mixed_precision = "fp16"; is_main_process = True
    num_processes = 1; local_process_index = 0
    def print(self, *a, **k): pass
    def wait_for_everyone(self): pass
    def backward(self, loss): loss.backward()
accelerator = Acc()
weight_dtype = torch.float16
weight_dtype_high_precision = torch.float32

class A: pass
args = A()
args.guidance_scale = 7.5
args.sds_tau = 1e-4
args.sds_t_min = 400; args.sds_t_max = 800; args.sds_num_t = 15; args.sds_num_eps = 1
args.skip_final_steps = 0; args.skip_final_steps_pct = 50.0
args.target_male_ratio = 0.5; args.uncertainty_threshold = 0.2; args.factor1 = 0.2
args.h_loss_form = "raw"
args.use_attn_weight = True
args.attn_grad_threshold = 0.2
args.region_mask_mode = os.environ.get("REGION", "none")   # 'none' or 'attn'

# ---------------- load models exactly as the script does ----------------
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.loaders import LoraLoaderMixin

print("[load] SD1.5 ...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet")
noise_scheduler = DPMSolverMultistepScheduler.from_config(MODEL, subfolder="scheduler")
ddpm_forward = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
text_encoder.requires_grad_(False); unet.requires_grad_(False); vae.requires_grad_(False)
unet.enable_gradient_checkpointing(); vae.enable_gradient_checkpointing()
text_encoder.to(DEVICE, dtype=weight_dtype); unet.to(DEVICE, dtype=weight_dtype); vae.to(DEVICE, dtype=weight_dtype)
eval_text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder")
eval_text_encoder.requires_grad_(False); eval_text_encoder.to(DEVICE, dtype=weight_dtype)

# TE-LoRA exactly as in the script (fp32 lora params, rank 50, patch_mlp)
RANK = 50
text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=RANK, patch_mlp=True)
text_encoder_lora_params = list(text_encoder_lora_params)
CE_loss = nn.CrossEntropyLoss(reduction="none")
print(f"[load] TE-LoRA params: {len(text_encoder_lora_params)} tensors, "
      f"{sum(p.numel() for p in text_encoder_lora_params):,} scalars")

# ---------------- exec extracted defs into this module's globals ----------------
G = globals()
for name in ["_AttnStore", "make_grad_hook", "CrossAttnCapture", "MidBlockCapture",
             "_text_embeds", "find_token_positions", "generate_dynamic_targets",
             "gen_dynamic_weights_sds", "generate_image_no_gradient",
             "generate_image_w_gradient", "sds_logits_from_images"]:
    exec(segments[name], G)

# ---------------- one faithful training step ----------------
def flat_grad(scalar, params):
    gs = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return torch.cat([ (g if g is not None else torch.zeros_like(p)).reshape(-1)
                       for g, p in zip(gs, params) ]).float()

def run(region, w_img, w_face, seed=5991, N=6, chunk=2, steps=12, tau_mode="fixed"):
    torch.cuda.empty_cache()
    args.region_mask_mode = region
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    prompt = "A photo of the face of a critic, a person"
    noises = torch.randn([N, 4, 64, 64], generator=g, device=DEVICE, dtype=weight_dtype_high_precision)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
        # Step 1: targets from FULLY-denoised no-grad images (skip=0), like L3500/3512/3565
        if True:
            imgs_full, _ = generate_image_no_gradient(prompt, noises, steps,
                                which_text_encoder=text_encoder, which_unet=unet,
                                skip_final_steps=0, skip_final_steps_pct=0.0)
            preds0, probs0, logits0, sds_f0, sds_m0, _ = sds_logits_from_images(
                imgs_full, tau=args.sds_tau, t_min=args.sds_t_min, t_max=args.sds_t_max,
                num_t=args.sds_num_t, num_eps=args.sds_num_eps, gate_grad=False,
                sds_text_encoder=eval_text_encoder, sds_unet=unet, sds_scheduler=ddpm_forward,
                compute_realistic_sds=False)
    # gender contrastive signal g = sds_m - sds_f over the batch (this is what tau divides)
    g0 = (sds_m0 - sds_f0).float()
    std_g = g0.std().item(); mean_g = g0.mean().item()
    tau_use = std_g if tau_mode == "adaptive" else args.sds_tau   # 'adaptive' = z-score scale
    targets_all, unc_all = generate_dynamic_targets(probs0, target_male_ratio=0.5, w_uncertainty=True)
    targets_all[unc_all > args.uncertainty_threshold] = -1
    with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):

        # take one train_GPU_batch_size chunk (mirrors the N_backward loop's first chunk)
        idxs = list(range(chunk))
        noises_ij = noises[idxs]; targets_ij = targets_all[idxs]

        # Step 4: gradient generation (skip=50) + hspace, then SDS-with-grad (+realistic)
        imgs_ij, _, loss_hspace_ij = generate_image_w_gradient(prompt, noises_ij, steps,
                            which_text_encoder=text_encoder, which_unet=unet,
                            skip_final_steps=args.skip_final_steps,
                            skip_final_steps_pct=args.skip_final_steps_pct,
                            return_hspace_loss=True,
                            h_reference_text_encoder=eval_text_encoder, h_reference_unet=unet)
        loss_hspace_ij = loss_hspace_ij.to(weight_dtype)
        preds_ij, probs_ij, logits_ij, _, _, sds_realistic_ij = sds_logits_from_images(
            imgs_ij, tau=tau_use, t_min=args.sds_t_min, t_max=args.sds_t_max,
            num_t=args.sds_num_t, num_eps=args.sds_num_eps, gate_grad=True,
            sds_text_encoder=eval_text_encoder, sds_unet=unet, sds_scheduler=ddpm_forward,
            compute_realistic_sds=True)

    # --- loss assembly OUTSIDE autocast (matches real training: CE returns fp16) ---
    loss_fair_ij = torch.ones(len(idxs), dtype=weight_dtype, device=DEVICE) * (-1)
    idxs_valid = (targets_ij != -1).nonzero().view(-1)
    logits_h = logits_ij.half()
    if idxs_valid.numel() > 0:
        loss_fair_ij[idxs_valid] = CE_loss(logits_h[idxs_valid], targets_ij[idxs_valid])
    loss_face_ij = sds_realistic_ij.to(weight_dtype) if sds_realistic_ij is not None \
                   else torch.zeros(len(idxs), dtype=weight_dtype, device=DEVICE)
    dyn = gen_dynamic_weights_sds(targets_ij, preds_ij, factor=args.factor1, out_dtype=loss_hspace_ij.dtype)

    params = text_encoder_lora_params
    # per-term gradient vectors (as each enters loss_ij.mean())
    g_fair = flat_grad(loss_fair_ij.mean(), params)
    g_img_raw = flat_grad(loss_hspace_ij.mean(), params)                 # unweighted, no dyn
    g_img_eff = flat_grad((dyn * loss_hspace_ij).mean(), params)         # dyn-weighted (pre w_img)
    g_face_raw = flat_grad(loss_face_ij.mean(), params)
    g_img_w = w_img * g_img_eff
    g_face_w = w_face * g_face_raw
    g_total = g_fair + g_img_w + g_face_w

    def nrm(v): return v.norm().item()
    def cos(a, b):
        na, nb = a.norm(), b.norm()
        return (torch.dot(a, b) / (na * nb + 1e-20)).item() if na > 0 and nb > 0 else float("nan")

    print("\n" + "=" * 78)
    print(f"REGION={region}  w_img={w_img}  w_face={w_face}  seed={seed}  TAU_MODE={tau_mode}  tau_use={tau_use:.3e}  | valid targets={idxs_valid.numel()}/{len(idxs)}")
    print(f"  gender signal g=sds_m-sds_f over batch(N={len(g0)}): mean={mean_g:.3e} std={std_g:.3e} min={g0.min():.3e} max={g0.max():.3e}  (fixed tau={args.sds_tau:.0e})")
    print(f"  targets_ij={targets_ij.tolist()}  preds_sds={preds_ij.tolist()}  dyn_w={[round(x,2) for x in dyn.tolist()]}")
    print(f"  probs_sds[:,male]={[round(x,3) for x in probs_ij[:,1].tolist()]}  (loss_fair={loss_fair_ij[idxs_valid].mean().item():.4f}, ln2={math.log(2):.4f})")
    print(f"  loss_hspace(mean)={loss_hspace_ij.mean().item():.4f}   loss_realistic(mean)={loss_face_ij.mean().item():.5f}  (loss_face.requires_grad={loss_face_ij.requires_grad})")
    print("-" * 78)
    print("  RAW per-term grad norm on TE-LoRA (unweighted signal strength):")
    print(f"    ||g_fair||       = {nrm(g_fair):.4e}")
    print(f"    ||g_img_raw||    = {nrm(g_img_raw):.4e}   (hspace, no dyn, no w_img)")
    print(f"    ||g_img_eff||    = {nrm(g_img_eff):.4e}   (dyn-weighted, pre w_img)")
    print(f"    ||g_face_raw||   = {nrm(g_face_raw):.4e}   (realistic SDS)")
    print("  AS ACTUALLY WEIGHTED into loss_ij (this is what the optimizer sees):")
    print(f"    ||g_fair||       = {nrm(g_fair):.4e}   share={nrm(g_fair)/ (nrm(g_fair)+nrm(g_img_w)+nrm(g_face_w))*100:.1f}%")
    print(f"    ||w_img*g_img||  = {nrm(g_img_w):.4e}   share={nrm(g_img_w)/(nrm(g_fair)+nrm(g_img_w)+nrm(g_face_w))*100:.1f}%")
    print(f"    ||w_face*g_face||= {nrm(g_face_w):.4e}   share={nrm(g_face_w)/(nrm(g_fair)+nrm(g_img_w)+nrm(g_face_w))*100:.1f}%")
    print(f"    ||g_total||      = {nrm(g_total):.4e}")
    print("  DIRECTION (cosine): do the terms fight each other?")
    print(f"    cos(fair, img_w)  = {cos(g_fair, g_img_w):+.3f}")
    print(f"    cos(fair, face_w) = {cos(g_fair, g_face_w):+.3f}")
    print(f"    cos(img_w, face_w)= {cos(g_img_w, g_face_w):+.3f}")
    print(f"    cos(fair, total)  = {cos(g_fair, g_total):+.3f}   cos(img_w, total)={cos(g_img_w, g_total):+.3f}")
    # how much of fair's push survives after img+face (projection of total onto fair dir)
    if g_fair.norm() > 0:
        proj = torch.dot(g_total, g_fair) / g_fair.norm()
        print(f"    fair-dir component of total = {proj.item():+.4e}  (vs ||g_fair||={nrm(g_fair):.4e})")
    sys.stdout.flush()
    return dict(region=region, gf=nrm(g_fair), gi=nrm(g_img_w), gc=nrm(g_face_w))

if __name__ == "__main__":
    # match run_600iter.sh: region=attn, w_img=2, w_face=0 ; also test w_face=4 (config default) and region=none
    # reproduce the ORIGINAL 3-seed table exactly (fixed tau=1e-4, N=4, chunk=2)
    for sd in [5991, 7, 42]:
        run("attn", w_img=2.0, w_face=4.0, N=4, chunk=2, seed=sd, tau_mode="fixed")
    print("\n[done]")
