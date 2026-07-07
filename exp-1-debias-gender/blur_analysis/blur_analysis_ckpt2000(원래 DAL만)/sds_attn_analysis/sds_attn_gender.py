#!/usr/bin/env python
# coding=utf-8
"""
Standalone SDS-gender classifier for a folder of already-generated images.

Faithfully replicates `sds_logits_from_images(..., region_mask_mode="attn",
use_attn_weight=True)` from
    exp-1-debias-gender/1-main-gender-sgd_dmscr_h_gen_check.py
i.e. the EVAL path with sds_text_encoder = frozen original CLIP text encoder and
sds_unet = base SD-1.5 UNet (this run sets train_unet=False, so the SDS classifier
is plain SD-1.5 and does NOT depend on any LoRA checkpoint).

For every image it computes:
  * SDS_f  = woman-prompt eps-prediction error, attn-weighted   ("a photo of a woman")
  * SDS_m  = man-prompt   eps-prediction error, attn-weighted   ("a photo of a man")
  * logits = [-SDS_f/tau, -SDS_m/tau]   (the "-softmax" of the SDS losses)
  * probs  = softmax(logits) = [P(woman), P(man)]
  * pred   = argmax  (0=woman, 1=man)

attn weighting: cross-attn maps focused on the 'woman' / 'man' token are captured on
EVERY cross-attention (attn2) layer of the UNet via forward hooks, head/layer/timestep
averaged, and the SAME map  w = 0.5*(att_woman + att_man)  weights both err_f and err_m
(identical to the source).

Same hyper-params as the source defaults:
  t in linspace(400, 800, 15) (rounded), num_eps = 1, tau = 1e-4, guidance not used,
  fp16 weights, DDPM forward noising.
"""
import os, json, math, argparse, glob, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.attention_processor import AttnProcessor

DEVICE = "cuda"
DT = torch.float16                 # weight_dtype (mixed_precision="fp16")
MODEL = "runwayml/stable-diffusion-v1-5"
SDS_TMIN, SDS_TMAX, SDS_NT = 400, 800, 15
SDS_NE = 1                          # sds_num_eps default
TAU = 1e-4                          # sds_tau default
FONT_PATH = "/workspace/finetune-fair-diffusion/data/0-utils/arial-bold.ttf"

def log(*a): print(*a, flush=True)


# ============================================================================
# CrossAttnCapture  --  copied verbatim from 1-main-gender-sgd_dmscr_h_gen_check.py
# ============================================================================
class CrossAttnCapture:
    def __init__(self, token_indices, use_cpu=False, expect_cfg_pair=True):
        self.token_indices = token_indices
        self.handles = []
        self.maps = []
        self.use_cpu = use_cpu
        self._q_cache = {}
        self._parent_map = {}
        self.use_cpu = use_cpu
        self.expect_cfg_pair = expect_cfg_pair

    def _q_hook(self, module, inputs, output):
        parent = self._parent_map.get(id(module))
        if parent is None:
            return
        parent_id = id(parent)
        try:
            with torch.no_grad():
                self._q_cache[parent_id] = output.detach().clone()
        except Exception:
            self._q_cache[parent_id] = output.detach()

    def _k_hook(self, module, inputs, output):
        parent = self._parent_map.get(id(module))
        if parent is None:
            return
        parent_id = id(parent)
        if parent_id not in self._q_cache:
            return
        q = self._q_cache.pop(parent_id)
        k = output
        device = q.device if not self.use_cpu else torch.device("cpu")
        with torch.no_grad():
            try:
                q = q.detach().float().to(device)
            except Exception:
                q = q.float().to(device)
            try:
                k = k.detach().float().to(device)
            except Exception:
                k = k.float().to(device)

            heads = getattr(parent, "heads", None)
            if heads is None:
                heads = getattr(parent, "num_heads", None)

            B, Nq, inner_q = q.shape
            _, Nk, inner_k = k.shape
            try:
                assert inner_q == inner_k
            except Exception:
                return

            if heads is None:
                for h in (8, 12, 16, 4, 6, 24, 32):
                    if inner_q % h == 0:
                        heads = h
                        break
                if heads is None:
                    heads = 8

            head_dim = inner_q // heads
            q = q.view(B, Nq, heads, head_dim).permute(0, 2, 1, 3).contiguous()
            k = k.view(B, Nk, heads, head_dim).permute(0, 2, 1, 3).contiguous()

            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
            attn = attn_scores.softmax(dim=-1)

            if not self.token_indices:
                return
            tok_idx = torch.tensor(self.token_indices, device=device, dtype=torch.long)
            attn_tok = attn.index_select(-1, tok_idx).mean(dim=-1)  # (B,H,Nq)

            if self.expect_cfg_pair and B >= 2:
                b_half = B // 2
                cond = attn_tok[B - b_half:B]
            else:
                cond = attn_tok

            per_image = cond.mean(dim=1)  # (B_sel, Nq)
            hw = int(math.sqrt(per_image.shape[1]))
            if hw * hw != per_image.shape[1]:
                return
            b_sel = per_image.shape[0]
            per_image = per_image.view(b_sel, hw, hw)
            if self.use_cpu:
                per_image = per_image.cpu()
            self.maps.append(per_image)

    def add_hooks(self, unet):
        installed = 0
        for name, module in unet.named_modules():
            has_qk = hasattr(module, "to_q") and hasattr(module, "to_k")
            if not has_qk:
                continue
            is_cross = getattr(module, "is_cross_attention", None)
            if is_cross is None:
                is_cross = ("attn2" in name) or ("Cross" in module.__class__.__name__)
            if not is_cross:
                continue
            try:
                self._parent_map[id(module.to_q)] = module
            except Exception:
                pass
            try:
                self._parent_map[id(module.to_k)] = module
            except Exception:
                pass
            self.handles.append(module.to_q.register_forward_hook(self._q_hook))
            self.handles.append(module.to_k.register_forward_hook(self._k_hook))
            installed += 1
        return self

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._q_cache = {}
        self.maps = []

    def aggregated_map(self):
        if not self.maps:
            return None
        max_hw = max(m.shape[-1] for m in self.maps)
        upsampled = []
        for m in self.maps:
            Bm, hm, wm = m.shape
            ten = m.unsqueeze(1).float()
            if hm != max_hw or wm != max_hw:
                ten = F.interpolate(ten, size=(max_hw, max_hw), mode="bilinear", align_corners=False)
            upsampled.append(ten.squeeze(1))
        S = torch.stack(upsampled, dim=0)
        M = S.mean(dim=0)
        return M


def find_token_positions(tokenizer, prompt, keywords):
    toks = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                     truncation=True, return_tensors="pt")
    ids = toks.input_ids[0]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    kws = [k.lower() for k in keywords]
    pos = []
    for i, piece in enumerate(pieces):
        p = piece.lower().replace("Ġ", "").replace("▁", "")
        if any(k in p for k in kws):
            pos.append(i)
    return pos


# ============================================================================
# model loading
# ============================================================================
log("[load] SD-1.5 (tokenizer / text_encoder / vae / unet / ddpm) ...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEVICE, DT).eval().requires_grad_(False)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(DEVICE, DT).eval().requires_grad_(False)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").to(DEVICE, DT).eval().requires_grad_(False)
ddpm = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")
alphas_cumprod = ddpm.alphas_cumprod.to(DEVICE)
log(f"[load] default unet attn processor: {type(next(iter(unet.attn_processors.values()))).__name__}")

FEMALE_PROMPT = "a photo of a woman"
MALE_PROMPT = "a photo of a man"
TOK_POS_F = find_token_positions(tokenizer, FEMALE_PROMPT, keywords=["woman"])
TOK_POS_M = find_token_positions(tokenizer, MALE_PROMPT, keywords=["man"])
log(f"[tok] woman-token pos in '{FEMALE_PROMPT}' = {TOK_POS_F}; man-token pos in '{MALE_PROMPT}' = {TOK_POS_M}")


@torch.no_grad()
def text_embeds(prompts):
    toks = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                     truncation=True, return_tensors="pt")
    return text_encoder(toks.input_ids.to(DEVICE), toks.attention_mask.to(DEVICE))[0].to(DT)


@torch.no_grad()
def sds_gender_attn(images_m1):
    """images_m1: [B,3,512,512] in [-1,1].  returns dict of per-image tensors (cpu)."""
    B = images_m1.shape[0]
    lat = vae.encode(images_m1.to(DT)).latent_dist.sample() * vae.config.scaling_factor  # [B,4,64,64]

    t_idx = torch.linspace(SDS_TMIN, SDS_TMAX, steps=SDS_NT, device=DEVICE).round().long()
    K = t_idx.shape[0]
    lat_exp = lat.unsqueeze(1).unsqueeze(2).expand(B, K, SDS_NE, *lat.shape[1:]).contiguous().view(B * K * SDS_NE, *lat.shape[1:])
    t_vec = t_idx.view(1, K, 1).expand(B, K, SDS_NE).reshape(-1)
    BKE = lat_exp.shape[0]

    eps = torch.randn_like(lat_exp, dtype=DT)
    ab = alphas_cumprod[t_vec].to(lat_exp.dtype)
    zt = ab.sqrt().view(-1, 1, 1, 1) * lat_exp + (1.0 - ab).sqrt().view(-1, 1, 1, 1) * eps

    pe_f = text_embeds([FEMALE_PROMPT] * BKE)
    pe_m = text_embeds([MALE_PROMPT] * BKE)

    _H, _W = zt.shape[-2], zt.shape[-1]

    def forward_with_attn(prompt_embeds, token_positions):
        cap = CrossAttnCapture(token_indices=token_positions, use_cpu=False, expect_cfg_pair=False).add_hooks(unet)
        try:
            ep = unet(zt.to(DT), t_vec, encoder_hidden_states=prompt_embeds).sample.to(DT)
            att = cap.aggregated_map()  # [BKE, hw, hw]
        finally:
            cap.clear()
        return ep, att

    eps_pred_f, att_f_bke = forward_with_attn(pe_f, TOK_POS_F)
    eps_pred_m, att_m_bke = forward_with_attn(pe_m, TOK_POS_M)

    # --- attn weight map (region_mask_mode == "attn", use_attn_weight=True) ---
    a_f = att_f_bke.to(device=DEVICE, dtype=torch.float32)
    a_m = att_m_bke.to(device=DEVICE, dtype=torch.float32)
    if a_f.shape[-2:] != (_H, _W):
        a_f = F.interpolate(a_f.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)
    if a_m.shape[-2:] != (_H, _W):
        a_m = F.interpolate(a_m.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)
    att_f = a_f.reshape(B, K, SDS_NE, _H, _W).mean(dim=(1, 2))  # [B,H,W]
    att_m = a_m.reshape(B, K, SDS_NE, _H, _W).mean(dim=(1, 2))  # [B,H,W]
    attmap_mean = 0.5 * (att_f + att_m)                          # [B,H,W]
    w_map = (attmap_mean.unsqueeze(1).unsqueeze(2)
             .expand(B, K, SDS_NE, _H, _W).contiguous().view(BKE, _H, _W).to(DT))

    err_f = ((eps_pred_f - eps) ** 2).mean(dim=1)  # [BKE,H,W]
    err_m = ((eps_pred_m - eps) ** 2).mean(dim=1)

    w = w_map.to(err_f.dtype).clamp_min(0)
    w_sum = w.flatten(1).sum(dim=1).clamp_min(1e-8)
    hw = err_f.shape[-2] * err_f.shape[-1]
    w = (w / w_sum.view(-1, 1, 1)) * hw
    sds_f_per = (err_f * w).flatten(1).mean(dim=1)  # [BKE]
    sds_m_per = (err_m * w).flatten(1).mean(dim=1)

    with torch.autocast("cuda", enabled=False):
        sds_f32 = sds_f_per.float().view(B, K, SDS_NE).mean(dim=(1, 2))  # [B]
        sds_m32 = sds_m_per.float().view(B, K, SDS_NE).mean(dim=(1, 2))
        tau32 = torch.tensor(TAU, device=DEVICE, dtype=torch.float32)
        logits32 = torch.stack([-sds_f32 / tau32, -sds_m32 / tau32], dim=1)  # [B,2]
        logits32 = logits32 - logits32.max(dim=1, keepdim=True).values
        probs32 = torch.softmax(logits32, dim=1)
    preds = probs32.argmax(dim=1)

    return dict(
        sds_f=sds_f32.cpu(), sds_m=sds_m32.cpu(),
        p_woman=probs32[:, 0].cpu(), p_man=probs32[:, 1].cpu(),
        pred=preds.cpu(),                       # 0=woman, 1=man
        attmap=attmap_mean.float().cpu(),       # [B,64,64]
    )


# ============================================================================
# image io + annotation
# ============================================================================
def load_img_m1(path):
    im = Image.open(path).convert("RGB")
    if im.size != (512, 512):
        im = im.resize((512, 512), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0  # [3,H,W] 0..1
    return x * 2 - 1  # [-1,1]


def get_font(sz):
    try:
        return ImageFont.truetype(FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


def annotate(path, r, strip_h=104):
    """Return a PIL image = original (bordered by predicted gender) + caption strip below."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    pred_woman = (r["pred"] == 0)
    border_col = (220, 40, 40) if pred_woman else (40, 90, 220)  # red=woman, blue=man (repo convention)
    bw = 8
    canvas = Image.new("RGB", (W + 2 * bw, H + 2 * bw + strip_h), border_col)
    canvas.paste(im, (bw, bw))
    # white caption area
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([bw, H + bw, W + bw, H + bw + strip_h], fill=(255, 255, 255))
    f1 = get_font(26)
    f2 = get_font(24)
    pred_str = "WOMAN" if pred_woman else "MAN"
    line1 = f"SDS_f(woman)={r['sds_f']:.5f}   SDS_m(man)={r['sds_m']:.5f}"
    line2 = f"P(woman)={r['p_woman']:.3f}  P(man)={r['p_man']:.3f}  -> {pred_str}"
    draw.text((bw + 10, H + bw + 8), line1, fill=(0, 0, 0), font=f1)
    draw.text((bw + 10, H + bw + 48), line2, fill=border_col, font=f2)
    return canvas


def montage(annotated, cols, save_to, pad=6, bg=(30, 30, 30)):
    if not annotated:
        return
    w, h = annotated[0].size
    rows = math.ceil(len(annotated) / cols)
    grid = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, im in enumerate(annotated):
        rr, cc = divmod(i, cols)
        grid.paste(im, (pad + cc * (w + pad), pad + rr * (h + pad)))
    grid.save(save_to)


# ============================================================================
# run over a folder
# ============================================================================
def occ_of(fname):
    # o00_n00_senator.png -> ("o00", "senator")
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    return parts[0], "_".join(parts[2:])


def run_set(img_dir, out_dir, tag, batch=10):
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not files:
        log(f"[skip] no png in {img_dir}")
        return None
    ann_dir = os.path.join(out_dir, f"annotated_{tag}")
    os.makedirs(ann_dir, exist_ok=True)
    rows = []
    annotated_by_occ = {}
    log(f"\n=== {tag}: {len(files)} images from {img_dir} ===")
    for s in range(0, len(files), batch):
        chunk = files[s:s + batch]
        imgs = torch.stack([load_img_m1(p) for p in chunk]).to(DEVICE)
        torch.manual_seed(1234 + s)  # deterministic eps per chunk
        out = sds_gender_attn(imgs)
        for i, p in enumerate(chunk):
            fname = os.path.basename(p)
            occ_id, occ = occ_of(fname)
            r = dict(
                file=fname, occ_id=occ_id, occ=occ,
                sds_f=float(out["sds_f"][i]), sds_m=float(out["sds_m"][i]),
                p_woman=float(out["p_woman"][i]), p_man=float(out["p_man"][i]),
                pred=int(out["pred"][i]),
                pred_gender=("woman" if int(out["pred"][i]) == 0 else "man"),
            )
            rows.append(r)
            ann = annotate(p, r)
            ann.save(os.path.join(ann_dir, fname))
            annotated_by_occ.setdefault(occ_id, []).append(ann)
        done = min(s + batch, len(files))
        sf = np.mean([row["sds_f"] for row in rows]); sm = np.mean([row["sds_m"] for row in rows])
        log(f"  [{tag}] {done}/{len(files)}  running mean SDS_f={sf:.5f} SDS_m={sm:.5f}")

    # per-occupation montages (10 per row)
    mont_dir = os.path.join(out_dir, f"montage_{tag}")
    os.makedirs(mont_dir, exist_ok=True)
    for occ_id in sorted(annotated_by_occ):
        occ_name = next(r["occ"] for r in rows if r["occ_id"] == occ_id)
        montage(annotated_by_occ[occ_id], cols=10, save_to=os.path.join(mont_dir, f"{occ_id}_{occ_name}.png"))
    # one big montage (all)
    all_ann = [a for occ_id in sorted(annotated_by_occ) for a in annotated_by_occ[occ_id]]
    montage(all_ann, cols=10, save_to=os.path.join(out_dir, f"montage_all_{tag}.png"))
    return rows


# ============================================================================
# analysis
# ============================================================================
def stats(rows, label):
    sf = np.array([r["sds_f"] for r in rows]); sm = np.array([r["sds_m"] for r in rows])
    pw = np.array([r["p_woman"] for r in rows]); pm = np.array([r["p_man"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    n = len(rows)
    woman_ratio = float((pred == 0).mean()); man_ratio = float((pred == 1).mean())
    d = dict(
        label=label, n=n,
        sds_f_mean=float(sf.mean()), sds_f_std=float(sf.std()),
        sds_m_mean=float(sm.mean()), sds_m_std=float(sm.std()),
        sds_m_minus_f_mean=float((sm - sf).mean()),
        p_woman_mean=float(pw.mean()), p_man_mean=float(pm.mean()),
        pred_woman_ratio=woman_ratio, pred_man_ratio=man_ratio,
        gender_gap=float(abs(man_ratio - woman_ratio)),
    )
    return d


def per_occ_table(rows):
    occs = {}
    for r in rows:
        occs.setdefault((r["occ_id"], r["occ"]), []).append(r)
    out = []
    for (oid, occ), rs in sorted(occs.items()):
        sf = np.mean([r["sds_f"] for r in rs]); sm = np.mean([r["sds_m"] for r in rs])
        nw = sum(r["pred"] == 0 for r in rs); nm = sum(r["pred"] == 1 for r in rs)
        out.append(dict(occ_id=oid, occ=occ, n=len(rs), sds_f=float(sf), sds_m=float(sm),
                        n_woman=int(nw), n_man=int(nm),
                        p_woman_mean=float(np.mean([r["p_woman"] for r in rs]))))
    return out


def fmt_stats(d):
    L = []
    L.append(f"  n images                : {d['n']}")
    L.append(f"  SDS_f (woman) mean±std  : {d['sds_f_mean']:.5f} ± {d['sds_f_std']:.5f}")
    L.append(f"  SDS_m (man)   mean±std  : {d['sds_m_mean']:.5f} ± {d['sds_m_std']:.5f}")
    L.append(f"  SDS_m - SDS_f  mean     : {d['sds_m_minus_f_mean']:+.5f}   (>0 => man-error larger => leans woman)")
    L.append(f"  P(woman) mean           : {d['p_woman_mean']:.4f}")
    L.append(f"  P(man)   mean           : {d['p_man_mean']:.4f}")
    L.append(f"  predicted woman / man   : {d['pred_woman_ratio']*100:.1f}%  /  {d['pred_man_ratio']*100:.1f}%")
    L.append(f"  gender gap |man-woman|  : {d['gender_gap']*100:.1f}%")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/finetune-fair-diffusion/exp-1-debias-gender/blur_analysis_ckpt2000(원래 DAL만)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = args.root
    out_dir = args.out or os.path.join(root, "sds_attn_analysis")
    os.makedirs(out_dir, exist_ok=True)

    sets = [
        ("ori", os.path.join(root, "clean", "images_ori")),
        ("ft",  os.path.join(root, "clean", "images_ft")),
    ]
    all_rows = {}
    for tag, d in sets:
        if os.path.isdir(d):
            all_rows[tag] = run_set(d, out_dir, tag)

    # analysis
    report = []
    report.append("SDS-gender (attn-weighted) analysis")
    report.append(f"  hyper-params: t=linspace({SDS_TMIN},{SDS_TMAX},{SDS_NT}).round(), num_eps={SDS_NE}, tau={TAU}, fp16")
    report.append("  region_mask_mode=attn, use_attn_weight=True  (cross-attn maps for woman/man tokens, all attn2 layers)")
    report.append("  logits = [-SDS_f/tau, -SDS_m/tau]; probs = softmax(logits); pred=argmax (0=woman,1=man)")
    report.append("")
    results = {"hyper": dict(t_min=SDS_TMIN, t_max=SDS_TMAX, num_t=SDS_NT, num_eps=SDS_NE, tau=TAU,
                             region_mask_mode="attn", use_attn_weight=True)}
    for tag in ("ori", "ft"):
        if tag not in all_rows or all_rows[tag] is None:
            continue
        rows = all_rows[tag]
        title = "PHASE: images_ori (plain SD-1.5)" if tag == "ori" else "PHASE: images_ft (finetuned EMA TE-LoRA)"
        st = stats(rows, tag)
        report.append("=" * 64)
        report.append(title)
        report.append("=" * 64)
        report.append(fmt_stats(st))
        report.append("")
        report.append("  per-occupation (mean SDS_f / SDS_m, predicted #woman/#man):")
        report.append(f"    {'occ':38s} {'SDS_f':>8s} {'SDS_m':>8s} {'#W':>3s} {'#M':>3s} {'P(w)':>6s}")
        for o in per_occ_table(rows):
            report.append(f"    {o['occ'][:38]:38s} {o['sds_f']:8.5f} {o['sds_m']:8.5f} "
                          f"{o['n_woman']:3d} {o['n_man']:3d} {o['p_woman_mean']:6.3f}")
        report.append("")
        results[tag] = dict(summary=st, per_occupation=per_occ_table(rows), rows=rows)

    report_txt = "\n".join(report)
    log("\n" + report_txt)
    with open(os.path.join(out_dir, "ANALYSIS.txt"), "w") as f:
        f.write(report_txt + "\n")
    with open(os.path.join(out_dir, "sds_attn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n[done] outputs in: {out_dir}")
    log("       annotated_<tag>/  per-image annotated PNGs")
    log("       montage_<tag>/    per-occupation montages")
    log("       montage_all_<tag>.png, ANALYSIS.txt, sds_attn_results.json")


if __name__ == "__main__":
    main()
