#!/usr/bin/env python
"""
STANDALONE diagnostic — pick the hard-masking threshold for the gender cross-attention map.

This does NOT touch the training file and involves NO SCR / no finetuning. For a few
generated images it reproduces EXACTLY the scorer's attention map:
  * woman/man class-token cross-attention (attn2 layers only),
  * timesteps = linspace(residual_t_min, residual_t_max, residual_num_timesteps).round(),
  * a fresh eps per timestep, zt = scheduler.add_noise(z0, eps, t),
  * captured maps resized to (64,64) and averaged over {timestep, prompt, block, head},
  * spatially normalized to SUM=1 per image  (== `common_attn` used by the scorer).

Then it builds the HARD mask on the PER-IMAGE MAX-NORMALIZED map:
      gate = common / common.amax()          # in [0,1], does NOT change common's sum-to-1 values
      mask = gate >= threshold
and sweeps threshold in [--thr_start .. --thr_end] step --thr_step, so you can SEE and MEASURE
which threshold best isolates the face/gender region.

Outputs (in --out):
  * attn_thr_grid_*.jpg : rows = images, cols = [gen | common-attn | (face bbox) | thr=0.10 .. 0.50]
  * attn_thr_summary.txt: per-threshold numbers -> area fraction, and (if a face detector is
                          available) face-coverage / background-spill / IoU. Use these to pick.

Run:
  python attn_threshold_experiment.py --out attn_thr_out --num_imgs 8 --seed 0
  # optional: --prompts "a photo of a doctor" "a photo of a nurse" --dtype fp16
"""
import os, math, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompts", nargs="+",
                   default=["a photo of a doctor", "a photo of a firefighter",
                            "a photo of a chef", "a photo of a nurse"])
    p.add_argument("--num_imgs", type=int, default=8, help="total images across prompts")
    p.add_argument("--gen_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    # scorer attmap settings (match the training defaults)
    p.add_argument("--residual_t_min", type=int, default=400)
    p.add_argument("--residual_t_max", type=int, default=800)
    p.add_argument("--residual_num_timesteps", type=int, default=15)
    p.add_argument("--woman_prompt", default="a photo of a woman")
    p.add_argument("--man_prompt", default="a photo of a man")
    p.add_argument("--woman_word", default="woman")
    p.add_argument("--man_word", default="man")
    p.add_argument("--attn_res", type=int, default=0,
                   help="if >0, aggregate ONLY the cross-attn maps whose native grid == this size "
                        "(e.g. 16 = the most semantic layers); 0 = all blocks (faithful to the scorer)")
    # threshold sweep on the max-normalized map
    p.add_argument("--norm", default="minmax", choices=["minmax", "max"],
                   help="how to rescale the sum-to-1 map before thresholding. 'minmax'=(a-min)/(max-min) "
                        "(matches attmap_threshold_mask_visualization.py; removes the diffuse floor so low "
                        "thresholds localize); 'max'=a/max (keeps the floor).")
    p.add_argument("--thr_start", type=float, default=0.10)
    p.add_argument("--thr_end", type=float, default=0.50)
    p.add_argument("--thr_step", type=float, default=0.05)
    p.add_argument("--out", default="attn_thr_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--no_face", action="store_true", help="skip insightface face-bbox detection / IoU")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cross-attention capture processor — copied verbatim from the training file so
# the attmap is computed identically. Installed on cross-attention (attn2) only.
# ---------------------------------------------------------------------------
class _Ctx:
    def __init__(self):
        self.enabled = False
        self.token_idxs = None
        self.store = []


class CrossAttnCaptureProcessor:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        bsz, seq, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, seq, bsz)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  # [B*heads, q_hw, k_tokens]
        if self.ctx.enabled and self.ctx.token_idxs is not None:
            with torch.no_grad():
                col = attention_probs[..., self.ctx.token_idxs].mean(dim=-1)  # [B*heads, q_hw]
            self.ctx.store.append((col.detach(), attn.heads))
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def main():
    args = parse_args()
    device = args.device
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    thresholds = [round(args.thr_start + i * args.thr_step, 4)
                  for i in range(int(round((args.thr_end - args.thr_start) / args.thr_step)) + 1)]
    print(f"[setup] thresholds = {thresholds}")

    # ---- load frozen SD ----
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder").to(device, dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.pretrained, subfolder="vae").to(device, dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(args.pretrained, subfolder="unet").to(device, dtype).eval()
    scheduler = DPMSolverMultistepScheduler.from_config(args.pretrained, subfolder="scheduler")
    assert scheduler.config.prediction_type == "epsilon", "scorer assumes epsilon-prediction"
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    # install capture processor on cross-attention (attn2) layers only
    ctx = _Ctx()
    procs = dict(unet.attn_processors)
    for name in list(procs.keys()):
        if name.endswith("attn2.processor"):
            procs[name] = CrossAttnCaptureProcessor(ctx)
    unet.set_attn_processor(procs)

    # ---- text helpers ----
    @torch.no_grad()
    def encode(prompt):
        tok = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        emb = text_encoder(tok.input_ids.to(device), tok.attention_mask.to(device))[0]
        return emb.to(dtype)

    def find_word_idxs(prompt, word):
        pid = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True).input_ids
        wid = tokenizer(word, add_special_tokens=False).input_ids
        L = len(wid)
        for i in range(len(pid) - L + 1):
            if pid[i:i + L] == wid:
                return list(range(i, i + L))
        raise ValueError(f"could not find '{word}' {wid} in '{prompt}' -> {pid}")

    woman_emb = encode(args.woman_prompt)
    man_emb = encode(args.man_prompt)
    uncond_emb = encode("")
    woman_idx = find_word_idxs(args.woman_prompt, args.woman_word)
    man_idx = find_word_idxs(args.man_prompt, args.man_word)
    print(f"[setup] woman token idxs={woman_idx}, man token idxs={man_idx}")

    # ---- generation (CFG, DPM-Solver), returns image[-1,1] and z0 (UNet-scale latent) ----
    @torch.no_grad()
    def generate(prompt, n, gen):
        text = encode(prompt).expand(n, -1, -1)
        emb = torch.cat([uncond_emb.expand(n, -1, -1), text])
        scheduler.set_timesteps(args.gen_steps, device=device)
        lat = torch.randn((n, 4, 64, 64), generator=gen, device=device, dtype=dtype)
        for t in scheduler.timesteps:
            inp = scheduler.scale_model_input(torch.cat([lat] * 2), t)
            noise = unet(inp, t, encoder_hidden_states=emb).sample
            nu, nt = noise.chunk(2)
            noise = nu + args.guidance_scale * (nt - nu)
            lat = scheduler.step(noise, t, lat).prev_sample
        z0 = lat
        img = vae.decode((1 / vae.config.scaling_factor) * z0.to(dtype)).sample.clamp(-1, 1)
        return img, z0

    # ---- common attmap (faithful to compute_gender_attmaps): [n,64,64], sum-to-1 ----
    @torch.no_grad()
    def common_attmap(z0):
        n, H, W = z0.shape[0], z0.shape[-2], z0.shape[-1]
        ts = torch.linspace(args.residual_t_min, args.residual_t_max,
                            args.residual_num_timesteps, device=device).round().long()
        accum = {"woman": torch.zeros(n, H, W, device=device),
                 "man": torch.zeros(n, H, W, device=device)}
        counts = {"woman": 0, "man": 0}
        cfg = [("woman", woman_emb.expand(n, -1, -1), woman_idx),
               ("man", man_emb.expand(n, -1, -1), man_idx)]
        for t in ts:
            tb = t.repeat(n)
            eps = torch.randn_like(z0)
            zt = scheduler.add_noise(z0, eps, tb).to(dtype)
            for cls, c, idx in cfg:
                ctx.store = []
                ctx.token_idxs = idx
                ctx.enabled = True
                _ = unet(zt, tb, encoder_hidden_states=c).sample
                ctx.enabled = False
                for col, heads in ctx.store:
                    hw = col.shape[-1]
                    s = int(round(math.sqrt(hw)))
                    if args.attn_res > 0 and s != args.attn_res:
                        continue
                    a = col.view(n, heads, s, s).float()
                    a = torch.nn.functional.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                    accum[cls] = accum[cls] + a.mean(dim=1)
                    counts[cls] += 1
                ctx.store = []
        woman = accum["woman"] / max(counts["woman"], 1)
        man = accum["man"] / max(counts["man"], 1)
        common = (woman + man) / 2.0
        common = common / (common.sum(dim=(1, 2), keepdim=True) + 1e-8)  # SUM=1 per image
        return common  # [n,64,64]

    # ---- optional face detector (for IoU / coverage numbers) ----
    face_app = None
    if not args.no_face:
        try:
            from insightface.app import FaceAnalysis
            face_app = FaceAnalysis(allowed_modules=["detection"])
            face_app.prepare(ctx_id=0, det_size=(512, 512))
            print("[setup] insightface face detector ready (IoU/coverage will be reported)")
        except Exception as e:
            print(f"[setup] no face detector ({e}); skipping bbox/IoU, area-fraction only")

    def detect_bbox(img_tensor):
        """img_tensor [3,512,512] in [-1,1] -> [x0,y0,x1,y1] in 512px or None."""
        if face_app is None:
            return None
        arr = ((img_tensor * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        faces = face_app.get(arr[:, :, ::-1])  # expects BGR
        if not faces:
            return None
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return [float(v) for v in faces[-1].bbox]

    def bbox_grid_mask(bbox, H=64, W=64, img_px=512):
        s = img_px / H
        x0, y0, x1, y1 = [v / s for v in bbox]
        m = torch.zeros(H, W)
        xi0, yi0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        xi1, yi1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
        if xi1 > xi0 and yi1 > yi0:
            m[yi0:yi1, xi0:xi1] = 1
        return m

    # ---- overlay helpers ----
    def to_pil(img_tensor):
        return transforms.ToPILImage()((img_tensor * 0.5 + 0.5).clamp(0, 1).cpu())

    def label(pil, text):
        p = pil.copy()
        d = ImageDraw.Draw(p)
        d.rectangle([0, 0, len(text) * 7 + 8, 16], fill=(0, 0, 0))
        d.text((4, 3), text, fill="white")
        return p

    def heat_overlay(att_hw, img_tensor):
        H, W = img_tensor.shape[-2:]
        a = torch.nn.functional.interpolate(att_hw[None, None].float(), size=(H, W),
                                            mode="bilinear", align_corners=False)[0, 0]
        a = (a / (a.max() + 1e-8)).clamp(0, 1).cpu().numpy()
        base = np.asarray(to_pil(img_tensor)).astype(np.float32) / 255.0
        heat = np.stack([a, np.zeros_like(a), 1 - a], axis=-1)  # red=high, blue=low
        out = (0.55 * base + 0.45 * heat)
        return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))

    def mask_overlay(mask_hw, img_tensor):
        H, W = img_tensor.shape[-2:]
        m = torch.nn.functional.interpolate(mask_hw[None, None].float(), size=(H, W),
                                            mode="nearest")[0, 0].cpu()
        img01 = (img_tensor * 0.5 + 0.5).clamp(0, 1).cpu()
        out = img01 * (0.30 + 0.70 * m)          # dim the KEPT (unmasked) region to 30%
        out[0] = torch.clamp(out[0] + 0.35 * m, 0, 1)  # red tint on the RELEASED (masked) region
        return transforms.ToPILImage()(out)

    # ---- run ----
    per_prompt = max(1, math.ceil(args.num_imgs / len(args.prompts)))
    cell = 200
    rows = []
    # accumulators for the numeric summary
    area = {t: [] for t in thresholds}
    cover = {t: [] for t in thresholds}   # fraction of the face bbox that the mask releases
    spill = {t: [] for t in thresholds}   # fraction of the mask that lies OUTSIDE the face bbox
    iou = {t: [] for t in thresholds}
    peak_maxmean = []                     # attmap peakiness: max/mean (flat~1, peaked>>1)
    peak_mass50 = []                      # fraction of cells covering 50% of attention mass

    n_done = 0
    for p_idx, prompt in enumerate(args.prompts):
        if n_done >= args.num_imgs:
            break
        n = min(per_prompt, args.num_imgs - n_done)
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * p_idx)
        images, z0 = generate(prompt, n, gen)
        common = common_attmap(z0)                                   # [n,64,64] sum-to-1
        if args.norm == "minmax":
            cmin = common.amin(dim=(1, 2), keepdim=True)
            cmax = common.amax(dim=(1, 2), keepdim=True)
            gate = ((common - cmin) / (cmax - cmin + 1e-8)).clamp(0, 1)  # subtract the floor
        else:
            gate = common / (common.amax(dim=(1, 2), keepdim=True) + 1e-8)  # keep the floor

        for i in range(n):
            img = images[i]
            ci = common[i].flatten()
            peak_maxmean.append((ci.max() / (ci.mean() + 1e-8)).item())
            sc = torch.sort(ci, descending=True).values
            c50 = int((torch.cumsum(sc, 0) >= 0.5 * sc.sum()).float().argmax().item()) + 1
            peak_mass50.append(c50 / ci.numel())
            bbox = detect_bbox(img)
            bmask = bbox_grid_mask(bbox) if bbox is not None else None

            panels = [label(to_pil(img), f"gen p{p_idx}#{i}"),
                      label(heat_overlay(common[i], img), "common-attn")]
            bbox_pil = to_pil(img)
            if bbox is not None:
                ImageDraw.Draw(bbox_pil).rectangle(bbox, outline="lime", width=3)
            panels.append(label(bbox_pil, "face bbox" if bbox is not None else "no face"))

            for thr in thresholds:
                mask = (gate[i] >= thr).float().cpu()             # [64,64]
                a = mask.mean().item()
                area[thr].append(a)
                if bmask is not None:
                    inter = (mask * bmask).sum().item()
                    union = ((mask + bmask) > 0).float().sum().item()
                    cover[thr].append(inter / (bmask.sum().item() + 1e-8))
                    spill[thr].append((mask.sum().item() - inter) / (mask.sum().item() + 1e-8))
                    iou[thr].append(inter / (union + 1e-8))
                panels.append(label(mask_overlay(mask, img), f"thr={thr:.2f} a={a*100:.0f}%"))

            row = Image.new("RGB", (cell * len(panels), cell), "black")
            for j, pl in enumerate(panels):
                row.paste(pl.resize((cell, cell)), (j * cell, 0))
            rows.append(row)
            n_done += 1

    # save grid
    if rows:
        gw, gh = rows[0].size
        grid = Image.new("RGB", (gw, gh * len(rows)), "black")
        for i, r in enumerate(rows):
            grid.paste(r, (0, i * gh))
        grid_path = os.path.join(args.out, f"attn_thr_grid_seed{args.seed}.jpg")
        grid.save(grid_path, quality=92)
        print(f"[out] grid saved -> {grid_path}")

    # numeric summary -> the "number" to pick
    have_face = any(len(iou[t]) for t in thresholds)
    lines = ["threshold sweep summary (averaged over images)",
             f"images={n_done}  attmap: {args.woman_prompt} / {args.man_prompt}, "
             f"t=[{args.residual_t_min},{args.residual_t_max}] x{args.residual_num_timesteps}, "
             f"attn_res={'all' if args.attn_res == 0 else args.attn_res}, norm={args.norm}",
             f"attmap peakiness: max/mean={np.mean(peak_maxmean):.2f} (flat~1, peaked>>1); "
             f"cells for 50% mass={np.mean(peak_mass50) * 100:.1f}% (flat~50%, peaked<<50%)",
             ""]
    header = f"{'thr':>6} | {'mask_area%':>10}"
    if have_face:
        header += f" | {'face_cover%':>11} | {'bg_spill%':>9} | {'IoU':>6}"
    lines += [header, "-" * len(header)]
    for t in thresholds:
        row = f"{t:>6.2f} | {np.mean(area[t]) * 100:>10.1f}"
        if have_face and len(iou[t]):
            row += (f" | {np.mean(cover[t]) * 100:>11.1f}"
                    f" | {np.mean(spill[t]) * 100:>9.1f}"
                    f" | {np.mean(iou[t]):>6.3f}")
        lines.append(row)
    if have_face:
        best = max(thresholds, key=lambda t: (np.mean(iou[t]) if len(iou[t]) else -1))
        lines += ["",
                  f"best-IoU threshold = {best:.2f}  (IoU={np.mean(iou[best]):.3f}, "
                  f"face_cover={np.mean(cover[best]) * 100:.0f}%, bg_spill={np.mean(spill[best]) * 100:.0f}%)",
                  "note: 'face_cover' = how much of the face box gets released (want HIGH);",
                  "      'bg_spill'  = how much of the released mask is background (want LOW).",
                  "      pick the threshold trading these off; if IoU stays low at every threshold,",
                  "      the cross-attn is too diffuse to localize the face -> use the face bbox instead."]
    summary = "\n".join(lines)
    print("\n" + summary + "\n")
    with open(os.path.join(args.out, "attn_thr_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"[out] summary saved -> {os.path.join(args.out, 'attn_thr_summary.txt')}")


if __name__ == "__main__":
    main()
