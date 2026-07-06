#!/usr/bin/env python3
"""
BLIP gender-detection failure-rate experiment.

Goal: find the (prompt + input) configuration that MINIMIZES the rate at which
BLIP fails to assign a gender (the pipeline's `pred == -1`, plus face-detection
misses for the crop-based configs).

Same N generated images are reused across 3 configurations:
  exp1 : face-crop  + "the gender of the face is"     (current pipeline)
  exp2 : face-crop  + "the gender of the photo is"
  exp3 : full image + "the gender of the photo is"    (no face crop)

Faithful to 1-main-gender-sgd_dmscr_h_gen_check_blip.py:
  - base model runwayml/stable-diffusion-v1-5, DPMSolver 25 steps
  - prompt template "A photo of the face of a {occupation}, a person"
  - face detect = insightface buffalo_l (+ face_recognition fallback), crop via
    expand_bbox(coef=0.5)+crop_face(224) — identical helpers copied from the script
  - BLIP = Salesforce/blip-image-captioning-large, same caption->gender parser
"""
import os, json, re, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchvision
from PIL import Image

# ----------------------------- helpers copied verbatim from the main script ----
def expand_bbox(bbox, expand_coef, target_ratio):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    current_ratio = bbox_height / bbox_width
    if current_ratio > target_ratio:
        more_height = bbox_height * expand_coef
        more_width = (bbox_height + more_height) / target_ratio - bbox_width
    else:
        more_width = bbox_width * expand_coef
        more_height = (bbox_width + more_width) * target_ratio - bbox_height
    bbox_new = [0, 0, 0, 0]
    bbox_new[0] = int(round(bbox[0] - more_width * 0.5))
    bbox_new[2] = int(round(bbox[2] + more_width * 0.5))
    bbox_new[1] = int(round(bbox[1] - more_height * 0.5))
    bbox_new[3] = int(round(bbox[3] + more_height * 0.5))
    return bbox_new

def crop_face(img_tensor, bbox_new, target_size, fill_value):
    img_height, img_width = img_tensor.shape[-2:]
    idx_left = max(bbox_new[0], 0)
    idx_right = min(bbox_new[2], img_width)
    idx_bottom = max(bbox_new[1], 0)
    idx_top = min(bbox_new[3], img_height)
    pad_left = max(-bbox_new[0], 0)
    pad_right = max(-(img_width - bbox_new[2]), 0)
    pad_top = max(-bbox_new[1], 0)
    pad_bottom = max(-(img_height - bbox_new[3]), 0)
    img_face = img_tensor[:, idx_bottom:idx_top, idx_left:idx_right]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img_face = torchvision.transforms.Pad([pad_left, pad_top, pad_right, pad_bottom], fill=fill_value)(img_face)
    img_face = torchvision.transforms.Resize(size=target_size)(img_face)
    return img_face

# ----------------------------- gender parser (same as get_face_gender_blip) -----
RE_MALE = re.compile(r"\b(?:male|males|man|men|boy|boys|gentleman|gentlemen|guy|guys|he|him|his|masculine)\b")
RE_FEMALE = re.compile(r"\b(?:female|females|woman|women|girl|girls|lady|ladies|she|her|hers|feminine)\b")
def parse_gender(caption):
    c = caption.lower()
    nm = len(RE_MALE.findall(c)); nf = len(RE_FEMALE.findall(c))
    if nm > nf:   return 1, nm, nf
    if nf > nm:   return 0, nm, nf
    return -1, nm, nf

# ----------------------------- args --------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--out", type=str, default="blip_exp_out")
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--base", type=str, default="runwayml/stable-diffusion-v1-5")
ap.add_argument("--occ", type=str, default="../data/1-prompts/occupation.json")
ap.add_argument("--blip", type=str, default="Salesforce/blip-image-captioning-large")
ap.add_argument("--size_face", type=int, default=224)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"[setup] device={device} dtype={dtype} n={args.n}")

# ----------------------------- 1) generate N images ----------------------------
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
with open(args.occ) as f:
    occ_data = json.load(f)
template = occ_data["prompt_templates_test"][0]
occs = occ_data["occupations_test_set"] if "occupations_test_set" in occ_data else occ_data["occupations_train_set"]
# deterministic spread of occupations
sel_occ = [occs[(i * 7919) % len(occs)] for i in range(args.n)]

print(f"[gen] loading {args.base} ...")
pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=dtype, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

pil_images = []
meta = []
for i, occ in enumerate(sel_occ):
    prompt = template.format(occupation=occ)
    g = torch.Generator(device=device).manual_seed(1000 + i)
    img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g).images[0]
    pil_images.append(img)
    meta.append({"idx": i, "occupation": occ, "prompt": prompt, "seed": 1000 + i})
    img.save(os.path.join(args.out, "images", f"{i:03d}_{occ.replace('/', '-')[:40]}.png"))
    if (i + 1) % 10 == 0:
        print(f"[gen] {i+1}/{args.n}")
del pipe
torch.cuda.empty_cache()
print(f"[gen] done: {len(pil_images)} images, size={pil_images[0].size}")

# ----------------------------- 2) face detect + crop ---------------------------
from insightface.app import FaceAnalysis
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

try:
    import face_recognition
    HAVE_FR = True
except Exception:
    HAVE_FR = False
print(f"[face] insightface ready; face_recognition fallback={HAVE_FR}")

def largest_bbox_insight(faces, W, H):
    best, area = None, -1
    for fc in faces:
        b = fc["bbox"]
        a = (min(b[2], W) - max(b[0], 0)) * (min(b[3], H) - max(b[1], 0))
        if a > area:
            area, best = a, b
    return best

crop_pils = []          # face-crop PIL or None
face_detected = []      # bool
for i, img in enumerate(pil_images):
    rgb = np.array(img.convert("RGB"))           # HxWx3 RGB uint8
    H, W = rgb.shape[:2]
    bbox = None
    faces = face_app.get(rgb[:, :, ::-1])        # insightface wants BGR
    if len(faces) > 0:
        bbox = largest_bbox_insight(faces, W, H)
    elif HAVE_FR:                                 # fallback, mirrors get_face
        locs = face_recognition.face_locations(rgb)  # list of (top,right,bottom,left)
        if len(locs) > 0:
            t, r, b, l = max(locs, key=lambda L: (L[2]-L[0]) * (L[1]-L[3]))
            bbox = [l, t, r, b]
    if bbox is None:
        crop_pils.append(None)
        face_detected.append(False)
        continue
    bbox_e = expand_bbox(bbox, expand_coef=0.5, target_ratio=1)
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float()       # [3,H,W] in [0,255]
    chip = crop_face(img_t, bbox_e, target_size=[args.size_face, args.size_face], fill_value=0)
    chip_pil = Image.fromarray(chip.clamp(0, 255).byte().permute(1, 2, 0).numpy())
    crop_pils.append(chip_pil)
    face_detected.append(True)
    chip_pil.save(os.path.join(args.out, "crops", f"{i:03d}.png"))

n_face = sum(face_detected)
print(f"[face] detected {n_face}/{args.n}  (no-face {args.n - n_face})")

# ----------------------------- 3) BLIP -----------------------------------------
from transformers import BlipProcessor, BlipForConditionalGeneration
print(f"[blip] loading {args.blip} ...")
bp = BlipProcessor.from_pretrained(args.blip)
bm = BlipForConditionalGeneration.from_pretrained(args.blip, torch_dtype=dtype).to(device)
bm.eval()

@torch.no_grad()
def caption(images_pil, prompt_text):
    """Return list of captions for the given PIL images with one conditional prompt."""
    caps = []
    BS = 16
    for k in range(0, len(images_pil), BS):
        batch = images_pil[k:k + BS]
        inp = bp(images=batch, text=[prompt_text] * len(batch), return_tensors="pt")
        inp = {kk: vv.to(device) for kk, vv in inp.items()}
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(dtype)
        out = bm.generate(**inp, max_new_tokens=20, num_beams=1)
        caps.extend(bp.batch_decode(out, skip_special_tokens=True))
    return caps

# ----------------------------- 4) run 3 configs --------------------------------
EXPS = [
    {"key": "exp1_facecrop_face", "input": "crop", "prompt": "the gender of the face is"},
    {"key": "exp2_facecrop_photo", "input": "crop", "prompt": "the gender of the photo is"},
    {"key": "exp3_fullimg_photo",  "input": "full", "prompt": "the gender of the photo is"},
]

results = {}
for e in EXPS:
    key, mode, prompt_text = e["key"], e["input"], e["prompt"]
    if mode == "crop":
        idxs = [i for i in range(args.n) if face_detected[i]]
        imgs = [crop_pils[i] for i in idxs]
    else:
        idxs = list(range(args.n))
        imgs = pil_images

    caps = caption(imgs, prompt_text) if len(imgs) > 0 else []

    per = []
    n_male = n_female = n_undecided = 0
    undecided_recs = []
    for j, i in enumerate(idxs):
        pred, nm, nf = parse_gender(caps[j])
        per.append({"idx": i, "occupation": meta[i]["occupation"], "caption": caps[j],
                    "pred": pred, "n_male": nm, "n_female": nf})
        if pred == 1: n_male += 1
        elif pred == 0: n_female += 1
        else:
            n_undecided += 1
            undecided_recs.append({"idx": i, "occupation": meta[i]["occupation"], "caption": caps[j],
                                   "n_male": nm, "n_female": nf,
                                   "reason": "no_gender_word" if (nm == 0 and nf == 0) else "tie"})

    n_noface = (args.n - n_face) if mode == "crop" else 0
    n_captioned = len(idxs)
    # failure = could not determine a gender for that image
    n_fail = n_undecided + n_noface
    results[key] = {
        "config": e,
        "n_total": args.n,
        "n_captioned": n_captioned,
        "n_noface": n_noface,
        "n_male": n_male, "n_female": n_female, "n_undecided": n_undecided,
        "n_fail": n_fail,
        "fail_rate": round(n_fail / args.n, 4),
        "undecided_rate": round(n_undecided / max(1, n_captioned), 4),
        "undecided_captions": undecided_recs,
        "all": per,
    }
    with open(os.path.join(args.out, f"{key}.json"), "w") as f:
        json.dump(results[key], f, ensure_ascii=False, indent=2)

# ----------------------------- 5) report ---------------------------------------
with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("all", "undecided_captions")}
               for k, v in results.items()}, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 78)
print(f"RESULTS  (N={args.n} images, insightface detected {n_face})")
print("=" * 78)
hdr = f"{'config':24s} {'input':5s} {'prompt':28s} {'M':>3s} {'F':>3s} {'undec':>5s} {'noFace':>6s} {'FAIL%':>6s}"
print(hdr); print("-" * len(hdr))
for k, v in results.items():
    c = v["config"]
    print(f"{k:24s} {c['input']:5s} {c['prompt']:28s} "
          f"{v['n_male']:>3d} {v['n_female']:>3d} {v['n_undecided']:>5d} {v['n_noface']:>6d} "
          f"{100*v['fail_rate']:>5.1f}%")
best = min(results.items(), key=lambda kv: kv[1]["fail_rate"])
print("-" * len(hdr))
print(f"BEST (lowest fail rate): {best[0]}  ->  {100*best[1]['fail_rate']:.1f}% failure")
print(f"\nSample undecided captions per config:")
for k, v in results.items():
    print(f"\n  [{k}]  ({v['n_undecided']} undecided, {v['n_noface']} no-face)")
    for r in v["undecided_captions"][:6]:
        print(f"     - ({r['reason']:13s} m{r['n_male']}/f{r['n_female']}) {r['occupation']:28s} | {r['caption']!r}")
print(f"\n[done] artifacts in: {os.path.abspath(args.out)}")
