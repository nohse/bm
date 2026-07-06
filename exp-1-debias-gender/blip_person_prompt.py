#!/usr/bin/env python3
"""Reuse the SAME 50 images/crops from blip_exp_out/ and test a new prompt
"the gender of the person is" with BOTH face-crop and full-image inputs.
Only BLIP is loaded (no SD generation, no face detection re-run)."""
import os, json, re, warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image

OUT = "blip_exp_out"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

RE_M = re.compile(r"\b(?:male|males|man|men|boy|boys|gentleman|gentlemen|guy|guys|he|him|his|masculine)\b")
RE_F = re.compile(r"\b(?:female|females|woman|women|girl|girls|lady|ladies|she|her|hers|feminine)\b")
NEG = re.compile(r"\bnot a[n]? (man|woman|male|female)\b.*\bbut (?:a |an |the )?(man|woman|male|female)\b")
MALEW = {"man", "male"}

def parse_orig(cap):
    c = cap.lower(); nm = len(RE_M.findall(c)); nf = len(RE_F.findall(c))
    return (1 if nm > nf else (0 if nf > nm else -1)), nm, nf

def parse_neg(cap):                       # negation-aware: "not a X, but a Y" -> Y
    c = cap.lower(); m = NEG.search(c)
    if m:
        return 1 if m.group(2) in MALEW else 0
    nm = len(RE_M.findall(c)); nf = len(RE_F.findall(c))
    return 1 if nm > nf else (0 if nf > nm else -1)

# --- reconstruct idx -> occupation, and detected set, from saved artifacts ---
occ_by_idx = {r["idx"]: r["occupation"] for r in json.load(open(f"{OUT}/exp3_fullimg_photo.json"))["all"]}
N = len(occ_by_idx)
detected = sorted(int(f[:3]) for f in os.listdir(f"{OUT}/crops") if f.endswith(".png"))
img_file = {}
for f in os.listdir(f"{OUT}/images"):
    if f.endswith(".png"):
        img_file[int(f[:3])] = os.path.join(OUT, "images", f)
full_images = [Image.open(img_file[i]).convert("RGB") for i in range(N)]
crop_images = {i: Image.open(f"{OUT}/crops/{i:03d}.png").convert("RGB") for i in detected}
print(f"[load] N={N} full images, {len(detected)} crops (no-face idx={[i for i in range(N) if i not in detected]})")

# --- BLIP ---
from transformers import BlipProcessor, BlipForConditionalGeneration
bp = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
bm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=dtype).to(device)
bm.eval()

@torch.no_grad()
def caption(images, prompt_text):
    caps = []
    for k in range(0, len(images), 16):
        batch = images[k:k + 16]
        inp = bp(images=batch, text=[prompt_text] * len(batch), return_tensors="pt")
        inp = {kk: vv.to(device) for kk, vv in inp.items()}
        inp["pixel_values"] = inp["pixel_values"].to(dtype)
        out = bm.generate(**inp, max_new_tokens=20, num_beams=1)
        caps.extend(bp.batch_decode(out, skip_special_tokens=True))
    return caps

PROMPT = "the gender of the person is"
EXPS = [
    {"key": "exp4_facecrop_person", "mode": "crop"},
    {"key": "exp5_fullimg_person",  "mode": "full"},
]
results = {}
for e in EXPS:
    key, mode = e["key"], e["mode"]
    idxs = detected if mode == "crop" else list(range(N))
    imgs = [crop_images[i] for i in idxs] if mode == "crop" else full_images
    caps = caption(imgs, PROMPT)
    noface = (N - len(detected)) if mode == "crop" else 0
    und = und_neg = nm = nf = 0
    undrecs = []
    for j, i in enumerate(idxs):
        p, a, b = parse_orig(caps[j]); pn = parse_neg(caps[j])
        if p == 1: nm += 1
        elif p == 0: nf += 1
        else:
            und += 1
            undrec = {"idx": i, "occupation": occ_by_idx[i], "caption": caps[j],
                      "reason": "no_gender_word" if (a == 0 and b == 0) else "tie"}
            undrec["recovered_by_negation_fix"] = (pn != -1)
            undrecs.append(undrec)
        if pn == -1: und_neg += 1
    results[key] = {
        "config": {"key": key, "input": mode, "prompt": PROMPT},
        "n_total": N, "n_captioned": len(idxs), "n_noface": noface,
        "n_male": nm, "n_female": nf, "n_undecided": und,
        "fail_rate": round((und + noface) / N, 4),
        "fail_rate_negfix": round((und_neg + noface) / N, 4),
        "undecided_captions": undrecs,
        "all": [{"idx": i, "occupation": occ_by_idx[i], "caption": caps[j],
                 "pred": parse_orig(caps[j])[0]} for j, i in enumerate(idxs)],
    }
    json.dump(results[key], open(f"{OUT}/{key}.json", "w"), ensure_ascii=False, indent=2)

# --- combined report (incl. prior exp1-3 from summary.json) ---
prior = json.load(open(f"{OUT}/summary.json"))
rows = []
for k, v in prior.items():
    rows.append((k, v["config"]["input"], v["config"]["prompt"], v["n_male"], v["n_female"],
                 v["n_undecided"], v["n_noface"], v["fail_rate"], v.get("fail_rate", None)))
for k, v in results.items():
    rows.append((k, v["config"]["input"], v["config"]["prompt"], v["n_male"], v["n_female"],
                 v["n_undecided"], v["n_noface"], v["fail_rate"], v["fail_rate_negfix"]))

print("\n" + "=" * 92)
print(f"COMBINED  (same N={N} images, 49 detected)")
print("=" * 92)
hdr = f"{'config':24s} {'input':5s} {'prompt':30s} {'M':>3s} {'F':>3s} {'und':>4s} {'noF':>4s} {'FAIL%':>6s} {'+parser%':>8s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    negfix = f"{100*r[8]:.1f}" if r[8] is not None else " -"
    print(f"{r[0]:24s} {r[1]:5s} {r[2]:30s} {r[3]:>3d} {r[4]:>3d} {r[5]:>4d} {r[6]:>4d} {100*r[7]:>5.1f}% {negfix:>7s}%")
best = min(rows, key=lambda r: r[7])
print("-" * len(hdr))
print(f"BEST raw fail-rate: {best[0]} -> {100*best[7]:.1f}%")

print("\nperson-prompt undecided captions:")
for k, v in results.items():
    print(f"\n  [{k}] undecided={v['n_undecided']} noface={v['n_noface']}")
    for r in v["undecided_captions"]:
        rec = " (neg-fix recovers)" if r["recovered_by_negation_fix"] else ""
        print(f"     - ({r['reason']:13s}) {r['occupation'][:24]:24s} | {r['caption']!r}{rec}")
