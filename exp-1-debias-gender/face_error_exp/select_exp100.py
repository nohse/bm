#!/usr/bin/env python
"""Build a fresh 50 face + 50 no-face set (100 images) in a SEPARATE folder exp100/.
Faces: insightface-detected (from occ_candidates) AND dlib-HOG confirmed (CPU).
No-face: insightface=0 AND dlib-HOG=0 (recorded), spread across the 5 no-face
templates + occupations for diversity. Pure CPU, no GPU memory used."""
import os, json, glob, shutil
from collections import defaultdict
import numpy as np
from PIL import Image
import face_recognition

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp100")
N = 50


def hog(path):
    return len(face_recognition.face_locations(np.array(Image.open(path).convert("RGB")), model="hog"))


def reset(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)


# ---- FACE pool (occ_candidates, label==face) ----
occ = []
for f in sorted(glob.glob(os.path.join(HERE, "occ_candidates", "shard_*.json"))):
    occ += json.load(open(f))
faces = [r for r in occ if r["label"] == "face"]
face_sel, seen = [], set()
for r in sorted(faces, key=lambda r: -r["det_score"]):        # clearest first, diverse occupations
    if len(face_sel) >= N:
        break
    if r["prompt_idx"] in seen:
        continue
    if hog(os.path.join(HERE, r["file"])) >= 1:                # 2-detector consensus
        face_sel.append(r); seen.add(r["prompt_idx"])
# fill if short (allow repeated occupations)
for r in sorted(faces, key=lambda r: -r["det_score"]):
    if len(face_sel) >= N:
        break
    if r in face_sel:
        continue
    if hog(os.path.join(HERE, r["file"])) >= 1:
        face_sel.append(r)
print(f"FACE selected: {len(face_sel)}")

# ---- NO-FACE pool (noface_candidates: insight=0 & hog=0), spread across templates ----
nf = []
for f in sorted(glob.glob(os.path.join(HERE, "noface_candidates", "shard_*.json"))):
    nf += json.load(open(f))
by_tmpl = defaultdict(list)
for r in nf:
    by_tmpl[r["template"]].append(r)
# round-robin across templates, one per occupation within a template
nf_sel, seen_occ = [], set()
tmpls = sorted(by_tmpl)
ptr = {t: 0 for t in tmpls}
while len(nf_sel) < N:
    progressed = False
    for t in tmpls:
        lst = sorted(by_tmpl[t], key=lambda r: (r["occ_idx"], r["seed"]))
        while ptr[t] < len(lst):
            r = lst[ptr[t]]; ptr[t] += 1
            if (t, r["occ_idx"]) in seen_occ:
                continue
            nf_sel.append(r); seen_occ.add((t, r["occ_idx"])); progressed = True
            break
        if len(nf_sel) >= N:
            break
    if not progressed:
        break
# fill any remainder
for r in nf:
    if len(nf_sel) >= N:
        break
    if r not in nf_sel:
        nf_sel.append(r)
nf_sel = nf_sel[:N]
print(f"NO-FACE selected: {len(nf_sel)} (templates used: {sorted(set(r['template'] for r in nf_sel))})")

# ---- write folder + manifest + montages ----
reset(os.path.join(OUT, "face"))
reset(os.path.join(OUT, "noface"))
manifest = []
for i, r in enumerate(face_sel):
    dst = os.path.join("exp100", "face", f"face_{i:02d}.png")
    shutil.copy(os.path.join(HERE, r["file"]), os.path.join(HERE, dst))
    manifest.append(dict(file=dst, label="face", src=r["file"], occupation=r.get("prompt", "")))
for i, r in enumerate(nf_sel):
    dst = os.path.join("exp100", "noface", f"noface_{i:02d}.png")
    shutil.copy(os.path.join(HERE, r["file"]), os.path.join(HERE, dst))
    manifest.append(dict(file=dst, label="noface", src=r["file"],
                         occupation=r.get("occupation", ""), template=r.get("template")))
json.dump(manifest, open(os.path.join(OUT, "manifest_exp100.json"), "w"), indent=2)


def montage(recs, path, cols=10, thumb=110):
    rows = (len(recs) + cols - 1) // cols
    cv = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, r in enumerate(recs):
        im = Image.open(os.path.join(HERE, r["file"])).convert("RGB").resize((thumb, thumb))
        cv.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    cv.save(path)
montage(manifest[:N], os.path.join(OUT, "montage_face.png"))
montage(manifest[N:], os.path.join(OUT, "montage_noface.png"))
print(f"wrote exp100/ : {N} face + {N} noface, manifest_exp100.json, montages")
