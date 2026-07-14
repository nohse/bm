#!/usr/bin/env python
"""Build the TRUE-label evaluation manifest (3-detector consensus):
  group 'true'  : 20 genuine FACE  (insightface+HOG+CNN all detect a face)
                  20 genuine NOFACE (insightface+HOG+CNN all detect 0 faces, person context)
  group 'missed': 20 insightface-MISSED faces (insightface=0 but HOG+CNN find a face; truth=face)
                  -> diagnostic: does the error-method recover what insightface missed?
"""
import os, json, glob
import numpy as np
from PIL import Image
import face_recognition

HERE = os.path.dirname(os.path.abspath(__file__))
N = 20


def cnn_faces(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    return len(face_recognition.face_locations(rgb, model="cnn"))


def hog_faces(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    return len(face_recognition.face_locations(rgb, model="hog"))


# ---- genuine NO-FACE (consensus 0) from the new noface_candidates ----
nf = []
for f in sorted(glob.glob(os.path.join(HERE, "noface_candidates", "shard_*.json"))):
    nf += json.load(open(f))
print(f"noface candidates (insight=0 & hog=0): {len(nf)}")
genuine_nf, seen_occ = [], set()
# diversity: one per occupation first, verify CNN==0
for r in sorted(nf, key=lambda r: (r["occ_idx"], r["template"])):
    if len(genuine_nf) >= N:
        break
    if r["occ_idx"] in seen_occ:
        continue
    if cnn_faces(os.path.join(HERE, r["file"])) == 0:
        genuine_nf.append(r); seen_occ.add(r["occ_idx"])
# fill if short (allow repeat occupations)
for r in nf:
    if len(genuine_nf) >= N:
        break
    if r in genuine_nf:
        continue
    if cnn_faces(os.path.join(HERE, r["file"])) == 0:
        genuine_nf.append(r)
print(f"genuine NO-FACE selected (3-detector consensus 0): {len(genuine_nf)}")

# ---- genuine FACE (insightface>=1 and HOG>=1) from occ_candidates ----
occ = []
for f in sorted(glob.glob(os.path.join(HERE, "occ_candidates", "shard_*.json"))):
    occ += json.load(open(f))
faces = [r for r in occ if r["label"] == "face"]
genuine_face, seen_occ_f = [], set()
for r in sorted(faces, key=lambda r: -r["det_score"]):
    if len(genuine_face) >= N:
        break
    if r["prompt_idx"] in seen_occ_f:
        continue
    if hog_faces(os.path.join(HERE, r["file"])) >= 1:
        genuine_face.append(r); seen_occ_f.add(r["prompt_idx"])
print(f"genuine FACE selected: {len(genuine_face)}")

# ---- insightface-MISSED faces (insight=0 but HOG>=1) from occ_candidates ----
missed_pool = [r for r in occ if r["label"] == "noface"]
missed, seen_occ_m = [], set()
for r in missed_pool:
    if len(missed) >= N:
        break
    p = os.path.join(HERE, r["file"])
    if hog_faces(p) >= 1 and cnn_faces(p) >= 1:   # genuinely has a face insightface missed
        missed.append(r)
print(f"insightface-MISSED faces selected: {len(missed)}")

manifest = []
for r in genuine_face:
    manifest.append(dict(file=r["file"], group="true", true_label="face",
                         insight_label="face", occupation=r.get("prompt", "")))
for r in genuine_nf:
    manifest.append(dict(file=r["file"], group="true", true_label="noface",
                         insight_label="noface", occupation=r.get("occupation", "")))
for r in missed:
    manifest.append(dict(file=r["file"], group="missed", true_label="face",
                         insight_label="noface", occupation=r.get("prompt", "")))
json.dump(manifest, open(os.path.join(HERE, "manifest_true.json"), "w"), indent=2)


def montage(recs, path, cols=10, thumb=128):
    rows = max(1, (len(recs) + cols - 1) // cols)
    cv = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, r in enumerate(recs):
        im = Image.open(os.path.join(HERE, r["file"])).convert("RGB").resize((thumb, thumb))
        cv.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    cv.save(path)


montage(genuine_face, os.path.join(HERE, "montage_true_face.png"))
montage(genuine_nf, os.path.join(HERE, "montage_true_noface.png"))
montage(missed, os.path.join(HERE, "montage_missed.png"))
print(f"\nmanifest_true.json: {len(genuine_face)} face + {len(genuine_nf)} noface (group=true) "
      f"+ {len(missed)} missed")
print("saved montage_true_face.png, montage_true_noface.png, montage_missed.png")
