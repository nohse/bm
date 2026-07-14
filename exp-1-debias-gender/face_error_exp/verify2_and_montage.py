#!/usr/bin/env python
"""Independent cross-verification with a SECOND face detector (face_recognition / dlib-HOG)
and a visual montage so the 50/50 split can be eyeballed."""
import os, json, glob
import numpy as np
from PIL import Image, ImageDraw
import face_recognition

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "manifest.json")) as f:
    manifest = json.load(f)

# ---- second-detector cross check ----
agree_face, agree_non, disagree = 0, 0, []
for m in manifest:
    img = face_recognition.load_image_file(os.path.join(HERE, m["file"]))
    locs = face_recognition.face_locations(img, model="hog")
    fr_has = len(locs) > 0
    insight_has = m["n_faces"] >= 1
    if m["label"] == "face":
        if fr_has: agree_face += 1
        else: disagree.append((m["file"], "insight=face, fr=noface"))
    else:
        if not fr_has: agree_non += 1
        else: disagree.append((m["file"], f"insight=noface, fr={len(locs)}face"))

n_face = sum(1 for m in manifest if m["label"] == "face")
n_non = sum(1 for m in manifest if m["label"] == "nonface")
lines = [
    "SECOND-DETECTOR CROSS-VERIFICATION (face_recognition / dlib-HOG)",
    f"FACE set:     {agree_face}/{n_face} also have a face per 2nd detector",
    f"NON-FACE set: {agree_non}/{n_non} also have NO face per 2nd detector",
    f"disagreements: {len(disagree)}",
]
for f, why in disagree:
    lines.append(f"   {f}: {why}")
summary = "\n".join(lines)
print(summary)
with open(os.path.join(HERE, "verify2_summary.txt"), "w") as f:
    f.write(summary + "\n")

# ---- montage ----
def montage(files, cols=10, thumb=128):
    rows = (len(files) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, fp in enumerate(files):
        im = Image.open(fp).convert("RGB").resize((thumb, thumb))
        canvas.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    return canvas

face_files = sorted(glob.glob(os.path.join(HERE, "faces", "*.png")))
non_files = sorted(glob.glob(os.path.join(HERE, "nonfaces", "*.png")))
montage(face_files).save(os.path.join(HERE, "montage_faces.png"))
montage(non_files).save(os.path.join(HERE, "montage_nonfaces.png"))
print("saved montage_faces.png, montage_nonfaces.png")
