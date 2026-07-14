#!/usr/bin/env python
"""Aggregate 8 generation shards; select 20 face-detected + 20 no-face-detected images.

Controlled design: prefer occupations that produced BOTH a face and a no-face image,
and take one of each from the SAME occupation. This matches the two classes on
occupation semantics, so the classifier's only systematic cue is face presence
(the hard setting). Remaining slots are filled from the leftover pool."""
import os, json, glob, shutil
from collections import defaultdict
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(HERE, "occ_candidates")
N_PER = 20

recs = []
for f in sorted(glob.glob(os.path.join(CAND, "shard_*.json"))):
    recs += json.load(open(f))
print(f"total candidates: {len(recs)}")
face = [r for r in recs if r["label"] == "face"]
noface = [r for r in recs if r["label"] == "noface"]
print(f"  face-detected: {len(face)}   no-face: {len(noface)}")

by_occ_face = defaultdict(list)
by_occ_non = defaultdict(list)
for r in face:
    by_occ_face[r["prompt_idx"]].append(r)
for r in noface:
    by_occ_non[r["prompt_idx"]].append(r)

# occupations that have BOTH -> matched pairs first
both = sorted(set(by_occ_face) & set(by_occ_non))
sel_face, sel_non, used_occ = [], [], set()
for occ in both:
    if len(sel_face) >= N_PER or len(sel_non) >= N_PER:
        break
    f = max(by_occ_face[occ], key=lambda r: r["det_score"])  # clearest face
    n = by_occ_non[occ][0]
    sel_face.append(f); sel_non.append(n); used_occ.add(occ)
print(f"matched-occupation pairs selected: {len(sel_face)}")

# fill remaining face slots (diverse occupations, clearest faces)
def fill(target_list, by_occ, pool, n_need):
    # first pass: one per NEW occupation
    for occ in sorted(by_occ, key=lambda o: -max(r["det_score"] for r in by_occ[o])):
        if len(target_list) >= n_need:
            break
        if occ in used_occ:
            continue
        target_list.append(max(by_occ[occ], key=lambda r: r["det_score"]))
        used_occ.add(occ)
    # second pass: allow repeats of occupation if still short
    i = 0
    flat = sorted(pool, key=lambda r: -r["det_score"])
    while len(target_list) < n_need and i < len(flat):
        if flat[i] not in target_list:
            target_list.append(flat[i])
        i += 1

used_occ_face = set(r["prompt_idx"] for r in sel_face)
used_occ_non = set(r["prompt_idx"] for r in sel_non)
# fill faces
for occ in sorted(by_occ_face, key=lambda o: -max(r["det_score"] for r in by_occ_face[o])):
    if len(sel_face) >= N_PER:
        break
    if occ in used_occ_face:
        continue
    sel_face.append(max(by_occ_face[occ], key=lambda r: r["det_score"])); used_occ_face.add(occ)
flat_face = sorted(face, key=lambda r: -r["det_score"])
fi = 0
while len(sel_face) < N_PER and fi < len(flat_face):
    if flat_face[fi] not in sel_face:
        sel_face.append(flat_face[fi])
    fi += 1
# fill nofaces
for occ in sorted(by_occ_non):
    if len(sel_non) >= N_PER:
        break
    if occ in used_occ_non:
        continue
    sel_non.append(by_occ_non[occ][0]); used_occ_non.add(occ)
ni = 0
while len(sel_non) < N_PER and ni < len(noface):
    if noface[ni] not in sel_non:
        sel_non.append(noface[ni])
    ni += 1

sel_face = sel_face[:N_PER]
sel_non = sel_non[:N_PER]
assert len(sel_face) == N_PER and len(sel_non) == N_PER, (len(sel_face), len(sel_non))

manifest = []
for r in sel_face:
    manifest.append({**r, "label": "face"})
for r in sel_non:
    manifest.append({**r, "label": "noface"})
json.dump(manifest, open(os.path.join(HERE, "manifest_occ.json"), "w"), indent=2)

# montage for eyeballing
def montage(recs, path, cols=10, thumb=128):
    rows = (len(recs) + cols - 1) // cols
    cv = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, r in enumerate(recs):
        im = Image.open(os.path.join(HERE, r["file"])).convert("RGB").resize((thumb, thumb))
        cv.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    cv.save(path)
montage(sel_face, os.path.join(HERE, "montage_occ_face.png"))
montage(sel_non, os.path.join(HERE, "montage_occ_noface.png"))

n_both = len(set(r["prompt_idx"] for r in sel_face) & set(r["prompt_idx"] for r in sel_non))
print(f"SELECTED: {len(sel_face)} face + {len(sel_non)} noface")
print(f"occupations appearing in BOTH classes: {n_both}")
print(f"face det_score range: {min(r['det_score'] for r in sel_face):.3f}-{max(r['det_score'] for r in sel_face):.3f}")
print("saved manifest_occ.json, montage_occ_face.png, montage_occ_noface.png")
