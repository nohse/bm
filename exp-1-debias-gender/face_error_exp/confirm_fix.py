#!/usr/bin/env python
"""Confirm the truncated-z0 fix on a larger set: 24 face + 12 noface occupations,
2 scoring noise seeds, capture-path processor, K=8. Candidates:
  (portrait vs faceless, t500-950), (portrait vs faceless, t600-950),
  (face vs faceless, t600-950)  <- keeps the user's original prompt pair."""
import torch, numpy as np
import importlib.util, sys
spec = importlib.util.spec_from_file_location("v2", "verify_truncated_z0_v2.py")
# reuse v2's models/functions without rerunning its __main__ sweep: import pieces manually
exec(open("verify_truncated_z0_v2.py").read().split('print("=== generating')[0])

OCC_F = ["doctor","butcher","senator","cosmetologist","geologist","lifeguard","narrator",
         "researcher","custodian","administrator","economist","bartender","cardiologist",
         "blaster","citizen","inventor","promoter","salesperson","sergeant","stocker",
         "violinist","janitor","trucker","maid"]
OCC_N = ["doctor","butcher","senator","lifeguard","custodian","economist","cardiologist",
         "janitor","trucker","maid","inventor","stocker"]
FACE_GEN = [f"A photo of the face of a {o}, a person" for o in OCC_F]
NOFACE_GEN = [f"A photo of a {o} seen from behind, back of the head, no face visible" for o in OCC_N]

print("generating truncated z0 (train-style) ...", flush=True)
ZF = torch.cat([gen_z0_train_style(p, 1000+i, True) for i,p in enumerate(FACE_GEN)])
ZN = torch.cat([gen_z0_train_style(p, 8000+i, True) for i,p in enumerate(NOFACE_GEN)])

set_cross_attn_processor(AttnProcessor)
for fp, np_, tlo, thi in [
    ("a portrait photo of a person","a faceless photo",500,950),
    ("a portrait photo of a person","a faceless photo",600,950),
    ("a photo of a face","a faceless photo",600,950),
    ("a photo of a face","a faceless photo",50,950),   # current default (broken) for reference
]:
    accs_f, accs_n = [], []
    for sd in (777, 1234):
        Ef_f=E_of(ZF,fp,tlo,thi,8,seed=sd); En_f=E_of(ZF,np_,tlo,thi,8,seed=sd)
        Ef_n=E_of(ZN,fp,tlo,thi,8,seed=sd); En_n=E_of(ZN,np_,tlo,thi,8,seed=sd)
        accs_f.append(int((Ef_f<En_f).sum())); accs_n.append(int((Ef_n>=En_n).sum()))
    print(f"[{fp[:22]:<22} | {np_[:16]:<16} | t{tlo}-{thi}] "
          f"face {accs_f}/{len(ZF)}  noface {accs_n}/{len(ZN)}", flush=True)
print("done")
