"""Extract face bounding boxes for the 50 saved toy images (CPU insightface).

Ground truth for the localization metrics in scr_softmask_experiment.py: the SCR gradient gate is
supposed to release the gender-identity region, so we score each candidate mask by how much of its
damping lands inside the detected face box vs the background.

Writes: <out>/face_boxes.pt  {"boxes": [N,4] in 64x64 latent coords (-1 if no face), "det": [N] bool}
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--maps",
        default="/workspace/finetune-fair-diffusion/exp-1-debias-gender/attmap_experiments/"
        "attmap_threshold_mask_out_50/maps/attmaps_and_images.pt",
    )
    ap.add_argument("--out", default="./scr_softmask_out")
    ap.add_argument("--det_size", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = torch.load(args.maps, map_location="cpu", weights_only=False)
    images = d["images"].float()  # [N,3,512,512] in [-1,1]
    n = images.shape[0]

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root="/workspace/.insightface",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection"],
    )
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    boxes = torch.full((n, 4), -1.0)
    det = torch.zeros(n, dtype=torch.bool)
    H = images.shape[-1]  # 512
    scale = 64.0 / H  # latent grid is 64x64

    for i in range(n):
        img = ((images[i].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        bgr = img[:, :, ::-1].copy()
        faces = app.get(bgr)
        if len(faces) == 0:
            print(f"[{i:02d}] no face")
            continue
        # largest face
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        boxes[i] = torch.tensor([x1 * scale, y1 * scale, x2 * scale, y2 * scale])
        det[i] = True
        area = (x2 - x1) * (y2 - y1) / (H * H)
        print(f"[{i:02d}] face bbox px=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) area_frac={area:.3f}")

    torch.save({"boxes": boxes, "det": det}, os.path.join(args.out, "face_boxes.pt"))
    print(f"\ndetected {int(det.sum())}/{n}; saved -> {os.path.join(args.out, 'face_boxes.pt')}")
    if det.any():
        b = boxes[det]
        af = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (64 * 64))
        print(f"face area fraction of the 64x64 latent: mean={af.mean():.3f} std={af.std():.3f} "
              f"min={af.min():.3f} max={af.max():.3f}")


if __name__ == "__main__":
    main()
