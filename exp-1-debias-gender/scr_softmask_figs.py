"""Figures for the SCR soft-gate toy experiment. Run scr_softmask_experiment.py first."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

import scr_softmask_lib as L

OUT = "./scr_softmask_out"
DEPTH = 0.8
CM = "RdYlBu"   # multiplier m: red = released (0.2), blue = preserved (1.0)


def load():
    R = json.load(open(os.path.join(OUT, "results.json")))
    D = torch.load(os.path.join(OUT, "masks.pt"), map_location="cpu", weights_only=False)
    return R, D


# --------------------------------------------------------------------------------------------------
def fig_grid(R, D, n_show=6):
    """THE headline figure: what each gate actually does to the gradient, sample by sample."""
    a, imgs, fm, det = D["attn"], D["images"].float(), D["face_mask"], D["det"]
    idx = [i for i in range(a.shape[0]) if det[i]]
    # show the samples that most expose the baseline's instability: the extremes of its damped area
    area = L.s_hard(a, 0.15).flatten(1).mean(1)
    order = sorted(idx, key=lambda i: float(area[i]))
    pick = [order[0], order[1], order[len(order) // 3], order[2 * len(order) // 3],
            order[-2], order[-1]][:n_show]

    gates = [
        ("CURRENT\nhard  minmax>=0.15", L.s_hard(a, 0.15)),
        ("METHOD 1 (as asked)\nsoftmax  tau=0.25", L.s_softmax(a, tau=0.25)),
        ("METHOD 2 (as asked)\ntop-p 50%  soft", L.s_topp(a, p=0.5, gamma=1.0)),
        ("RECOMMENDED\nmass-sigmoid p=.65 t=.12", L.s_mass_sigmoid(a, 0.65, 0.12)),
    ]
    ncol = 2 + len(gates)
    fig, ax = plt.subplots(len(pick), ncol, figsize=(2.05 * ncol, 2.15 * len(pick)))
    g, _ = L.minmax_norm(a)

    for r, i in enumerate(pick):
        im = ((imgs[i].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        ax[r, 0].imshow(im)
        x1, y1, x2, y2 = (fm[i].nonzero().float().min(0).values.tolist()[::-1] +
                          fm[i].nonzero().float().max(0).values.tolist()[::-1])
        ax[r, 0].add_patch(Rectangle((x1 * 8, y1 * 8), (x2 - x1) * 8, (y2 - y1) * 8,
                                     ec="lime", fc="none", lw=1.6))
        ax[r, 0].set_ylabel(f"#{i}", fontsize=9)
        ax[r, 1].imshow(g[i], cmap="magma")

        for c, (_, s) in enumerate(gates):
            m = (1 - DEPTH * s[i]).clamp(0, 1)
            h = ax[r, 2 + c].imshow(m, cmap=CM, vmin=0.2, vmax=1.0)
            lam = float((1 - m).mean())
            dmp = float(s[i].mean())          # effective damped area = mean(s) (honest for soft masks)
            ax[r, 2 + c].set_title(f"$\\lambda$={lam:.2f}  area$_{{eff}}$={dmp:.0%}", fontsize=8, pad=2)
        for c in range(ncol):
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])

    for c, t in enumerate(["generated\n(face box)", "gender attn\n(min-max)"] + [t for t, _ in gates]):
        ax[0, c].text(.5, 1.42, t, transform=ax[0, c].transAxes, ha="center", va="bottom",
                      fontsize=9.5, fontweight="bold")
    cb = fig.colorbar(h, ax=ax.ravel().tolist(), fraction=.013, pad=.012)
    cb.set_label("SCR gradient multiplier  m   (0.2 = released $\\to$ person may change,  "
                 "1.0 = preserved)", fontsize=9)
    fig.suptitle("SCR gradient gate, sample by sample.  Rows sorted by how much the CURRENT gate damps: "
                 "it swings 25%$\\to$90% of the image.\nThe mass-based gate spends the same budget every time.",
                 fontsize=11, y=.99)
    fig.savefig(f"{OUT}/fig1_mask_grid.png", dpi=125, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def fig_consistency(R, D):
    """Per-sample budget: the user's actual complaint, quantified."""
    a = D["attn"]
    gates = [
        ("CURRENT\nhard minmax>=0.15", L.s_hard(a, .15), "#c0392b"),
        ("hard top-p\np=0.75", (L.s_topp(a, .75) > 0).float(), "#e67e22"),
        ("soft nucleus\n.30/.55", L.s_soft_nucleus(a, .30, .55), "#16a085"),
        ("RECOMMENDED\nmass-sigmoid\np=.65 t=.12", L.s_mass_sigmoid(a, .65, .12), "#2471a3"),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for k, (nm, s, col) in enumerate(gates):
        lam = (DEPTH * s).flatten(1).mean(1).numpy()
        x = k + np.random.RandomState(0).uniform(-.13, .13, len(lam))
        ax[0].scatter(x, lam, s=13, alpha=.55, color=col, zorder=3)
        ax[0].plot([k - .3, k + .3], [lam.mean()] * 2, color="k", lw=2, zorder=4)
        cv = lam.std() / lam.mean()
        ax[0].text(k, 0.80, f"CV\n{cv:.3f}", ha="center", fontsize=10, fontweight="bold",
                   color=col)
        ax[0].vlines(k, lam.min(), lam.max(), color=col, lw=1, alpha=.5)
    ax[0].set_xticks(range(len(gates)))
    ax[0].set_xticklabels([g[0] for g in gates], fontsize=8.5)
    ax[0].set_ylabel("$\\lambda$ = released SCR gradient energy   (per sample)")
    ax[0].set_title("Each dot is one image. The current gate's damping\n"
                    "swings 5x across samples; the mass-based gates do not.", fontsize=10)
    ax[0].set_ylim(0, .9)
    ax[0].grid(alpha=.25, axis="y")

    # what the current gate REALLY is: a top-p gate whose p is out of control
    prob = L.to_prob(a)
    mass = (prob * L.s_hard(a, .15)).flatten(1).sum(1).numpy()
    ax[1].hist(mass, bins=18, color="#c0392b", alpha=.8, edgecolor="w")
    ax[1].axvline(mass.mean(), color="k", ls="--", lw=2,
                  label=f"mean = {mass.mean():.2f} of the attention mass")
    ax[1].axvline(.65, color="#2471a3", lw=2.5, label="proposed fixed cut  p = 0.65")
    ax[1].set_xlabel("fraction of the gender-attention MASS that the current gate releases")
    ax[1].set_ylabel("# images")
    ax[1].set_title(f"The current gate is already a top-p gate --\n"
                    f"but its p is uncontrolled: {mass.min():.2f} to {mass.max():.2f}", fontsize=10)
    ax[1].legend(fontsize=8.5)
    ax[1].grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_budget_consistency.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def fig_design_space(R, D):
    """Where the quality actually comes from: depth, not softness."""
    fom = R["fixed_budget_contrast"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # (a) preservation contrast at fixed global SCR strength, by gate x factor2
    gates = ["hard (value thr)", "hard_topp (mass)", "soft_nucleus", "mass_sigmoid t=.12"]
    f2s = [0.2, 0.1, 0.0]
    w = .26
    cols = ["#c0392b", "#e67e22", "#16a085"]
    for j, f2 in enumerate(f2s):
        vals = [fom[f"{gt} | factor2={f2}"]["P"] for gt in gates]
        ax[0].bar(np.arange(len(gates)) + (j - 1) * w, vals, w, label=f"factor2={f2}", color=cols[j])
        for i, v in enumerate(vals):
            ax[0].text(i + (j - 1) * w, v + .08, f"{v:.1f}", ha="center", fontsize=8)
    ax[0].axhline(fom["hard (value thr) | factor2=0.2"]["P"], color="k", ls="--", lw=1.2,
                  label="current gate")
    ax[0].set_xticks(range(len(gates)))
    ax[0].set_xticklabels(["hard\n(value)", "hard top-p\n(mass)", "soft\nnucleus", "mass\nsigmoid"], fontsize=9)
    ax[0].set_ylabel("preservation contrast  P = $m_{bg}$ / $m_{face}$")
    ax[0].set_title("At a FIXED global SCR strength (<m>=0.463,\n"
                    "so weight_loss_scr never changes): depth is the lever", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.25, axis="y")

    # (b) the softness trade-off: localization vs stability, as tau sweeps
    tau_rows = R["sweeps"]["mass_sigmoid_tau"]
    st = R["stability"]
    taus = [r["param"] for r in tau_rows]
    sel = [r["selectivity"] for r in tau_rows]
    ax[1].plot(taus, sel, "o-", color="#2471a3", lw=2)
    for t, s in zip(taus, sel):
        if t in (0.02, 0.12, 0.5):
            ax[1].annotate(f"$\\tau$={t}", (t, s), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax[1].axhline(R["baseline"]["selectivity"]["mean"], color="#c0392b", ls="--",
                  label=f"current gate ({R['baseline']['selectivity']['mean']:.2f})")
    ax[1].set_xlabel("temperature $\\tau$  (mass domain)")
    ax[1].set_ylabel("release selectivity  $R_{face}$ / $R_{bg}$")
    ax[1].set_title("Softening costs localization, monotonically.\n"
                    "$\\tau \\to 0$ = hard nucleus.", fontsize=10)
    ax[1].legend(fontsize=8.5)
    ax[1].grid(alpha=.25)

    # (c) stability under an attention-source change (real perturbation)
    if st:
        names = ["hard (thr)", "hard_topp (mass)", "soft_nucleus", "mass_sigmoid t=.12", "mass_sigmoid t=.25"]
        names = [n for n in names if n in st]
        vals = [st[n]["rel_delta"] for n in names]
        cc = ["#c0392b"] + ["#e67e22"] * (len(names) - 3) + ["#2471a3", "#1a5276"]
        ax[2].barh(range(len(names)), vals, color=cc[:len(names)])
        for i, v in enumerate(vals):
            ax[2].text(v + .01, i, f"{v:.3f}", va="center", fontsize=9)
        ax[2].set_yticks(range(len(names)))
        ax[2].set_yticklabels(names, fontsize=9)
        ax[2].invert_yaxis()
        ax[2].set_xlabel("relative gate drift  mean|$\\Delta m$| / $\\lambda$   (lower = more stable)")
        ax[2].set_title("Swap the attention source (ALL / MID / RES16 blocks).\n"
                        "How much does the gate move? All at $\\lambda$=0.32.", fontsize=10)
        ax[2].grid(alpha=.25, axis="x")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_design_space.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def fig_caveat(R, D):
    """The honest control: how much does the attention map really know about where the person is?"""
    a, fm, det, imgs = D["attn"], D["face_mask"], D["det"], D["images"].float()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
    m = R["map"]
    names = ["gender\nattention", "center prior\n(ignores attn)", "shuffled attn\n(no structure)"]
    vals = [m["auc_attn_vs_face"], m["auc_center_prior_vs_face"], m["auc_shuffled_attn"]]
    b = ax[0].bar(names, vals, color=["#2471a3", "#e67e22", "#95a5a6"])
    ax[0].bar_label(b, fmt="%.3f", fontsize=10)
    ax[0].axhline(.5, color="k", ls=":", lw=1)
    ax[0].set_ylim(.4, 1.0)
    ax[0].set_ylabel("AUC:  does this pixel lie in the face?")
    ax[0].set_title("CAVEAT. On this prompt the attention map is barely\n"
                    "better than 'the face is in the middle'.", fontsize=10)
    ax[0].grid(alpha=.25, axis="y")

    mean_attn = L.minmax_norm(a)[0][det].mean(0)
    im = ax[1].imshow(mean_attn, cmap="magma")
    ax[1].contour(fm[det].mean(0), levels=[.5], colors="lime", linewidths=2)
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[1].set_title("mean gender-attention over the 34 images\n"
                    "(green = mean face box). It is centered and diffuse.", fontsize=10)
    fig.colorbar(im, ax=ax[1], fraction=.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_caveat_control.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    R, D = load()
    fig_grid(R, D)
    fig_consistency(R, D)
    fig_design_space(R, D)
    fig_caveat(R, D)
    print("figures ->", ", ".join(sorted(f for f in os.listdir(OUT) if f.endswith(".png"))))
