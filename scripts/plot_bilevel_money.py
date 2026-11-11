#!/usr/bin/env python3
"""Bilevel money plot: seeded_radius_cog optimum vs MDI overlay intensity.
Usage: python plot_bilevel_money.py money/money_results_0x_1x_3x.json"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
mdir = os.path.dirname(path) or "."
J = json.load(open(path))
cfg = J["il_cfg"]
pnames  = [p["name"] for p in cfg["algorithm"]["optimize_params"]]
plabels = [p["label"] for p in cfg["algorithm"]["optimize_params"]]

def find_results(obj, out):
    if isinstance(obj, dict):
        if "geom_values" in obj and "best_params" in obj: out.append(obj)
        else:
            for v in obj.values(): find_results(v, out)
    elif isinstance(obj, list):
        for v in obj: find_results(v, out)

res = []; find_results(J["results"], res)
seen = {}
for r in res: seen[r["geom_values"].get("mdi_x", 0)] = r
res = [seen[k] for k in sorted(seen)]
assert res, "no results found in JSON"
X  = [r["geom_values"].get("mdi_x", 0) for r in res]
Rb = [r["best_params"][pnames[0]] for r in res]
Eb = [r["best_params"][pnames[1]] for r in res]
S  = [r["best_score"] for r in res]

L = Rgrid = Egrid = None
tpath = os.path.join(mdir, "loss_landscape_tensor.npz")
if os.path.exists(tpath):
    d = np.load(tpath)
    L, Rgrid, Egrid = d["score"], d["grid_R_cluster_mm"], d["grid_E_threshold_GeV"]
    print(f"landscapes: {tpath} shape={L.shape}")

shades = ["#8ed3dc", "#3ba7b8", "#0b5f6b", "#083f47"][:max(3, len(X))]
fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.0))
fig.suptitle("Bilevel coupling: reconstruction optimum vs MDI background intensity",
             fontsize=12.5)

def dotpanel(ax, Y, ylab, title):
    ax.plot(X, Y, "-", color="#0b5f6b", lw=1.6, zorder=1)
    for x, y, c in zip(X, Y, shades): ax.plot([x], [y], "o", ms=8, color=c, zorder=2)
    ax.set_xlabel("MDI overlay intensity [× nominal]"); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5)

dotpanel(axs[0][0], Rb, plabels[0], "Optimal cluster radius (degenerate — see text)")
dotpanel(axs[0][1], Eb, plabels[1], "Optimal energy threshold")
dotpanel(axs[1][0], S, "best score (snr_energy)", "Achievable score at optimum")

ax = axs[1][1]
if L is not None and len(L) == len(X):
    for i, (x, c) in enumerate(zip(X, shades)):
        land = np.asarray(L[i], float)                    # (R, Ethr)
        k = int(np.nanargmax(np.nanmax(land, axis=1)))    # best R row (plateau)
        sl = land[k, :]                                   # score vs E_thr
        ax.plot(Egrid, sl, color=c, lw=1.8, marker=".", ms=5,
                label=f"{x:g}×")
        j = int(np.nanargmax(sl))
        ax.plot([Egrid[j]], [sl[j]], "o", ms=8, color=c)
        # 99.5%-of-max bands, both axes, printed for the caption
        nearE = sl >= 0.995 * sl[j]
        ax.plot(Egrid[nearE], np.full(int(nearE.sum()), sl[j]), lw=5, color=c,
                alpha=.22, solid_capstyle="butt")
        rrow = land[:, j]; nearR = rrow >= 0.995 * np.nanmax(rrow)
        print(f"  {x:g}x: grid max {sl[j]:.4f} | E_thr plateau "
              f"[{Egrid[nearE].min():.2f}, {Egrid[nearE].max():.2f}] GeV | "
              f"R plateau [{Rgrid[nearR].min():.0f}, {Rgrid[nearR].max():.0f}] mm")
    ax.set_xlabel(plabels[1]); ax.set_ylabel("score")
    ax.set_title("Score vs threshold at plateau R (per intensity)", fontsize=10.5)
    ax.legend(fontsize=9, frameon=False, title="MDI intensity")
else:
    ax.axis("off"); ax.text(.5, .5, "loss_landscape_tensor.npz not found",
                            ha="center", va="center", color="0.5")

for a in axs.flat:
    if a.axison: a.grid(alpha=0.25, lw=0.6)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(f"bilevel_money.{ext}", dpi=200)
print("wrote bilevel_money.png/.pdf")
for r in res:
    print(f"mdi_x={r['geom_values'].get('mdi_x'):>4}: " +
          ", ".join(f"{k}={v:.3g}" for k, v in r["best_params"].items()) +
          f", score={r['best_score']:.4g}")
