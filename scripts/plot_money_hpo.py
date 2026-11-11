#!/usr/bin/env python3
"""HPO-style money plot: landscape heatmaps + DE trajectories per MDI
intensity, anytime performance, and the headline. Conventions:
opt_history_f and score grids hold the SCORE (higher better), -inf = failed."""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

mdir = sys.argv[1] if len(sys.argv) > 1 else "money"
files = sorted(glob.glob(os.path.join(mdir, "landscape_*.npz")))
assert files, f"no landscape_*.npz in {mdir}"
runs = []
for f in files:
    d = np.load(f, allow_pickle=True)
    runs.append(dict(x=float(d["geom_mdi_x"]), score=d["score"],
                     Rg=d["grid_R_cluster_mm"], Eg=d["grid_E_threshold_GeV"],
                     hx=d["opt_history_x"], hf=d["opt_history_f"],
                     bR=float(d["best_R_cluster_mm"]),
                     bE=float(d["best_E_threshold_GeV"]),
                     bS=float(d["best_score"])))
runs.sort(key=lambda r: r["x"])

shades = ["#8ed3dc", "#3ba7b8", "#0b5f6b"]
fin = np.concatenate([r["score"][np.isfinite(r["score"])] for r in runs])
vmin, vmax = fin.min(), fin.max()

fig = plt.figure(figsize=(13, 8.5))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1], hspace=0.34, wspace=0.28)
fig.suptitle("Inner-loop optimization vs MDI background intensity — "
             "landscapes, trajectories, anytime performance", fontsize=12.5)

for i, r in enumerate(runs):
    ax = fig.add_subplot(gs[0, i])
    S = np.ma.masked_where(~np.isfinite(r["score"]), r["score"])
    pm = ax.pcolormesh(r["Eg"], r["Rg"], S, cmap="Blues", vmin=vmin, vmax=vmax,
                       shading="nearest")
    hx = r["hx"]; n = len(hx)
    ax.scatter(hx[:, 1], hx[:, 0], c=np.arange(n), cmap="Greys", s=7,
               alpha=.6, linewidths=0, zorder=3)
    ax.plot([r["bE"]], [r["bR"]], marker="*", ms=17, color="#d95926",
            mec="white", mew=0.8, zorder=4)
    ax.set_title(f"{r['x']:g}x MDI  -  best {r['bS']:.3f}", fontsize=10.5,
                 color=shades[i])
    ax.set_xlabel("E threshold [GeV]")
    if i == 0: ax.set_ylabel("Cluster radius R [mm]")
    if i == len(runs) - 1:
        cb = fig.colorbar(pm, ax=ax, pad=0.02); cb.set_label("score", fontsize=9)

ax = fig.add_subplot(gs[1, 0:2])
for r, c in zip(runs, shades):
    best = np.maximum.accumulate(r["hf"])          # hf IS the score
    ok = np.isfinite(best)
    ev = np.arange(1, len(best) + 1)
    ax.plot(ev[ok], best[ok], color=c, lw=2,
            label=f"{r['x']:g}x  ({len(best)} evals, final {best[-1]:.3f})")
ax.set_xlabel("evaluations spent"); ax.set_ylabel("best score so far")
ax.set_title("Anytime performance of the inner-loop optimizer", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, title="MDI intensity")
ax.grid(alpha=0.25, lw=0.6)

ax = fig.add_subplot(gs[1, 2])
X = [r["x"] for r in runs]; S = [r["bS"] for r in runs]
ax.plot(X, S, "-", color="#0b5f6b", lw=1.6, zorder=1)
for x, s, c in zip(X, S, shades): ax.plot([x], [s], "o", ms=9, color=c, zorder=2)
for x, s in zip(X, S):
    ax.annotate(f"{s:.3f}", (x, s), textcoords="offset points", xytext=(6, 6),
                fontsize=8.5, color="#52514e")
ax.set_xlabel("MDI intensity [x nominal]"); ax.set_ylabel("best score")
ax.set_title("Achievable optimum", fontsize=10.5)
ax.grid(alpha=0.25, lw=0.6)

fig.savefig("bilevel_money_hpo.png", dpi=200, bbox_inches="tight")
fig.savefig("bilevel_money_hpo.pdf", bbox_inches="tight")
print("wrote bilevel_money_hpo.png/.pdf")
for r in runs:
    nfail = int(np.sum(~np.isfinite(r["hf"])))
    print(f"  {r['x']:g}x: {len(r['hx'])} evals ({nfail} failed), "
          f"best {r['bS']:.4f} at (R={r['bR']:.1f} mm, E={r['bE']:.3f} GeV)")
