#!/usr/bin/env python3
"""fig_mdi_theta, redrawn on the crystal-tower grid (replaces fig_theta() of make_mdimap_figs.py).

    PAPER=1 python make_theta_fig.py [escan_theta_norm.json]     -> paper_fig_mdi_theta.pdf (+ .png)

Input: escan_theta_norm.json from escan/escan_theta_norm.py (Perlmutter). The hit position is the
crystal centre, so a histogram in fixed 10 mrad bins holds two or three tower rows per bin and
shows the same sawtooth in every sample. Here the energy is summed over groups of three
consecutive towers (front + rear crystal row each) and divided by the polar angle they span.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theta_profile import load, towers, profile

SRC = sys.argv[1] if len(sys.argv) > 1 else "escan_theta_norm.json"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
CAT = {"Z": "#2a78d6", "WW": "#eb6834", "ZH": "#1baf7a", "tt": "#6250d6"}
PTN = {"Z": "Z", "WW": "WW", "ZH": "ZH", "tt": "tt̄"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                     "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white"})
PAPER = bool(os.environ.get("PAPER")); M = int(os.environ.get("NTOWERS", "3"))
R, fe, fc = load(SRC)
fig, ax = plt.subplots(figsize=(7.4 * 0.85, 4.6 * 0.85) if PAPER else (7.4, 4.6))
out = {}
for layer, lw, ls, al in (("crystal", 1.6, "-", 1.0), ("timing", 1.0, "--", 0.8)):
    idx, cen = towers(np.array(R["cells"][layer]["fine"], float), fc)
    for p in ("Z", "WW", "ZH", "tt"):
        s = R["samples"].get(f"escan_lumi2k_{p}_1x")
        if not s: continue
        c, w, d = profile(np.array(s[layer]["fine"]), idx, cen, M)
        edges = np.concatenate([c - 0.5 * w, [c[-1] + 0.5 * w[-1]]])
        ax.stairs(d, edges, color=CAT[p], lw=lw, ls=ls, alpha=al, label=PTN[p] if layer == "crystal" else None, baseline=None)
        out[f"{layer}_{p}"] = {"theta_mrad": c.tolist(), "width_mrad": w.tolist(), "MeV_per_mrad": d.tolist()}
ax.set_yscale("log"); ax.set_xlim(150, 1571)
ax.set_xlabel("Polar angle θ, folded  [mrad]"); ax.set_ylabel("MeV per mrad per crossing")
ax.legend(frameon=False, fontsize=9.5, title="solid: crystals   dashed: timing layer", title_fontsize=9, ncol=4, loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
name = "paper_fig_mdi_theta" if PAPER else "fig_mdi_theta"
fig.savefig(name + ".pdf"); fig.savefig(name + ".png", dpi=150)
json.dump({"n_towers_per_bin": M, "crossings": R["samples"]["escan_lumi2k_Z_1x"]["crossings"], "profiles": out}, open("fig_mdi_theta_profile.json", "w"))
print("wrote", name + ".pdf", "and fig_mdi_theta_profile.json")
