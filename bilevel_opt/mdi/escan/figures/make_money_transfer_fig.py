#!/usr/bin/env python3
"""Figure for the transfer test: money_transfer.json (+ example traces recomputed for one task)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.argv = sys.argv[:1] + [os.environ.get("DUMP", "/mnt/user-data/uploads/MDI/money_dump.json")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import money_transfer as mt                                   # surrogate + strategies (main guarded)

PAPER = bool(os.environ.get("PAPER")); SLIDE = bool(os.environ.get("SLIDE")) or PAPER
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
COL = {"de_lin": "#8a8983", "de_log": "#2a78d6", "warm_de": "#1baf7a", "warm_nm": "#eb6834", "matched": "#6250d6", "real": INK}
LAB = {"de_lin": "cold DE\nlinear threshold\n(inner loop now)", "de_log": "cold DE\nlog threshold",
       "warm_de": "warm DE\nfrom lower\nbackground", "warm_nm": "warm local\nfrom lower\nbackground",
       "matched": "settings from other\nworking point at\nsame background"}
LEG = {"de_lin": "cold DE, linear threshold (inner loop now)", "de_log": "cold DE, log threshold",
       "warm_de": "warm DE from the lower background", "warm_nm": "warm local search from the lower background"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                     "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white"})
T = json.load(open("money_transfer.json")); B = T["budget"]; rows = T["rows"]


def collect(kind, st):
    v = []
    for r in rows:
        if r["kind"] == kind and st in r:
            v += [x if x is not None else B + 1 for x in r[st]]
    return np.array(v)


groups = [("de_lin", collect("background step", "de_lin")), ("de_log", collect("background step", "de_log")),
          ("warm_de", collect("background step", "warm_de")), ("warm_nm", collect("background step", "warm_nm")),
          ("matched", collect("matched background", "warm_nm"))]
real = np.array([r["real_run"] if r["real_run"] else B + 1 for r in rows if r["kind"] == "background step"])

fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.0) if not PAPER else (11.5, 4.9),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
rng = np.random.default_rng(0)
for i, (k, v) in enumerate(groups):
    y = np.where(v > B, B * 1.35, v)
    a0.scatter(i + rng.uniform(-0.28, 0.28, len(y)), y, s=5, color=COL[k], alpha=0.25, lw=0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    a0.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=2.2, zorder=4)
    a0.plot([i, i], [max(q1, 1), min(q3, B)], color=INK, lw=1, zorder=4)
    fail = (v > B).mean()
    a0.text(i, 1.9e3, f"median {int(med)}" + (f"\n{fail:.0%} not reached" if fail > 0.005 else ""), ha="center", fontsize=7.5 if PAPER else 8.5, color=INK)
a0.scatter(-0.45 + 0 * real, np.where(real > B, B * 1.35, real), marker="_", s=60, color=INK, lw=1, zorder=5)
a0.text(-0.45, 0.8, "real runs", fontsize=8, color=INK, ha="center", va="top", rotation=90)
a0.axhline(B * 1.35, color=GRID, lw=0.8); a0.text(4.45, B * 1.35, "not reached\nin 640", fontsize=8, color=MUTED, va="center", ha="right")
a0.set_yscale("log"); a0.set_ylim(0.6, 4e3)
a0.set_xticks(range(len(groups))); a0.set_xticklabels([LAB[k] for k, _ in groups], fontsize=7.5 if PAPER else 8.5)
a0.set_ylabel(f"Evaluations to reach within {T['eps']:.0%} of the optimum")
a0.set_yticks([1, 3, 10, 30, 100, 300]); a0.set_yticklabels(["1", "3", "10", "30", "100", "300"]); a0.minorticks_off()
for s in ("top", "right"):
    a0.spines[s].set_visible(False)
a0.grid(True, axis="y", color=GRID, lw=0.8); a0.set_axisbelow(True)
a0.set_title(f"{len(real)} background steps x {T['nseed']} seeds" if PAPER else f"All barrel samples and working points ({len(real)} background steps, {T['nseed']} seeds each)",
             fontsize=10, color=INK, loc="left")

# example traces: Z, 0.2 GeV photons, 3x, warm from 1x
tgt, src = ("Z", "g200MeV_fixed", "3x"), ("Z", "g200MeV_fixed", "1x")
S, rec = mt.SUR[tgt]; Ssrc, rsrc = mt.SUR[src]
start = mt.to_uv([rsrc["best_R_cluster_mm"]], [rsrc["best_E_threshold_GeV"]])[0]
x = np.arange(1, B + 1)
for k in ("de_lin", "de_log", "warm_de", "warm_nm"):
    tr = np.array([mt.run(k, S, 2000 + s, start)[:B] for s in range(12)])
    tr = np.array([np.pad(t, (0, B - len(t)), mode="edge") for t in tr])
    res = 100 / np.maximum(np.median(tr, 0), 1e-9)
    a1.plot(x, res, color=COL[k], lw=2, label=LEG[k])
hs = np.maximum.accumulate(S.hist_s)
a1.plot(np.arange(1, len(hs) + 1), 100 / np.maximum(hs, 1e-9), color=INK, lw=1, ls=":", label="the real inner-loop run")
a1.axhline(100 / ((1 - T["eps"]) * S.fmax), color=MUTED, lw=0.8, ls="--")
a1.set_xscale("log"); a1.set_xlim(1, B); a1.set_ylim(4.2, 7.5)
a1.set_xlabel("Evaluations"); a1.set_ylabel("Best resolution so far [%]" + ("" if PAPER else "  (median of 12 seeds)"))
a1.set_title("Z, 0.2 GeV γ, 3× (warm start from 1×)" if PAPER else "Example: Z, 0.2 GeV γ, 3 crossings (warm start from 1 crossing)", fontsize=10, color=INK, loc="left")
a1.legend(frameon=False, fontsize=7.5 if PAPER else 8.5, loc="upper right")
for s in ("top", "right"):
    a1.spines[s].set_visible(False)
a1.grid(True, axis="y", color=GRID, lw=0.8)
if SLIDE:
    fig.tight_layout()
else:
    fig.text(0.012, 0.985, "Carrying the optimum between outer-loop points cuts the inner-loop cost 30–250-fold",
             fontsize=13.5, color=INK, ha="left", va="top", weight="bold")
    fig.text(0.012, 0.93, "Strategies replayed on a smoothed surrogate of each measured landscape (the ~600 evaluated points + grid). "
             "Cold DE on the surrogate reproduces the real runs.", fontsize=10, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
name = "fig_money_transfer"
if PAPER:
    fig.savefig(f"paper_{name}.pdf")
elif SLIDE:
    fig.savefig(f"slide_{name}.png", dpi=200)
else:
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}", dpi=170)
print("wrote", name)
