#!/usr/bin/env python3
"""The money plot with the physical pair background (campaign v3), from money_dump.json.

    python make_money_v3.py [money_dump.json]      env PT (Z), BASE (g200MeV_fixed), PAPER/SLIDE

Top: inner-loop landscapes at 0x / 1x / 3x background (resolution 100/score, kernel-smoothed from
the ~600 evaluated points and the 12 x 25 grid), with every point the optimizer evaluated, the
optimum (star) and the no-background optimum (open circle).
Bottom: anytime performance of the real runs; optimal radius and threshold against the background
energy in the crystals, all four working points.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/MDI/money_dump.json"
V3 = os.environ.get("V3SRC", "/mnt/user-data/uploads/MDI")
PT, BASE = os.environ.get("PT", "Z"), os.environ.get("BASE", "g200MeV_fixed")
PAPER = bool(os.environ.get("PAPER")); SLIDE = bool(os.environ.get("SLIDE")) or PAPER
INK, MUTED, GRID, ORANGE = "#1a1a19", "#6b6a63", "#e6e5df", "#eb6834"
CAT = {"Z": "#2a78d6", "WW": "#eb6834", "ZH": "#1baf7a", "tt": "#6250d6"}
MK = {"Z": "o", "WW": "s", "ZH": "D", "tt": "^"}
PTN = {"Z": "Z", "WW": "WW", "ZH": "ZH", "tt": "tt̄"}
ENAME = {"g100MeV_fixed": "0.1 GeV γ", "g200MeV_fixed": "0.2 GeV γ", "g500MeV_fixed": "0.5 GeV γ",
         "g1000MeV_fixed": "1 GeV γ", "g5000MeV_fixed": "5 GeV γ", "e1GeV_fixed": "1 GeV e⁻"}
LVLC = {"0x": "#86b6ef", "1x": "#2a78d6", "3x": "#0a2148"}
CMAP = LinearSegmentedColormap.from_list("res", ["#0a2148", "#104281", "#2a78d6", "#86b6ef", "#dbe8f8", "#f6f8fb"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.8,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
D = json.load(open(SRC))
LVL = {0.0: "0x", 1.0: "1x", 3.0: "3x"}
LE0, LE1 = -5.0, np.log10(0.6)


def recs(pt, base):
    return {LVL[float(r["geom_mdi_x"])]: r for r in D[f"money_v3_{pt}_{base}"].values()}


def smooth(rec, R, E, h=0.025):
    gR, gE = np.meshgrid(rec["grid_R_cluster_mm"], rec["grid_E_threshold_GeV"], indexing="ij")
    hx, hf = np.array(rec["opt_history_x"]), np.array(rec["opt_history_f"])
    P = np.vstack([np.column_stack([gR.ravel() / 300, (np.log10(np.maximum(gE.ravel(), 1e-5)) - LE0) / (LE1 - LE0)]),
                   np.column_stack([hx[:, 0] / 300, (np.log10(np.maximum(hx[:, 1], 1e-5)) - LE0) / (LE1 - LE0)])])
    F = np.concatenate([np.array(rec["score"]).ravel(), hf])
    Q = np.column_stack([R.ravel() / 300, (np.log10(E.ravel()) - LE0) / (LE1 - LE0)])
    out = []
    for i in range(0, len(Q), 3000):
        q = Q[i:i + 3000]; d2 = ((q[:, None, :] - P[None]) ** 2).sum(-1); w = np.exp(-0.5 * d2 / h ** 2) + 1e-300
        out.append((w * F).sum(1) / w.sum(1))
    return np.concatenate(out).reshape(R.shape)


def style(ax, grid=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ebkg(pt):
    p = os.path.join(V3, f"campaign_v3_{pt}_results.json")
    F = json.load(open(p))["forward"]
    return {"0x": 0.0, "1x": F["1x"]["energy_MeV_per_event_total"] / 1000, "3x": F["3x"]["energy_MeV_per_event_total"] / 1000}


def main():
    rr = recs(PT, BASE); eb = ebkg(PT)
    fig = plt.figure(figsize=(14, 9.6) if not PAPER else (12, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], width_ratios=[1, 1, 1, 0.06], hspace=0.45, wspace=0.34 if PAPER else 0.28,
                          left=0.07 if not PAPER else 0.09, right=0.95, top=0.86 if not SLIDE else 0.95, bottom=0.08)
    Eg = np.logspace(LE0, LE1, 160); Rg = np.linspace(0, 300, 121)
    EE, RR = np.meshgrid(Eg, Rg)
    o0 = rr["0x"]
    for i, lv in enumerate(("0x", "1x", "3x")):
        ax = fig.add_subplot(gs[0, i]); r = rr[lv]
        s = smooth(r, RR, EE)
        res = np.where(s > 0.5, 100 / np.maximum(s, 1e-9), np.nan)
        m = ax.pcolormesh(EE * 1e3, RR, res, cmap=CMAP, norm=LogNorm(2, 30), shading="auto", rasterized=True)
        hx = np.array(r["opt_history_x"])
        ax.scatter(hx[:, 1] * 1e3, hx[:, 0], s=3, c=INK, alpha=0.35, lw=0, zorder=3)
        ax.plot(r["best_E_threshold_GeV"] * 1e3, r["best_R_cluster_mm"], "*", ms=17, color=ORANGE, mec="white", mew=1.0, zorder=5)
        if lv != "0x":
            ax.plot(o0["best_E_threshold_GeV"] * 1e3, o0["best_R_cluster_mm"], "o", ms=10, mfc="none", mec=ORANGE, mew=1.6, zorder=5)
        ax.set_xscale("log"); ax.set_xlim(1e-2, 600); ax.set_ylim(0, 300)
        ax.set_xlabel("Per-crystal threshold  [MeV]")
        if i == 0:
            ax.set_ylabel("Cluster radius  [mm]")
        t = ("no background" if lv == "0x" else f"{lv}, {eb[lv]:.1f} GeV") if PAPER else ("no background" if lv == "0x" else f"{lv[0]} crossing{'s' if lv == '3x' else ''}, {eb[lv]:.1f} GeV in crystals")
        ax.set_title(f"{t}\nbest {100 / r['best_score']:.2f} %", fontsize=10.5, color=INK, loc="left")
        style(ax)
    cax = fig.add_subplot(gs[0, 3]); cb = fig.colorbar(m, cax=cax)
    cb.set_label("energy resolution  [%]", color=INK); cb.set_ticks([2, 3, 5, 10, 20, 30]); cb.set_ticklabels(["2", "3", "5", "10", "20", "30"])

    # anytime performance (real runs)
    ax = fig.add_subplot(gs[1, 0])
    for lv in ("0x", "1x", "3x"):
        f = np.maximum.accumulate(np.array(rr[lv]["opt_history_f"]))
        y = np.where(f > 0, 100 / np.maximum(f, 1e-9), np.nan)
        ax.plot(np.arange(1, len(y) + 1), y, color=LVLC[lv], lw=2, label={"0x": "none", "1x": "1 crossing", "3x": "3 crossings"}[lv])
    ax.set_yscale("log"); ax.set_ylim(2, 40); ax.set_yticks([2, 3, 5, 10, 20]); ax.set_yticklabels(["2", "3", "5", "10", "20"]); ax.minorticks_off()
    ax.set_xlabel("Evaluations"); ax.set_ylabel("Best resolution so far  [%]")
    ax.set_title("Inner-loop convergence", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, title="background", title_fontsize=9); style(ax, "y")

    # optimum against background energy, all working points
    ax1, ax2 = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    for b, ls in (("g200MeV_fixed", "-"), ("g1000MeV_fixed", "--")):
        for pt in ("Z", "WW", "ZH", "tt"):
            k = f"money_v3_{pt}_{b}"
            if k not in D:
                continue
            r = recs(pt, b); e = ebkg(pt)
            x = [e[l] for l in ("0x", "1x", "3x")]
            ax1.plot(x, [r[l]["best_R_cluster_mm"] for l in ("0x", "1x", "3x")], ls=ls, marker=MK[pt], color=CAT[pt], ms=6, lw=1.2,
                     mec="white", label=f"{PTN[pt]}" if b == "g200MeV_fixed" else None)
            ax2.plot(x, [r[l]["best_E_threshold_GeV"] * 1e3 for l in ("0x", "1x", "3x")], ls=ls, marker=MK[pt], color=CAT[pt],
                     ms=6, lw=1.2, mec="white")
    ax1.set_ylim(0, 320); ax1.set_ylabel("Optimal radius  [mm]"); ax2.set_ylabel("Optimal threshold  [MeV]")
    for ax in (ax1, ax2):
        ax.set_xlabel("$E_{\\rm bkg}$ in crystals  [GeV]" if PAPER else "Background energy in crystals  [GeV]"); ax.set_xlim(-0.5, 11.5); style(ax, "y")
    ax1.set_title("Optimum vs background" if PAPER else "The optimum moves (all working points)", fontsize=10.5, color=INK, loc="left")
    ax2.set_title("solid 0.2 GeV γ, dashed 1 GeV γ", fontsize=9 if PAPER else 9.5, color=MUTED, loc="left")
    ax1.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    if not SLIDE:
        fig.text(0.012, 0.985, f"Inner loop with the physical pair background ({PTN[PT]}, {ENAME[BASE]}, barrel)",
                 fontsize=13.5, color=INK, ha="left", va="top", weight="bold")
        fig.text(0.012, 0.945, "Landscapes: resolution 100/score, smoothed from every evaluated point (dots); star = optimum, "
                 "circle = optimum without background. Raw deposited energy.", fontsize=10, color=MUTED, ha="left", va="top")
    name = f"fig_money_v3_{PT}_{BASE.split('_')[0]}"
    if PAPER:
        fig.savefig(f"paper_{name}.pdf")
    elif SLIDE:
        fig.savefig(f"slide_{name}.png", dpi=200)
    else:
        for ext in ("png", "pdf"):
            fig.savefig(f"{name}.{ext}", dpi=170)
    print("wrote", name)


main()
