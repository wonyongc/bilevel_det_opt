#!/usr/bin/env python3
"""Energy-scan figures (talk + paper), from the numbers measured so far.

    python make_escan_figs.py           -> fig_*.png / fig_*.pdf (titled, for the paper draft)
    SLIDE=1 python make_escan_figs.py   -> slide_*.png (no in-figure titles)

Same style as figures_v2/make_figs.py.  All inputs are per bunch crossing (1x)
unless marked 3x; crystal energy is SCEPCal_MainEdep, timing layer SCEPCal_TimingEdep.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, MUTED, GRID, SURF = "#1a1a19", "#6b6a63", "#e6e5df", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated)
R1X, R3X = "#2a78d6", "#104281"                               # one-hue ramp: 1x, 3x
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
PAPER = bool(os.environ.get("PAPER"))
SLIDE = bool(os.environ.get("SLIDE")) or PAPER


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def head(fig, title, sub, top=0.84):
    if SLIDE:
        fig.tight_layout(); return
    fig.text(0.012, 0.975, title, fontsize=13.5, color=INK, ha="left", va="top", weight="bold")
    fig.text(0.012, 0.915, sub, fontsize=10, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, top])


def save(fig, name):
    if PAPER:
        fig.savefig(f"paper_{name}.pdf")
    elif SLIDE:
        fig.savefig(f"slide_{name}.png", dpi=220)
    else:
        for ext in ("png", "pdf"):
            fig.savefig(f"{name}.{ext}", dpi=200)
    plt.close(fig)


# ============================================================ data ==========
PTS = ["Z", "WW", "ZH", "tt"]
LBL = {"Z": "Z", "WW": "WW", "ZH": "ZH", "tt": "tt̄"}
ECM = {"Z": 91.2, "WW": 160.0, "ZH": 240.0, "tt": 365.0}
# WarpX prod, per crossing
PAIRS = {"Z": 1334, "WW": 1424, "ZH": 3927, "tt": 3960}
PAIR_E = {"Z": 663, "WW": 1070, "ZH": 3862, "tt": 5789}             # GeV
LUMI_RATIO = {"Z": 1.14, "WW": 1.19, "ZH": 1.19, "tt": 1.18}
# Geant4, LumiCal baseline: (crystal total, barrel, endcaps, timing layer) MeV/crossing
LUMI = {("Z", "1x"): (1064.0, 446.2, 617.8, 163.7), ("WW", "1x"): (1113.1, 459.8, 653.4, 169.8),
        ("ZH", "1x"): (3393.7, 1367.0, 2026.6, 511.0), ("tt", "1x"): (3550.6, 1451.4, 2099.1, 537.0),
        ("Z", "3x"): (3153.5, 1321.6, 1831.8, 491.6), ("WW", "3x"): (3362.5, 1396.4, 1966.1, 512.2),
        ("ZH", "3x"): (10198.5, 4097.8, 6100.7, 1527.1), ("tt", "3x"): (10630.4, 4347.6, 6282.8, 1585.6)}
# geometry ladder at 1x: (total, barrel, endcaps, timing or None)
GEOM = [("xtal", "Crystals,\nair,\nfield to 2.18 m"), ("vac", "Same,\nvacuum\nworld"),
        ("fld", "Field to\n2.62 m"), ("bp", "+ beam\npipe,\nuniform 2 T"),
        ("bpas3", "+ comp.\nfield\nmap"), ("lumi", "+ LumiCal\n(baseline)")]
LADDER = {"Z": {"xtal": (189, 80, 109, None), "vac": (98, 17, 81, None), "fld": (131.9, 75.6, 56.3, 41.2),
                "bp": (2576, 985, 1591, None), "bpas3": (1352.2, 681.3, 670.8, 247.3),
                "lumi": (1064.0, 446.2, 617.8, 163.7)},
          "tt": {"xtal": (487, 187, 299, None), "vac": (272, 41, 231, None), "fld": (317.7, 176.0, 141.7, 96.3),
                 "bp": (8350, 3080, 5270, None), "bpas3": (4448.5, 2191.7, 2256.8, 784.1),
                 "lumi": (3550.6, 1451.4, 2099.1, 537.0)}}
# trace run (bpas3, Z, 1x, every particle kept): birth point of particles entering the crystals,
# share of the energy they carry in (energy at birth), table B summed over radius
BIRTH = [("Collision point", 8.9), ("Pipe cone\n0.09–1.19 m", 6.6), ("Pipe split\n1.19–1.5 m", 43.1),
         ("1.5–2.0 m", 11.7), ("2.0–2.3 m", 6.4), ("2.3–3.0 m\n(endcap region)", 10.2), ("3–6 m", 13.0)]
# table B cells (|z| bin x r bin), % of entering energy
ZB = ["0–\n0.09", "0.09–\n0.5", "0.5–\n1.07", "1.07–\n1.19", "1.19–\n1.5", "1.5–\n2.0", "2.0–\n2.3", "2.3–\n3.0", "3–\n6"]
RB = ["< 12", "12–35", "35–100", "100–250", "250–1000"]
CELLS = np.array([[8.9, 0, 0, 0, 0], [0.3, 2.4, 0, 0, 0], [0, 2.9, 0.1, 0, 0], [0, 0.8, 0.1, 0, 0],
                  [19.5, 18.8, 4.2, 0.6, 0], [0.8, 6.1, 4.2, 0.6, 0], [0, 4.0, 2.1, 0.3, 0],
                  [0, 5.0, 3.2, 2.0, 0], [0, 0.8, 7.0, 5.2, 0]])
COMPOSITION = [("Photons > 5 MeV", 31.3), ("Photons 0.6–5 MeV", 43.6), ("Photons < 0.6 MeV\n(mostly annihilation)", 11.8),
               ("Pair e± straight\nfrom the IP", 8.4), ("Secondary e±\nand other", 4.9)]

# ======================================================= F1: geometry ladder =
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, pt in zip(axes, ("Z", "tt")):
    x = np.arange(len(GEOM))
    bar = np.array([LADDER[pt][g][1] for g, _ in GEOM])
    end = np.array([LADDER[pt][g][2] for g, _ in GEOM])
    ax.bar(x, bar, 0.62, color=BLUE, edgecolor=SURF, lw=2, label="Barrel  (|z| < 2.3 m)", zorder=3)
    ax.bar(x, end, 0.62, bottom=bar, color=ORANGE, edgecolor=SURF, lw=2, label="Endcaps", zorder=3)
    for i, (g, _) in enumerate(GEOM):
        tot = LADDER[pt][g][0]
        ax.text(i, tot * 1.02 + 30, f"{tot:,.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in GEOM], fontsize=9)
    ax.set_ylabel("Crystal energy per crossing  [MeV]")
    ax.set_title(f"{LBL[pt]}  (√s = {ECM[pt]:g} GeV)", fontsize=11, color=INK, loc="left")
    ax.set_ylim(0, max(LADDER[pt][g][0] for g, _ in GEOM) * 1.12)
    style(ax)
axes[0].legend(frameon=False, loc="upper left")
head(fig, "Geometry, not the beams, sets the crystal background",
     "Same WarpX pairs (one bunch crossing), six standalone geometries. Relative to the field-only case the beam pipe "
     "raises the deposit 20–26×;\ncompensating field + LumiCal bring it to 8–11×.", top=0.80)
save(fig, "fig_escan_geometry_ladder")

# ========================================== F2: energy scan + what it tracks =
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
e = np.array([ECM[p] for p in PTS])
for m, col, mk in (("1x", R1X, "o"), ("3x", R3X, "s")):
    y = [LUMI[(p, m)][0] for p in PTS]
    t = [LUMI[(p, m)][3] for p in PTS]
    ax.plot(e, y, "-", marker=mk, color=col, lw=2, ms=8, mec="white", mew=1.5, zorder=3,
            label=f"Crystals, {m[0]}× background")
    ax.plot(e, t, "--", marker=mk, color=col, lw=2, ms=8, mec="white", mew=1.5, zorder=3,
            label=f"Timing layer, {m[0]}×")
    ax.annotate(f"Crystals {m[0]}×\n{y[-1]/1000:.1f} GeV", (e[-1], y[-1]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.annotate(f"Timing layer {m[0]}×\n{t[-1]/1000:.2f} GeV", (e[-1], t[-1]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
ax.set_yscale("log"); ax.set_ylim(100, 20000)
ax.set_yticks([100, 300, 1000, 3000, 10000]); ax.set_yticklabels(["100", "300", "1,000", "3,000", "10,000"])
ax.minorticks_off()
ax.set_xticks(e); ax.set_xticklabels([f"{LBL[p]}\n{ECM[p]:g}" for p in PTS])
ax.set_xlim(60, 500)
ax.set_xlabel("Working point,  √s  [GeV]"); ax.set_ylabel("Energy per crossing  [MeV]")
ax.set_title("LumiCal baseline, four working points", fontsize=11, color=INK, loc="left")
ax.text(0.02, 0.97, "solid: crystals    dashed: timing layer    ● 1×   ■ 3×", transform=ax.transAxes,
        fontsize=9, color=MUTED, va="top")
style(ax, "both")

ax = axes[1]
idx = {"Crystal energy": ([LUMI[(p, "1x")][0] for p in PTS], BLUE, "o", "-"),
       "Number of pair particles": ([PAIRS[p] for p in PTS], AQUA, "D", "--"),
       "Energy carried by the pairs": ([PAIR_E[p] for p in PTS], ORANGE, "^", ":")}
for name, (v, col, mk, ls) in idx.items():
    v = np.array(v, float) / v[0]
    ax.plot(e, v, ls, marker=mk, color=col, lw=2, ms=8, mec="white", mew=1.5, zorder=3, label=name)
    ax.annotate(f"×{v[-1]:.1f}", (e[-1], v[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
ax.set_xticks(e); ax.set_xticklabels([f"{LBL[p]}\n{ECM[p]:g}" for p in PTS]); ax.set_xlim(60, 420)
ax.set_ylim(0, 9.5)
ax.set_xlabel("Working point,  √s  [GeV]"); ax.set_ylabel("Relative to the Z pole")
ax.set_title("It follows the number of pairs, not their energy", fontsize=11, color=INK, loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=9)
style(ax)
head(fig, "Crystal background across the FCC-ee energy scan",
     "0.78–0.90 MeV in the crystals per pair particle at every working point; 3× = 2.96–3.02 × 1×. "
     "Crystals + IDEA beam pipe, compensating field, LumiCal.")
save(fig, "fig_escan_energy_scan")

# ====================================== F3: where the crystal energy starts ==
fig = plt.figure(figsize=(12.5, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
ax = fig.add_subplot(gs[0])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
im = ax.imshow(CELLS.T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=20)
for i in range(CELLS.shape[0]):
    for j in range(CELLS.shape[1]):
        v = CELLS[i, j]
        if v >= 0.5:
            ax.text(i, j, f"{v:.0f}" if v >= 1.5 else f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 9 else INK)
ax.add_patch(Rectangle((3.5, -0.5), 1, 3, fill=False, ec=ORANGE, lw=2, zorder=4))
ax.text(4.0, 3.75, "orange box: pipe split\n+ field reversal", ha="center", va="center", fontsize=9.5, color=INK)
ax.set_xticks(range(len(ZB))); ax.set_xticklabels(ZB, fontsize=9)
ax.set_yticks(range(len(RB))); ax.set_yticklabels(RB)
ax.set_xlabel("Distance along the beam,  |z|  [m]"); ax.set_ylabel("Distance from the axis,  r  [mm]")
ax.set_title("Birth point of particles entering the crystals  (% of their energy)", fontsize=11, color=INK, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("% of entering energy"); cb.outline.set_visible(False)

ax = fig.add_subplot(gs[1])
names = [n for n, _ in COMPOSITION][::-1]; vals = [v for _, v in COMPOSITION][::-1]
cols = [MUTED, AQUA, BLUE, BLUE, BLUE][::-1]
cols = [BLUE, BLUE, BLUE, AQUA, MUTED][::-1]
ax.barh(range(len(vals)), vals, 0.62, color=cols, edgecolor=SURF, lw=2, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=9.5)
ax.set_xlim(0, 52); ax.set_xlabel("% of energy entering the crystals")
ax.set_title("What enters the crystals", fontsize=11, color=INK, loc="left")
style(ax, "x")
head(fig, "Crystal background = shower photons from pairs hitting the pipe split and the beam line",
     "Trace run, every particle kept (beam pipe + compensating field, no LumiCal; Z, 30 crossings). Energy at birth:\n"
     "photons lose some on the way, so distant sources are somewhat over-weighted.", top=0.80)
save(fig, "fig_escan_trace")

# ================================================ F4: the WarpX pair source ==
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
x = np.arange(len(PTS))
ax = axes[0]
ax.bar(x, [PAIRS[p] for p in PTS], 0.6, color=AQUA, edgecolor=SURF, lw=2, zorder=3)
for i, p in enumerate(PTS):
    ax.text(i, PAIRS[p] + 60, f"{PAIRS[p]:,}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"{LBL[p]}\n{ECM[p]:g} GeV" for p in PTS])
ax.set_ylabel("e⁺ + e⁻ per crossing"); ax.set_ylim(0, 4600)
ax.set_title("Incoherent pair particles", fontsize=11, color=INK, loc="left"); style(ax)
ax = axes[1]
mean = [PAIR_E[p] / PAIRS[p] * 1000 for p in PTS]
ax.bar(x, mean, 0.6, color=ORANGE, edgecolor=SURF, lw=2, zorder=3)
for i, v in enumerate(mean):
    ax.text(i, v + 25, f"{v:,.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"{LBL[p]}\n{ECM[p]:g} GeV" for p in PTS])
ax.set_ylabel("Mean energy per particle  [MeV]"); ax.set_ylim(0, 1700)
ax.set_title("Their mean energy", fontsize=11, color=INK, loc="left"); style(ax)
head(fig, "WarpX pair source at the four FCC-ee working points",
     "FSR 2025 parameters, 30 mrad crossing, generated QED table. Luminosity per crossing 14–19 % above the FSR value "
     "(" + ", ".join(f"{LBL[p]} {LUMI_RATIO[p]:.2f}" for p in PTS) + ").")
save(fig, "fig_escan_pairs")
print("done")
