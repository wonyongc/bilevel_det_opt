#!/usr/bin/env python3
"""Campaign v3 figures: the inner loop with the physical WarpX pair background.

    python make_v3_figs.py Z            (or Z WW ZH tt; reads campaign_v3_<pt>_results.json)

The inner-loop score (eres_true) is E_true / sigma_core (checked: 0x score = 1 / collect's
relative core width), so the re-optimized resolution at each background level is 1/score.
Fixed-settings numbers are collect_v2's: core width at the settings tuned with no background.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.environ.get("SRC", "/mnt/user-data/uploads/MDI")
INK, MUTED, GRID, SURF = "#1a1a19", "#6b6a63", "#e6e5df", "#ffffff"
RAMP = ["#b7d3f6", "#86b6ef", "#4f94e2", "#2a78d6", "#104281"]     # one hue, light -> dark = low -> high energy
ORANGE = "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.8,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
PAPER = bool(os.environ.get("PAPER"))
SLIDE = bool(os.environ.get("SLIDE")) or PAPER
LABS, X = ("0x", "1x", "3x"), np.array([0, 1, 3])
if PAPER:   # shrink the canvas so that the fixed font sizes read well at \linewidth
    _subplots = plt.subplots
    def _paper_subplots(*a, figsize=None, **k):
        if figsize is not None:
            f = 0.72 if figsize[0] > 10 else 0.85
            figsize = (figsize[0] * f, figsize[1] * f * (1.12 if figsize[0] > 10 else 1.0))
        return _subplots(*a, figsize=figsize, **k)
    plt.subplots = _paper_subplots
BARREL = ["g100MeV_fixed", "g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed", "g5000MeV_fixed"]
ENAME = {"g100MeV_fixed": "0.1 GeV", "g200MeV_fixed": "0.2 GeV", "g500MeV_fixed": "0.5 GeV",
         "g1000MeV_fixed": "1 GeV", "g5000MeV_fixed": "5 GeV"}
PTN = {"Z": "Z pole", "WW": "WW", "ZH": "ZH", "tt": "tt̄"}


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


def res_fixed(c):
    return np.array([c[l]["width"] * 100 for l in LABS])            # collect_v2 'width' is already sigma_core / E_true


def res_opt(c):
    return np.array([100 / c["optima"][l]["score"] for l in LABS])   # eres_true score = E_true / sigma_core


V2R = json.load(open(os.path.join(SRC, "campaign_v2_results.json")))
V2, V2BKG = V2R["configs"], V2R["forward"]["1x"]["energy_MeV_per_event_total"]

for pt in sys.argv[1:] or ["Z"]:
    R = json.load(open(os.path.join(SRC, f"campaign_v3_{pt}_results.json")))
    C, F = R["configs"], R["forward"]
    bkg1 = F["1x"]["energy_MeV_per_event_total"]

    # ---- F1: resolution vs background level, at fixed and at re-optimized settings ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    for ax, fn, ttl in ((axes[0], res_fixed, "Clustering tuned with no background"),
                        (axes[1], res_opt, "Clustering re-optimized at each level")):
        for b, col in zip(BARREL, RAMP):
            r = fn(C[b]) / fn(C[b])[0]
            ax.plot(X, r, "-o", color=col, lw=2, ms=7, mec="white", mew=1.5, zorder=3)
            ax.annotate(f"{ENAME[b]}  ×{r[-1]:.1f}", (X[-1], r[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        r2 = np.array([fn(V2[b]) / fn(V2[b])[0] for b in BARREL])
        ax.fill_between(X, r2.min(0), r2.max(0), color=MUTED, alpha=0.18, lw=0, zorder=1)
        ax.text(3.0, 0.955, "grey: earlier ad hoc background (v2), all energies", fontsize=8.5, color=MUTED, va="center", ha="right")
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(0.9, 9.5)
        ax.set_yticks([1, 1.5, 2, 3, 5, 8]); ax.set_yticklabels(["1", "1.5", "2", "3", "5", "8"]); ax.minorticks_off()
        ax.set_xticks(X); ax.set_xticklabels(["none", "1 crossing", "3 crossings"])
        ax.set_xlim(-0.2, 4.1)
        ax.set_xlabel("Background overlaid on each photon")
        ax.set_title(ttl, fontsize=11, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("Energy resolution / resolution without background")
    head(fig, f"With the physical background ({PTN[pt]}), the resolution degrades — re-optimizing recovers part of it",
         f"Barrel photons, 2,000 paired events each. Background: WarpX pairs, {bkg1:,.0f} MeV in the crystals per crossing\n"
         f"(earlier v2 sample: {V2BKG:.0f} MeV). Raw deposited energy, no digitization.", top=0.80)
    save(fig, f"fig_v3_{pt}_resolution")

    # ---- F2: where the optimum moves ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for b, col in zip(BARREL, RAMP):
        o = C[b]["optima"]
        Rm = [o[l]["R_mm"] for l in LABS]; Et = [o[l]["Ethr_GeV"] * 1e3 for l in LABS]
        axes[0].plot(X, Rm, "-o", color=col, lw=2, ms=7, mec="white", mew=1.5, zorder=3, label=ENAME[b])
        axes[1].plot(X, Et, "-o", color=col, lw=2, ms=7, mec="white", mew=1.5, zorder=3, label=ENAME[b])
    axes[0].set_ylabel("Optimal cluster radius  [mm]"); axes[0].set_ylim(0, 320)
    axes[0].axhline(300, color=MUTED, lw=0.8, ls=":"); axes[0].text(1.5, 307, "upper end of the search range", fontsize=8.5, color=MUTED, va="bottom", ha="center")
    axes[1].set_ylabel("Optimal hit threshold  [MeV]")
    for ax in axes:
        ax.set_xticks(X); ax.set_xticklabels(["none", "1 crossing", "3 crossings"]); ax.set_xlim(-0.2, 3.6)
        ax.set_xlabel("Background overlaid on each photon"); style(ax)
    axes[1].legend(frameon=False, loc="upper left", title="Barrel photon energy", title_fontsize=9, fontsize=9)
    head(fig, "The operating point moves: smaller clusters, higher thresholds",
         f"Inner-loop optimum of the seeded-radius clustering at each background level, {PTN[pt]} background.")
    save(fig, f"fig_v3_{pt}_optimum")

    # ---- console table ----
    print(f"\n{pt}: background {bkg1:.0f} MeV/crossing (1x), {F['3x']['energy_MeV_per_event_total']:.0f} (3x); "
          f"{F['1x']['hits_per_event_total']:.0f} / {F['3x']['hits_per_event_total']:.0f} crystals hit")
    for b in BARREL + ["e1GeV_fixed", "g200MeV_th200-300", "g200MeV_th300-500", "g1000MeV_th200-300", "g1000MeV_th300-500"]:
        c = C[b]
        print(f"  {b:20} fixed {' / '.join(f'{v:.2f}' for v in res_fixed(c))} %   re-opt {' / '.join(f'{v:.2f}' for v in res_opt(c))} %")
print("done")


# ---- F3 (simpler): actual resolution in %, one small panel per photon energy -------------
def pct(v):
    return f"{v:.2f}%" if v < 1 else (f"{v:.1f}%" if v < 100 else f"{v:.0f}%")


def fig_simple(pt, C):
    X2 = np.array([0, 1, 2])
    fig, axes = plt.subplots(1, len(BARREL), figsize=(14, 5.0))
    for ax, b in zip(axes, BARREL):
        fx, op = res_fixed(C[b]), res_opt(C[b])
        l1, = ax.plot(X2, fx, "-o", color=ORANGE, lw=2.2, ms=8, mec="white", mew=1.5, zorder=3)
        l2, = ax.plot(X2, op, "-o", color="#2a78d6", lw=2.2, ms=8, mec="white", mew=1.5, zorder=4)
        ax.set_ylim(0, max(fx.max(), op.max()) * 1.12)
        ax.annotate(pct(fx[-1]), (2, fx[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
        ax.annotate(pct(op[-1]), (2, op[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
        ax.set_xticks(X2); ax.set_xticklabels(["0", "1×", "3×"] if PAPER else ["none", "1×", "3×"]); ax.set_xlim(-0.3, 2.9)
        ax.set_title(f"{ENAME[b]} photons", fontsize=12, color=INK, loc="left", pad=22)
        ax.text(0, 1.03, f"{'no bkg' if PAPER else 'without background'}: {pct(fx[0])}", transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")
        style(ax)
    axes[0].set_ylabel("Energy resolution  [%]   (lower is better)")
    fig.supxlabel("Background added to each photon:  none · one bunch crossing (1×) · three crossings (3×)",
                  fontsize=10.5, color=INK, y=0.03)
    fig.legend([l1, l2], ["clustering kept at its no-background settings", "clustering re-tuned for the background"],
               loc="upper left", bbox_to_anchor=(0.012, 0.86 if not SLIDE else 1.0), ncol=2, frameon=False, fontsize=10.5)
    if SLIDE:
        fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    else:
        fig.text(0.012, 0.975, f"Energy resolution with the physical background ({PTN[pt]})", fontsize=13.5, color=INK,
                 ha="left", va="top", weight="bold")
        fig.text(0.012, 0.915, "Barrel photons, 2,000 events each; raw deposited energy. The gap between the lines is what "
                 "re-tuning the clustering recovers.", fontsize=10, color=MUTED, ha="left", va="top")
        fig.tight_layout(rect=[0, 0.04, 1, 0.80])
    save(fig, f"fig_v3_{pt}_resolution_pct")


for pt in sys.argv[1:] or ["Z"]:
    fig_simple(pt, json.load(open(os.path.join(SRC, f"campaign_v3_{pt}_results.json")))["configs"])


# ---- F4: all working points together: resolution vs background energy in the crystals --------
def fig_collapse(pts):
    CAT = {"Z": "#2a78d6", "WW": "#eb6834", "ZH": "#1baf7a", "tt": "#6250d6"}   # categorical slots 1,2,3,7
    MK = {"Z": "o", "WW": "s", "ZH": "D", "tt": "^"}
    show = ["g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed", "g5000MeV_fixed"]
    data = {}
    for pt in pts:
        p = os.path.join(SRC, f"campaign_v3_{pt}_results.json")
        if os.path.exists(p):
            data[pt] = json.load(open(p))
    fig, axes = plt.subplots(1, len(show), figsize=(14, 4.9))
    for ax, b in zip(axes, show):
        for pt, R in data.items():
            C, F = R["configs"], R["forward"]
            x = np.array([0, F["1x"]["energy_MeV_per_event_total"], F["3x"]["energy_MeV_per_event_total"]]) / 1000
            ax.plot(x, res_fixed(C[b]), MK[pt], mfc="none", mec=CAT[pt], mew=1.4, ms=9, alpha=0.6, zorder=3)
            ax.plot(x, res_opt(C[b]), MK[pt], color=CAT[pt], ms=8, mec="white", mew=1.2, zorder=5, label=PTN[pt])
        xs, ys = [], []
        for pt, R in data.items():
            F = R["forward"]
            xs += [0, F["1x"]["energy_MeV_per_event_total"] / 1000, F["3x"]["energy_MeV_per_event_total"] / 1000]
            ys += list(res_opt(R["configs"][b]))
        o = np.argsort(xs); ax.plot(np.array(xs)[o], np.array(ys)[o], "-", color=MUTED, lw=1, alpha=0.6, zorder=2)
        ax.set_ylim(0, None); ax.set_xlim(-0.4, 11.2)
        ax.set_title(f"{ENAME[b]} photons", fontsize=12, color=INK, loc="left")
        style(ax, "both")
    axes[0].set_ylabel("Energy resolution  [%]   (lower is better)")
    fig.supxlabel("Background energy in the crystals per event  [GeV]   (0 = none; then 1× and 3× bunch crossings)",
                  fontsize=10.5, color=INK, y=0.03)
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], marker="o", ls="", color=MUTED, ms=8), Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, mew=1.4, ms=9)]
    l += ["filled: clustering re-tuned", "open: no-background settings kept"]
    fig.legend(h, l, loc="upper left", ncol=len(h),
               bbox_to_anchor=(0.012, 0.86 if not SLIDE else 1.0), frameon=False, fontsize=9.5)
    if SLIDE:
        fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    else:
        fig.text(0.012, 0.975, "Across working points, the resolution depends only on how much background reaches the crystals",
                 fontsize=13.5, color=INK, ha="left", va="top", weight="bold")
        fig.text(0.012, 0.915, "Filled: clustering re-tuned for the background.  Open: clustering kept at its no-background settings. "
                 "Barrel photons, 2,000 events per point.\nZ at 3× and ZH at 1× carry the same background (≈3.3 GeV) and land on top of each other.", fontsize=10, color=MUTED, ha="left", va="top")
        fig.tight_layout(rect=[0, 0.04, 1, 0.76])
    save(fig, "fig_v3_all_collapse")


if os.environ.get("COLLAPSE"):
    fig_collapse(["Z", "WW", "ZH", "tt"])


# ---- F5: the background as a noise term: sqrt(sigma^2 - sigma_0^2) * E_true, in MeV, vs background energy ----
NOISE_B = ["g100MeV_fixed", "g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed"]
EGEV = {"g100MeV_fixed": 0.1, "g200MeV_fixed": 0.2, "g500MeV_fixed": 0.5, "g1000MeV_fixed": 1.0}


def noise_points(pts):
    out = []   # (pt, mult, Ebkg_GeV, base, noise_fixed_MeV, noise_opt_MeV)
    for pt in pts:
        p = os.path.join(SRC, f"campaign_v3_{pt}_results.json")
        if not os.path.exists(p):
            continue
        R = json.load(open(p)); C, F = R["configs"], R["forward"]
        for m, i in (("1x", 1), ("3x", 2)):
            eb = F[m]["energy_MeV_per_event_total"] / 1000
            for b in NOISE_B:
                s0, fx, op = res_fixed(C[b])[0], res_fixed(C[b])[i], res_opt(C[b])[i]
                q = lambda s: np.sqrt(max(s**2 - s0**2, 0)) / 100 * EGEV[b] * 1e3
                out.append((pt, m, eb, b, q(fx), q(op)))
    return out


def fig_noise(pts):
    CAT = {"Z": "#2a78d6", "WW": "#eb6834", "ZH": "#1baf7a", "tt": "#6250d6"}
    MK = {"Z": "o", "WW": "s", "ZH": "D", "tt": "^"}
    P = noise_points(pts)
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    seen = set()
    for pt, m, eb, b, nf, no in P:
        ax.plot(eb, nf, MK[pt], mfc="none", mec=CAT[pt], mew=1.4, ms=8, alpha=0.7, zorder=3)
        ax.plot(eb, no, MK[pt], color=CAT[pt], ms=7, mec="white", mew=1.0, zorder=5,
                label=PTN[pt] if pt not in seen else None)
        seen.add(pt)
    x = np.array([p[2] for p in P]); yo = np.array([p[5] for p in P]); yf = np.array([p[4] for p in P])
    for y, ls in ((yo, "-"), (yf, "--")):
        k, lc = np.polyfit(np.log(x), np.log(y), 1)
        xx = np.linspace(0.8, 11.5, 50)
        ax.plot(xx, np.exp(lc) * xx**k, ls, color=MUTED, lw=1, zorder=2)
        ax.annotate(f"∝ E$_{{bkg}}^{{{k:.2f}}}$", (xx[-1], np.exp(lc) * xx[-1]**k), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=MUTED)
        print(f"noise fit ({'re-tuned' if ls == '-' else 'kept'}): {np.exp(lc):.2f} MeV x (E_bkg/GeV)^{k:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 5, 10]); ax.set_xticklabels(["1", "2", "3", "5", "10"])
    ax.set_yticks([3, 5, 10, 20, 50, 100]); ax.set_yticklabels(["3", "5", "10", "20", "50", "100"]); ax.minorticks_off()
    ax.set_xlim(0.8, 16); ax.set_ylim(3, 110)
    ax.set_xlabel("Background energy in the crystals per event  [GeV]")
    ax.set_ylabel("Background term  $\\sqrt{\\sigma^2-\\sigma_0^2}$  [MeV]")
    style(ax, "both")
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, mew=1.4, ms=8)]
    l += ["open: no-background settings kept"]
    ax.legend(h, l, frameon=False, fontsize=9, loc="upper left", title="filled: re-tuned", title_fontsize=9)
    head(fig, "The background acts as an energy-independent noise term",
         "Barrel photons 0.1–1 GeV; quadrature difference to the no-background resolution, times E_true.", top=0.86)
    save(fig, "fig_v3_noise_term")


if os.environ.get("COLLAPSE"):
    fig_noise(["Z", "WW", "ZH", "tt"])
