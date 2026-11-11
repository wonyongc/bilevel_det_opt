#!/usr/bin/env python3
"""MDI background maps for the paper/deck, from escan_mdimap.json (escan_mdimap.py on Perlmutter).

    python make_mdimap_figs.py [escan_mdimap.json]       (PAPER=1 / SLIDE=1 as in the other scripts)

fig_mdi_rz      side view (|z|, r) of the background energy per crossing at the Z, three geometries
fig_mdi_theta   energy per crossing vs polar angle, four working points (crystals; timing layer dashed)
fig_mdi_faces   unrolled barrel (z, phi) and the two endcap faces (x, y) at the Z
fig_mdi_pairs   WarpX pair particles in the (theta, pT) plane at Z and tt, with what reaches where
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("SRC", "/mnt/user-data/uploads/MDI"), "escan_mdimap.json")
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
ORANGE = "#eb6834"
CAT = {"Z": "#2a78d6", "WW": "#eb6834", "ZH": "#1baf7a", "tt": "#6250d6"}
PTN = {"Z": "Z", "WW": "WW", "ZH": "ZH", "tt": "tt̄"}
CMAP = LinearSegmentedColormap.from_list("bl", ["#f4f7fb", "#b7d3f6", "#4f94e2", "#104281", "#0a2148"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.8,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
PAPER = bool(os.environ.get("PAPER"))
SLIDE = bool(os.environ.get("SLIDE")) or PAPER
if PAPER:
    _sub = plt.subplots
    plt.subplots = lambda *a, figsize=None, **k: _sub(*a, figsize=None if figsize is None else
                                                    (figsize[0] * (0.72 if figsize[0] > 10 else 0.85),
                                                     figsize[1] * (0.8 if figsize[0] > 10 else 0.85)), **k)

R = json.load(open(SRC)); S, E = R["samples"], {k: np.array(v) for k, v in R["edges"].items()}


def style(ax, grid=None):
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
    print("wrote", name)


def room(fig, wspace=0.12):
    fig.subplots_adjust(top=0.86 if SLIDE else 0.74, bottom=0.15, left=0.11 if PAPER else 0.07, right=0.88 if PAPER else 0.9, wspace=wspace)


def density(h, ex, ey):
    """energy per crossing per cm^2 (h in MeV per bin, edges in mm)"""
    a = np.outer(np.diff(ex), np.diff(ey)) / 100.0
    return np.where(h > 0, h / a, np.nan)


def pick(*names):
    return [n for n in names if n in S]


# ------------------------------------------------------------ F1: side view, three geometries ----
def fig_rz():
    geos = pick("escan_fld_Z_1x", "escan_bpas3_Z_1x", "escan_lumi2k_Z_1x")
    lab = {"escan_fld_Z_1x": "Crystals + field only", "escan_bpas3_Z_1x": "+ beam pipe, compensating field",
           "escan_lumi2k_Z_1x": "+ LumiCal (baseline)"}
    if not geos:
        return
    ez, er = E["rz_z"], E["rz_r"]
    D = [density(np.array(S[g]["rz"]), ez, er) for g in geos]
    vmax = np.nanmax([np.nanpercentile(d, 99.5) for d in D]); vmin = vmax * 1e-4
    fig, axes = plt.subplots(1, len(geos), figsize=(4.6 * len(geos) + 0.8, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g, d in zip(axes, geos, D):
        m = ax.pcolormesh(ez / 1000, er / 1000, d.T, cmap=CMAP, norm=LogNorm(vmin, vmax), shading="flat", rasterized=True)
        ax.axvline(R["zend_mm"] / 1000, color=MUTED, lw=0.8, ls=":")
        ax.add_patch(plt.Rectangle((1.074, 0.055), 0.1125, 0.057, fill=False, ec=ORANGE, lw=1.2))
        ax.annotate("LumiCal", (1.13, 0.115), xytext=(1.25, 0.55), fontsize=8.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
        ax.axvspan(1.19, 1.5, ymax=0.1 / 2.6, color=ORANGE, alpha=0.25, lw=0)
        ax.text(1.345, 0.13, "split", fontsize=8.5, color=ORANGE, ha="center")
        ax.set_title(f"{lab[g]}\n{S[g]['total_MeV']['mean']:,.0f} MeV per crossing", fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(0, 3.2); ax.set_ylim(0, 2.6); ax.set_xlabel("|z|  [m]")
        style(ax)
    axes[0].set_ylabel("r  [m]")
    room(fig)
    cb = fig.colorbar(m, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("MeV per cm² per crossing", color=INK)
    if not SLIDE:
        fig.text(0.012, 0.975, "Where the pair background lands in the calorimeter (Z, one crossing)",
                 fontsize=13.5, color=INK, ha="left", va="top", weight="bold")
        fig.text(0.012, 0.915, "Side view of the energy deposited in crystals and timing layer; dotted line: barrel/endcap boundary.",
                 fontsize=10, color=MUTED, ha="left", va="top")
    save(fig, "fig_mdi_rz")


# ------------------------------------------------------------ F1b: surface density along barrel and endcap --
def fig_profile():
    geos = pick("escan_fld_Z_1x", "escan_bpas3_Z_1x", "escan_lumi2k_Z_1x", "escan_lumi2k_tt_1x")
    lab = {"escan_fld_Z_1x": "Z: crystals + field only", "escan_bpas3_Z_1x": "Z: + beam pipe, compensating field",
           "escan_lumi2k_Z_1x": "Z: + LumiCal (baseline)", "escan_lumi2k_tt_1x": "tt̄: baseline"}
    sty = {"escan_fld_Z_1x": dict(color="#b7d3f6", lw=2), "escan_bpas3_Z_1x": dict(color="#4f94e2", lw=2),
           "escan_lumi2k_Z_1x": dict(color="#104281", lw=2.4), "escan_lumi2k_tt_1x": dict(color=CAT["tt"], lw=1.6, ls="--")}
    ez, er = E["rz_z"], E["rz_r"]; zc, rc = 0.5 * (ez[1:] + ez[:-1]), 0.5 * (er[1:] + er[:-1])
    ZE = R["zend_mm"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    zb = np.arange(0, ZE + 1, 100.0); rb = np.arange(450, 2401, 75.0)
    for g in geos:
        rz = np.array(S[g]["rz"])
        bar = np.where(zc < ZE, rz.sum(1), 0.0)
        rbar = rc[rz[zc < ZE].sum(0) > 0].min()
        hb = np.histogram(zc, zb, weights=bar)[0] / (2 * 2 * np.pi * rbar * np.diff(zb) / 100) * 1e3   # keV/cm^2, both halves
        end = rz[zc >= ZE].sum(0)
        he = np.histogram(rc, rb, weights=end)[0] / (2 * np.pi * (rb[1:] ** 2 - rb[:-1] ** 2) / 100) * 1e3  # both endcaps
        a0.step(0.5 * (zb[1:] + zb[:-1]) / 1000, hb, where="mid", label=lab[g], **sty[g])
        a1.step(0.5 * (rb[1:] + rb[:-1]) / 1000, he, where="mid", **sty[g])
    a0.set_xlabel("Barrel: |z|  [m]"); a1.set_xlabel("Endcaps: r  [m]")
    a0.set_ylabel("keV per cm² per crossing")
    for ax in (a0, a1):
        ax.set_yscale("log"); style(ax, "y")
    a0.set_xlim(0, ZE / 1000); a1.set_xlim(0.4, 2.4)
    lo = min(a0.get_ylim()[0], a1.get_ylim()[0]); hi = max(a0.get_ylim()[1], a1.get_ylim()[1])
    a0.set_ylim(lo, hi); a1.set_ylim(lo, hi)
    a0.set_ylim(0.015, 100); a1.set_ylim(0.015, 100)
    a0.legend(frameon=False, fontsize=9, loc="upper left")
    head(fig, "Background density on the calorimeter surface",
         "Crystals and timing layer, one bunch crossing; the beam pipe raises it everywhere, the LumiCal shadows the central barrel.")
    save(fig, "fig_mdi_profile")


# ------------------------------------------------------------ F2: polar angle, four working points --
def fig_theta():
    th = E["th"]; c = 0.5 * (th[1:] + th[:-1]) * 1000; w = np.diff(th) * 1000
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for p in ("Z", "WW", "ZH", "tt"):
        s = f"escan_lumi2k_{p}_1x"
        if s not in S:
            continue
        ax.step(c, np.array(S[s]["th_crystal"]) / w, where="mid", color=CAT[p], lw=1.6, label=PTN[p])
        ax.step(c, np.array(S[s]["th_timing"]) / w, where="mid", color=CAT[p], lw=1.0, ls="--", alpha=0.8)
    ax.set_yscale("log"); ax.set_xlim(150, 1571)
    ax.set_xlabel("Polar angle θ, folded  [mrad]"); ax.set_ylabel("MeV per mrad per crossing")
    ax.legend(frameon=False, fontsize=9.5, title="solid: crystals   dashed: timing layer", title_fontsize=9, ncol=4,
              loc="upper right")
    style(ax, "y")
    head(fig, "Background energy versus polar angle", "Baseline geometry, one bunch crossing; θ folded to 0–90°.")
    save(fig, "fig_mdi_theta")


# ------------------------------------------------------------ F3: barrel unrolled + endcap faces ----
def fig_faces(sample="escan_lumi2k_Z_1x"):
    if sample not in S:
        return
    s = S[sample]
    # barrel: per cm^2 using r ~ inner barrel radius estimated from the side view
    rz = np.array(s["rz"]); er = E["rz_r"]; ez = E["rz_z"]
    rb = er[:-1][(rz[ez[:-1] < R["zend_mm"]].sum(0) > 0)]
    rbar = float(rb.min()) if len(rb) else 2000.0
    area_b = np.outer(np.diff(E["bz"]), np.diff(E["bphi"]) * rbar) / 100.0
    hb = np.array(s["barrel"]); db = np.where(hb > 0, hb / area_b * 1e3, np.nan)
    dp = density(np.array(s["endcap_pos"]), E["exy"], E["exy"]) * 1e3; dn = density(np.array(s["endcap_neg"]), E["exy"], E["exy"]) * 1e3
    vmax = np.nanmax([np.nanpercentile(d, 99.5) for d in (db, dp, dn)]); vmin = vmax * 1e-3
    fig = plt.figure(figsize=(13.5, 4.6) if not PAPER else (13.5 * 0.72, 4.6 * 0.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])
    a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))
    m = a0.pcolormesh(E["bz"] / 1000, np.degrees(E["bphi"]), db.T, cmap=CMAP, norm=LogNorm(vmin, vmax), rasterized=True)
    a0.set_xlabel("z  [m]"); a0.set_ylabel("φ  [°]"); a0.set_yticks([-180, -90, 0, 90, 180])
    a0.set_title("Barrel, unrolled", fontsize=11, color=INK, loc="left")
    for ax, d, t in ((a1, dp, "Endcap z > 0"), (a2, dn, "Endcap z < 0")):
        ax.pcolormesh(E["exy"] / 1000, E["exy"] / 1000, d.T, cmap=CMAP, norm=LogNorm(vmin, vmax), rasterized=True)
        ax.set_aspect("equal"); ax.set_xlabel("x  [m]"); ax.set_title(t, fontsize=11, color=INK, loc="left")
        lim = np.max(np.abs(E["exy"][:-1][np.nansum(np.nan_to_num(d), 1) > 0])) / 1000 if np.nansum(np.nan_to_num(d)) else 2.6
        ax.set_xlim(-lim - 0.1, lim + 0.1); ax.set_ylim(-lim - 0.1, lim + 0.1)
    a1.set_ylabel("y  [m]")
    for ax in (a0, a1, a2):
        style(ax)
    room(fig, 0.3)
    cb = fig.colorbar(m, ax=[a0, a1, a2], fraction=0.02, pad=0.01); cb.set_label("keV per cm² per crossing", color=INK)
    if not SLIDE:
        fig.text(0.012, 0.975, "Pair background on the calorimeter surfaces (Z, one crossing)", fontsize=13.5, color=INK,
                 ha="left", va="top", weight="bold")
        fig.text(0.012, 0.915, "Crystals only; the beams cross in the x–z plane, outgoing beams towards +x.",
                 fontsize=10, color=MUTED, ha="left", va="top")
    save(fig, "fig_mdi_faces")
    return rbar


# ------------------------------------------------------------ F4: WarpX pairs in (theta, pT) -------
def fig_pairs(rbar_mm=2100.0):
    W = R.get("warpx", {})
    pts = [p for p in ("Z", "tt") if p in W]
    if not pts:
        return
    lpt = np.linspace(-6, 1, 36); lth = np.linspace(-5, 0.2, 27)          # as in check_yields.py
    fig, axes = plt.subplots(1, len(pts), figsize=(5.6 * len(pts) + 0.8, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    H = {p: sum(np.array(v) for k, v in W[p].items() if k.endswith("_pt_theta")) for p in pts}
    vmax = max(h.max() for h in H.values()); vmin = vmax * 1e-5
    B = 2.0
    pt_reach = lambda r_m: 0.3 * B * r_m / 2                                 # GeV: helix diameter = r
    for ax, p in zip(axes, pts):
        h = np.where(H[p] > 0, H[p], np.nan)
        m = ax.pcolormesh(10 ** lth * 1000, 10 ** lpt * 1000, h, cmap=CMAP, norm=LogNorm(vmin, vmax), rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvline(15, color=MUTED, lw=1, ls=":"); ax.text(17, 5e3, "outgoing\nbeams", fontsize=8.5, color=MUTED, va="top")
        ax.axvline(184, color=ORANGE, lw=1.2); ax.text(200, 2e-3, "crystals\nθ > 184 mrad", fontsize=8.5, color=ORANGE, va="bottom")
        for r_m, lab in ((0.01, "the beam pipe (r = 1 cm)"), (rbar_mm / 1000, f"r = {rbar_mm/1000:.1f} m (barrel)")):
            y = pt_reach(r_m) * 1000
            ax.axhline(y, color=MUTED, lw=1, ls="--"); ax.text(1.3e-2, y * 1.15, f"p$_T$ to reach {lab} in 2 T", fontsize=8.5, color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
        ax.set_xlim(1e-2, 1.6e3); ax.set_ylim(1e-3, 1e4)
        ax.set_xlabel("Polar angle θ, folded  [mrad]")
        ax.set_title(f"{PTN[p]}: {np.nansum(h):,.0f} pair particles per crossing", fontsize=10.5, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("Transverse momentum p$_T$  [MeV]")
    room(fig)
    cb = fig.colorbar(m, ax=axes, fraction=0.025, pad=0.01); cb.set_label("pair particles per bin per crossing", color=INK)
    if not SLIDE:
        fig.text(0.012, 0.975, "Incoherent pairs at the source: almost none reach the calorimeter directly", fontsize=13.5,
                 color=INK, ha="left", va="top", weight="bold")
        fig.text(0.012, 0.915, "WarpX output (e⁺ and e⁻, all channels) in the plane of polar angle and transverse momentum.",
                 fontsize=10, color=MUTED, ha="left", va="top")
    save(fig, "fig_mdi_pairs")


fig_profile()
fig_theta()
rb = fig_faces()
fig_pairs(rb or 2100.0)
for k, s in S.items():
    print(f"{k:24} {s['crossings']:5d} crossings  crystal {s['total_MeV']['mean']:8.0f} MeV  "
          f"(16-84 %: {s['total_MeV']['q']['16']:.0f}-{s['total_MeV']['q']['84']:.0f})")
