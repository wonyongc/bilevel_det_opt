#!/usr/bin/env python3
"""
plot_mdi_diagnostics.py — v5

Diagnostic plotter for an MDI cache directory.
Reads the HepMC3 file produced by the openPMD→HepMC converter and produces
a six-panel figure suitable for the FCC-ee MDI status deck and paper.

Design changes from v4:
  * 2x3 grid (was 3x3-ish) — no wasted axes, no panel-title overlap
  * Suptitle + subtitle decoupled; caption at the bottom of the figure
  * Polar-angle panel uses min(θ, π-θ) on a log axis ("angle from nearest
    beam axis"). With collinear emission this is the natural binning;
    the old linear θ histogram showed an empty middle and unreadable
    spikes at the edges.
  * Angular map replaced by E vs angle-from-beam (2D hexbin, log-log).
    The old θ-vs-φ map was degenerate because all photons are near the
    poles; the new panel actually shows physics.
  * Multiplicity panel: single annotation, no text overlap
  * pz F/B symmetry: forward / backward split by colour, asymmetry A printed

Usage:
  python plot_mdi_diagnostics.py --cache-dir $MDI_CACHE/scepcal_b33a3b3890ec0348 \
                                 --out mdi_smoke_v5.png
  python plot_mdi_diagnostics.py --hepmc /path/to/file.hepmc \
                                 --out mdi_smoke_v5.png

You can also import make_figure() and pass numpy arrays directly.
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---- Style ----------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "DejaVu Sans",
    "mathtext.fontset":  "cm",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "figure.dpi":        110,
})

COLOR_GAMMA = "#0F766E"   # teal — photons (consistent with deck/paper)
COLOR_FWD   = "#0EA5E9"   # sky blue — forward beam
COLOR_BWD   = "#F97316"   # orange — backward beam
COLOR_MUTE  = "#6B7280"   # caption text
COLOR_FAINT = "#D1D5DB"   # gridlines


# ---- HepMC3 reader --------------------------------------------------------
def read_hepmc(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal HepMC3 ASCII reader.

    Returns four numpy arrays (E, px, py, pz) in GeV for all final-state
    (status 1) particles across all events in the file.

    Format reference (HepMC3 ASCII v3):
      P  id  vertex_id  pid  px  py  pz  E  m  status
    """
    Es, pxs, pys, pzs = [], [], [], []
    n_lines = 0
    with open(path) as f:
        for line in f:
            n_lines += 1
            if not line.startswith("P "):
                continue
            tok = line.split()
            # P records have 10 fields after the leading 'P'
            if len(tok) < 10:
                continue
            try:
                px     = float(tok[4])
                py     = float(tok[5])
                pz     = float(tok[6])
                E      = float(tok[7])
                status = int(tok[9])
            except (ValueError, IndexError):
                continue
            if status != 1:
                continue
            Es.append(E)
            pxs.append(px)
            pys.append(py)
            pzs.append(pz)
    if not Es:
        print(f"WARN: read 0 particles from {path} ({n_lines} lines scanned)",
              file=sys.stderr)
    return (np.asarray(Es), np.asarray(pxs), np.asarray(pys), np.asarray(pzs))


def angle_from_beam(px, py, pz) -> np.ndarray:
    """
    min(θ, π-θ) in radians — angle to the nearest beam axis.
    Folds forward / backward beams onto a single half-axis so the
    collinear structure is visible on a log scale.
    """
    p = np.sqrt(px*px + py*py + pz*pz)
    cos_theta = np.where(p > 0, pz / p, 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.minimum(theta, np.pi - theta)


# ---- Figure construction --------------------------------------------------
def make_figure(E, px, py, pz, run_label=None, params_label=None):
    """
    Build the 6-panel diagnostic figure. Returns the matplotlib Figure.

    Inputs:
        E, px, py, pz : 1D numpy arrays in GeV
        run_label     : short string for the figure suptitle (e.g. 'Z-pole')
        params_label  : short string of run params for the subtitle
    """
    n = len(E)
    if n == 0:
        raise ValueError("No particles to plot")

    pT      = np.sqrt(px*px + py*py)
    aba_rad = angle_from_beam(px, py, pz)
    aba_deg = np.degrees(aba_rad)

    fig = plt.figure(figsize=(13.5, 7.5))
    gs  = fig.add_gridspec(2, 3,
                           hspace=0.45, wspace=0.30,
                           left=0.06, right=0.97,
                           top=0.86, bottom=0.10)

    # ---- (0,0) Energy spectrum ----
    ax = fig.add_subplot(gs[0, 0])
    E_pos = E[E > 0]
    bins  = np.logspace(np.log10(E_pos.min()),
                        np.log10(E_pos.max() * 1.05), 70)
    ax.hist(E_pos, bins=bins, color=COLOR_GAMMA,
            histtype="step", linewidth=1.6,
            label=fr"$\gamma$ : {n:,}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("E [GeV]"); ax.set_ylabel("counts / bin")
    ax.set_title("Photon energy spectrum")
    ax.legend(loc="upper left")
    ax.grid(True, which="major", color=COLOR_FAINT, lw=0.5)
    # annotate the max in a non-intrusive corner
    ax.text(0.97, 0.05,
            fr"$E_{{\max}} = {E.max()*1000:.1f}$ MeV",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=COLOR_MUTE)

    # ---- (0,1) pT spectrum ----
    ax = fig.add_subplot(gs[0, 1])
    pT_pos = pT[pT > 0]
    if len(pT_pos) > 0:
        bins = np.logspace(np.log10(pT_pos.min()),
                           np.log10(pT_pos.max() * 1.05), 70)
        ax.hist(pT_pos, bins=bins, color=COLOR_GAMMA,
                histtype="step", linewidth=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$p_T$ [GeV/$c$]"); ax.set_ylabel("counts / bin")
    ax.set_title("Transverse momentum")
    ax.grid(True, which="major", color=COLOR_FAINT, lw=0.5)
    ax.text(0.97, 0.05,
            fr"median $\sim {np.median(pT_pos)*1e3:.2f}$ MeV/$c$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=COLOR_MUTE)

    # ---- (0,2) Collinearity: angle from nearest beam axis ----
    ax = fig.add_subplot(gs[0, 2])
    aba_pos = aba_deg[aba_deg > 0]
    if len(aba_pos) > 0:
        lo = max(aba_pos.min(), 1e-7)
        bins = np.logspace(np.log10(lo), np.log10(90.0), 70)
        ax.hist(aba_pos, bins=bins, color=COLOR_GAMMA,
                histtype="step", linewidth=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"angle from nearest beam [deg]")
    ax.set_ylabel("counts / bin")
    ax.set_title(r"Collinearity, $\min(\theta,\,\pi-\theta)$")
    ax.grid(True, which="major", color=COLOR_FAINT, lw=0.5)
    # reference scales
    for x, lab in [(1e-4, "$10^{-4}$"), (1e-2, "$10^{-2}$"), (1.0, "$1^\\circ$")]:
        if lo < x < 90:
            ax.axvline(x, color=COLOR_MUTE, lw=0.4, ls=":")

    # ---- (1,0) E vs angle from beam (hexbin, log-log) ----
    ax = fig.add_subplot(gs[1, 0])
    mask = (aba_deg > 0) & (E > 0)
    if mask.sum() > 0:
        hb = ax.hexbin(aba_deg[mask], E[mask],
                       xscale="log", yscale="log",
                       gridsize=35, mincnt=1,
                       cmap="viridis", bins="log")
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label(r"$\log_{10}$ counts", fontsize=9)
    ax.set_xlabel(r"angle from beam axis [deg]")
    ax.set_ylabel(r"$E$ [GeV]")
    ax.set_title("Energy vs. collinearity")

    # ---- (1,1) pz forward/backward symmetry ----
    ax = fig.add_subplot(gs[1, 1])
    if len(pz) > 0:
        pz_lim = max(abs(np.percentile(pz,  1)),
                     abs(np.percentile(pz, 99)),
                     1e-3)
        bins = np.linspace(-pz_lim, pz_lim, 81)
        fwd = pz[pz > 0]; bwd = pz[pz < 0]
        ax.hist(fwd, bins=bins, color=COLOR_FWD, alpha=0.75,
                label=f"forward $p_z>0$: {len(fwd):,}")
        ax.hist(bwd, bins=bins, color=COLOR_BWD, alpha=0.75,
                label=f"backward $p_z<0$: {len(bwd):,}")
        ax.set_yscale("log")
        # asymmetry
        N = len(fwd) + len(bwd)
        A = (len(fwd) - len(bwd)) / N if N > 0 else 0.0
        ax.text(0.02, 0.97,
                fr"asymmetry $A = {A:+.3f}$",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color=COLOR_MUTE,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec=COLOR_MUTE, lw=0.5))
    ax.axvline(0, color=COLOR_MUTE, lw=0.5)
    ax.set_xlabel(r"$p_z$ [GeV/$c$]"); ax.set_ylabel("counts / bin")
    ax.set_title("F/B symmetry")
    ax.legend(loc="upper right")
    ax.grid(True, which="major", color=COLOR_FAINT, lw=0.5)

    # ---- (1,2) Multiplicity ----
    # We only have photons in the read; if the user wants pair species
    # they'd need to be passed separately. For the smoke-test case the
    # pair species are zero by construction.
    ax = fig.add_subplot(gs[1, 2])
    species_names = [r"$\gamma$" + "\n(beamstr.)",
                     r"$e^-$"   + "\n(pair)",
                     r"$e^+$"   + "\n(pair)"]
    counts = [n, 0, 0]
    colors = [COLOR_GAMMA, COLOR_FAINT, COLOR_FAINT]
    # log scale requires nonzero base; use 0.5 floor
    floored = [max(c, 0.5) for c in counts]
    bars = ax.bar(species_names, floored, color=colors,
                  edgecolor=[COLOR_GAMMA, COLOR_MUTE, COLOR_MUTE], linewidth=1)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * (1.4 if c > 0 else 1.4),
                f"{c:,}", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=COLOR_GAMMA if c > 0 else COLOR_MUTE)
    ax.set_yscale("log")
    ax.set_ylim(0.4, max(counts) * 5)
    ax.set_ylabel("count / bunch crossing")
    ax.set_title("Multiplicity by species")
    # single muted footer annotation
    ax.text(0.5, -0.32,
            r"pair species suppressed by Breit–Wheeler at $\chi \ll 1$",
            transform=ax.transAxes, ha="center", fontsize=9,
            color=COLOR_MUTE, style="italic")
    ax.grid(True, which="major", axis="y", color=COLOR_FAINT, lw=0.5)

    # ---- Suptitle + subtitle + footer ----
    title = "FCC-ee MDI smoke test"
    if run_label:
        title += f" — {run_label}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.965)
    if params_label:
        fig.text(0.5, 0.915, params_label,
                 ha="center", fontsize=10.5, color=COLOR_MUTE)
    fig.text(0.5, 0.015,
             "Smoke-test fidelity: $\\chi_{\\min}=10^{-5}$ engine threshold lowered to fire "
             "at our undersampled $\\chi$. "
             "Multiplicity and spectrum should NOT be quoted as FCC-ee predictions.",
             ha="center", fontsize=8.5, color=COLOR_MUTE, style="italic")
    return fig


# ---- CLI ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path,
                    help="MDI cache directory; will look for *.hepmc inside")
    ap.add_argument("--hepmc", type=Path,
                    help="Explicit .hepmc file (takes priority over --cache-dir)")
    ap.add_argument("--out", type=Path, default=Path("mdi_diag.png"))
    ap.add_argument("--run-label", default="Z-pole, 29k photons",
                    help="Short label appended to the suptitle")
    ap.add_argument("--params-label",
                    default=r"$\sigma_y = 3\,\mu\mathrm{m}$, $2\times10^6$ macros, $256\!\times\!128\!\times\!256$ cells",
                    help="Run-parameter string for the subtitle")
    args = ap.parse_args()

    if args.hepmc:
        path = args.hepmc
    elif args.cache_dir:
        cands = sorted(args.cache_dir.glob("*.hepmc"))
        if not cands:
            print(f"ERROR: no .hepmc in {args.cache_dir}", file=sys.stderr)
            sys.exit(1)
        path = cands[0]
        if len(cands) > 1:
            print(f"NOTE: {len(cands)} hepmc files in {args.cache_dir}, "
                  f"using {path.name}", file=sys.stderr)
    else:
        ap.error("pass --cache-dir or --hepmc")

    print(f"Reading {path}")
    E, px, py, pz = read_hepmc(path)
    print(f"  {len(E):,} stable particles")
    if len(E) == 0:
        print("ERROR: nothing to plot", file=sys.stderr); sys.exit(1)

    fig = make_figure(E, px, py, pz,
                      run_label=args.run_label,
                      params_label=args.params_label)
    fig.savefig(args.out, dpi=140, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
