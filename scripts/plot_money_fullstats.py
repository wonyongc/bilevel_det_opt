#!/usr/bin/env python3
"""Full-statistics physics distributions at the fixed working point.

All 2000 events per intensity (no event pairing needed):
  P1: |COG - seed| distance    -- background pull on the reconstructed position
  P2: cluster size n_hits      -- occupancy inside the cluster
  P3: cluster energy E_clus    -- scale shift and tails
  P4: seed-crystal energy      -- with the three DE optima E*_thr marked,
                                  showing the acceptance-degenerate tail

Fixed WP: R = 50 mm, E_thr = 10 MeV.  Outputs
bilevel_money_physics_fullstats.{png,pdf} in RUN.
"""

import glob, os, sys
import numpy as np

RUN    = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
BRANCH = "SCEPCal_MainEdep"
FIXED  = (50.0, 0.010)
LABELS = {0.0: "0x", 1.0: "1x", 3.0: "3x"}
COLORS = {"0x": "#666666", "1x": "#1f77b4", "3x": "#c62828"}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot


def load_optima():
    out = {}
    for f in sorted(glob.glob(os.path.join(RUN, "money", "landscape_*.npz"))):
        d = np.load(f)
        lab = LABELS.get(round(float(d["geom_mdi_x"]), 3))
        if lab:
            out[lab] = dict(R=float(d["best_R_cluster_mm"]),
                            Ethr=float(d["best_E_threshold_GeV"]))
    return out


def find_samples():
    base = os.path.join(RUN, "sim_outputs")
    cand = {"0x": "e1GeV_fixed_0x", "1x": "e1GeV_fixed", "3x": "e1GeV_fixed_3x"}
    out = {}
    for lab, d in cand.items():
        p = os.path.join(base, d, "merged.root")
        if os.path.exists(p):
            out[lab] = p
        else:
            print(f"  ! missing sample: {p}")
    return out


def load_hits(path):
    t = uproot.open(path + ":events")
    keys = [k for k in t.keys() if BRANCH in k and "contrib" not in k.lower()
            and not k.split("/")[-1].startswith("_")]

    def pick(needle):
        exact = f"{BRANCH}.{needle}"
        for k in keys:
            if k == exact or k.endswith("/" + exact):
                return k
        for k in keys:
            if needle.lower() in k.lower():
                return k
        raise RuntimeError(f"cannot resolve {BRANCH}.{needle}; keys={keys}")

    kE, kx, ky = pick("energy"), pick("position.x"), pick("position.y")
    arr = t.arrays([kE, kx, ky], library="np")
    E, X, Y = arr[kE], arr[kx], arr[ky]
    return [(np.asarray(E[i], float), np.asarray(X[i], float),
             np.asarray(Y[i], float)) for i in range(len(E))]


def analyze(events, R, Ethr):
    """Per-event at the WP: (E_clus, n, |COG-seed|, E_seed); NaN-padded."""
    Ec, N, D, S = [], [], [], []
    for E, x, y in events:
        if len(E) == 0:
            Ec.append(np.nan); N.append(np.nan); D.append(np.nan); S.append(np.nan)
            continue
        s = int(np.argmax(E))
        S.append(float(E[s]))
        dx, dy = x - x[s], y - y[s]
        sel = ((dx * dx + dy * dy) <= R * R) & (E >= Ethr)
        if not np.any(sel):
            Ec.append(np.nan); N.append(np.nan); D.append(np.nan)
            continue
        e = float(E[sel].sum())
        cx = float((E[sel] * x[sel]).sum() / e)
        cy = float((E[sel] * y[sel]).sum() / e)
        Ec.append(e); N.append(int(sel.sum()))
        D.append(float(np.hypot(cx - x[s], cy - y[s])))
    return (np.array(Ec), np.array(N), np.array(D), np.array(S))


def main():
    optima = load_optima()
    samples = find_samples()
    labs = [l for l in ("0x", "1x", "3x") if l in samples]
    Rf, Ef = FIXED

    res = {}
    for l in labs:
        ev = load_hits(samples[l])
        res[l] = analyze(ev, Rf, Ef)
        Ec, N, D, S = res[l]
        ok = np.isfinite(Ec)
        print(f"[{l}] N={len(Ec)}  acc={ok.mean():.3f}  "
              f"median E_clus={np.nanmedian(Ec):.4f} GeV  "
              f"median n_hits={np.nanmedian(N):.0f}  "
              f"median |COG-seed|={np.nanmedian(D):.3f} mm  "
              f"q95={np.nanpercentile(D,95):.3f} mm  "
              f"median E_seed={np.nanmedian(S):.4f} GeV")
        if l in optima:
            frac = np.nanmean(S >= optima[l]["Ethr"])
            print(f"     fraction of events with E_seed >= E*_thr"
                  f"({optima[l]['Ethr']:.3f} GeV): {frac:.4f}")

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.5))
    fig.suptitle("MDI physics observables, full statistics "
                 f"(1 GeV e$^-$ gun, fixed WP: R={Rf:.0f} mm, "
                 f"E$_{{thr}}$={Ef*1e3:.0f} MeV)", fontsize=13)

    a = ax[0, 0]
    for l in labs:
        D = res[l][2]
        D = D[np.isfinite(D)]
        a.hist(D, bins=np.linspace(0, 50, 51), histtype="step", lw=2,
               color=COLORS[l],
               label=f"{l}  (med {np.median(D):.2f} mm, q95 {np.percentile(D,95):.1f} mm)")
    a.set_yscale("log")
    a.set_xlabel("|COG $-$ seed|  [mm]")
    a.set_ylabel("events")
    a.set_title("Background pull on the reconstructed position")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    nmax = max(np.nanmax(res[l][1]) for l in labs)
    bins = np.arange(-0.5, min(nmax, 400) + 1.5, max(1, int(nmax // 80)))
    for l in labs:
        N = res[l][1]
        N = N[np.isfinite(N)]
        a.hist(N, bins=bins, histtype="step", lw=2, color=COLORS[l],
               label=f"{l}  (med {np.median(N):.0f} hits)")
    a.set_yscale("log")
    a.set_xlabel("cluster size $n_{hits}$")
    a.set_ylabel("events")
    a.set_title("Occupancy inside the cluster")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    for l in labs:
        Ec = res[l][0]
        Ec = Ec[np.isfinite(Ec)]
        a.hist(Ec, bins=np.linspace(0, 1.2, 61), histtype="step", lw=2,
               color=COLORS[l], label=f"{l}  (med {np.median(Ec):.4f} GeV)")
    a.set_yscale("log")
    a.set_xlabel("$E_{clus}$  [GeV]")
    a.set_ylabel("events")
    a.set_title("Cluster energy: scale shift and tails")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    for l in labs:
        S = res[l][3]
        S = S[np.isfinite(S)]
        a.hist(S, bins=np.linspace(0, 0.5, 76), histtype="step", lw=2,
               color=COLORS[l], label=f"{l}  seed energy")
        if l in optima:
            a.axvline(optima[l]["Ethr"], color=COLORS[l], ls="--", lw=1.5,
                      alpha=0.8)
            a.text(optima[l]["Ethr"], a.get_ylim()[1] * 0.5,
                   f" $E^*_{{thr}}$({l})", rotation=90, va="top", fontsize=7,
                   color=COLORS[l])
    a.set_yscale("log")
    a.set_xlabel("seed-crystal energy $E_{seed}$  [GeV]")
    a.set_ylabel("events")
    a.set_title("Why the optimum is degenerate: $E^*_{thr}$ sits in the tail")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RUN, f"bilevel_money_physics_fullstats.{ext}"),
                    dpi=170)
    print("wrote", os.path.join(RUN, "bilevel_money_physics_fullstats.png"))


if __name__ == "__main__":
    main()
