#!/usr/bin/env python3
"""Physics observables behind the money plot -- v2, truth-matched pairing.

v1 paired events by position in the hadd'ed file; the |dCOG| medians of
~3 m showed that ordering does NOT correspond between samples.  v2 pairs
events by the gun primary's truth momentum vector (MCParticles), which is
bitwise-identical between the 0x/1x/3x clones when the seeds matched.

Outputs bilevel_money_physics.{png,pdf} in RUN (overwrites v1).
"""

import glob, json, os, sys
import numpy as np

RUN    = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
BRANCH = "SCEPCal_MainEdep"
FIXED  = (50.0, 0.010)          # (R_cluster [mm], E_threshold [GeV])
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
                            Ethr=float(d["best_E_threshold_GeV"]),
                            score=float(d["best_score"]))
    return out


def find_samples():
    base = os.path.join(RUN, "sim_outputs")
    cand = {"0x": "e1GeV_fixed_0x", "1x": "e1GeV_fixed", "3x": "e1GeV_fixed_3x"}
    out = {}
    for lab, d in cand.items():
        path = os.path.join(base, d, "merged.root")
        if os.path.exists(path):
            out[lab] = path
        else:
            print(f"  ! missing sample: {path}")
    return out


def resolve_keys(t, stem, needles):
    """Find branch keys containing stem and each needle (canonical name first)."""
    keys = [k for k in t.keys() if stem in k and "contrib" not in k.lower()
            and not k.split("/")[-1].startswith("_")]
    out = []
    for n in needles:
        exact = f"{stem}.{n}"
        hit = None
        for k in keys:
            if k == exact or k.endswith("/" + exact):
                hit = k
                break
        if hit is None:
            for k in keys:
                if n.lower() in k.lower():
                    hit = k
                    break
        if hit is None:
            print(f"  available {stem} keys:", keys)
            raise RuntimeError(f"cannot resolve {stem}.{n}")
        out.append(hit)
    return out


def load_sample(path):
    """Returns (hits, truthkeys):
    hits      : list over events of (E, x, y) numpy arrays
    truthkeys : list over events of a hashable gun-primary momentum key"""
    t = uproot.open(path + ":events")

    kE, kx, ky = resolve_keys(t, BRANCH, ["energy", "position.x", "position.y"])
    arr = t.arrays([kE, kx, ky], library="np")
    E, X, Y = arr[kE], arr[kx], arr[ky]
    hits = [(np.asarray(E[i], float), np.asarray(X[i], float),
             np.asarray(Y[i], float)) for i in range(len(E))]

    kpx, kpy, kpz = resolve_keys(t, "MCParticles",
                                 ["momentum.x", "momentum.y", "momentum.z"])
    m = t.arrays([kpx, kpy, kpz], library="np")
    tk = []
    for i in range(len(m[kpx])):
        px = np.asarray(m[kpx][i], float)
        py = np.asarray(m[kpy][i], float)
        pz = np.asarray(m[kpz][i], float)
        if len(px) == 0:
            tk.append(None)
        else:  # gun primary = first MCParticle; float64 rounded for hashing
            tk.append((round(float(px[0]), 9), round(float(py[0]), 9),
                       round(float(pz[0]), 9)))
    return hits, tk


def cluster(ev, R, Ethr):
    E, x, y = ev
    if len(E) == 0:
        return None
    s = int(np.argmax(E))
    dx, dy = x - x[s], y - y[s]
    sel = ((dx * dx + dy * dy) <= R * R) & (E >= Ethr)
    if not np.any(sel):
        return None
    Ec = float(E[sel].sum())
    if Ec <= 0:
        return None
    return (Ec, int(sel.sum()),
            float((E[sel] * x[sel]).sum() / Ec),
            float((E[sel] * y[sel]).sum() / Ec))


def summarize(events, R, Ethr):
    clus = [cluster(ev, R, Ethr) for ev in events]
    Es = np.array([c[0] for c in clus if c is not None])
    acc = len(Es) / max(1, len(clus))
    if len(Es) == 0:
        return acc, np.nan, np.nan, clus
    mu = float(np.median(Es))
    q16, q84 = np.percentile(Es, [16, 84])
    return acc, mu, (0.5 * (q84 - q16) / mu if mu > 0 else np.nan), clus


def main():
    optima = load_optima()
    samples = find_samples()
    labs = [l for l in ("0x", "1x", "3x") if l in samples]
    if "0x" not in labs:
        sys.exit("need the 0x sample")

    data = {}
    for l in labs:
        hits, tk = load_sample(samples[l])
        data[l] = dict(hits=hits, tk=tk)
        print(f"{l}: {len(hits)} events   "
              f"first truth keys: {tk[:2]}")

    # ---- pairing diagnostics: order vs truth ------------------------------
    tk0 = data["0x"]["tk"]
    idx0 = {}
    for i, k in enumerate(tk0):
        if k is not None:
            idx0.setdefault(k, i)   # first occurrence wins
    dup0 = len(tk0) - len(idx0)
    if dup0:
        print(f"  note: {dup0} duplicate truth keys in 0x (kept first)")

    pairs = {}
    for l in [x for x in labs if x != "0x"]:
        tkl = data[l]["tk"]
        n = min(len(tk0), len(tkl))
        order_match = sum(1 for i in range(n) if tkl[i] == tk0[i]) / max(1, n)
        matched = [(idx0[k], j) for j, k in enumerate(tkl) if k in idx0]
        print(f"[pairing {l} vs 0x] order-aligned: {order_match:.1%}   "
              f"truth-matched: {len(matched)}/{len(tkl)} "
              f"({len(matched)/max(1,len(tkl)):.1%})")
        pairs[l] = matched

    # ---- observables at each intensity's own optimum ----------------------
    rows = {}
    for l in labs:
        R, Et = (optima[l]["R"], optima[l]["Ethr"]) if l in optima else FIXED
        acc, mu, res, _ = summarize(data[l]["hits"], R, Et)
        rows[l] = dict(R=R, Ethr=Et, acc=acc, mu=mu, res=res)
        print(f"[optimum {l}] R={R:.1f} mm  E_thr={Et:.3f} GeV  "
              f"acc={acc:.3f}  mu={mu:.4f} GeV  core sigma/mu={res:.4%}")

    # ---- fixed working point ----------------------------------------------
    Rf, Ef = FIXED
    fixed = {l: summarize(data[l]["hits"], Rf, Ef) for l in labs}
    for l in labs:
        a, m, r, _ = fixed[l]
        print(f"[fixed   {l}] R={Rf:.1f} mm  E_thr={Ef:.3f} GeV  "
              f"acc={a:.3f}  mu={m:.4f} GeV  core sigma/mu={r:.4%}")

    # ---- truth-matched paired comparisons ---------------------------------
    c0 = fixed["0x"][3]
    dcog, dee = {}, {}
    for l, matched in pairs.items():
        cl = fixed[l][3]
        dc, de = [], []
        for i0, il in matched:
            if c0[i0] is None or cl[il] is None:
                continue
            dc.append(np.hypot(cl[il][2] - c0[i0][2], cl[il][3] - c0[i0][3]))
            de.append((cl[il][0] - c0[i0][0]) / c0[i0][0])
        dcog[l], dee[l] = np.array(dc), np.array(de)
        if len(dc):
            print(f"[paired {l} vs 0x] N={len(dc)}  "
                  f"median |dCOG|={np.median(dc):.3f} mm  "
                  f"q95={np.percentile(dc,95):.3f} mm  "
                  f"median dE/E={np.median(de):+.4%}  "
                  f"core spread dE/E={0.5*(np.percentile(de,84)-np.percentile(de,16)):.4%}")
        else:
            print(f"[paired {l} vs 0x] NO truth-matched pairs -- samples "
                  f"were not generated with matching seeds")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.5))
    fig.suptitle("Physics observables behind the money plot "
                 "(1 GeV e$^-$ gun, SCEPCal MainEdep, truth-matched pairs)",
                 fontsize=13)
    xs = np.arange(len(labs))

    a = ax[0, 0]
    a.bar(xs - 0.18, [rows[l]["acc"] for l in labs], 0.34,
          color=[COLORS[l] for l in labs], label="at own optimum ($R^*$, $E^*_{thr}$)")
    a.bar(xs + 0.18, [fixed[l][0] for l in labs], 0.34,
          color=[COLORS[l] for l in labs], alpha=0.45,
          label=f"at fixed ({Rf:.0f} mm, {Ef*1e3:.0f} MeV)")
    for i, l in enumerate(labs):
        a.text(i - 0.18, rows[l]["acc"] + 0.01, f"{rows[l]['acc']:.3f}",
               ha="center", fontsize=7)
        a.text(i + 0.18, fixed[l][0] + 0.01, f"{fixed[l][0]:.3f}",
               ha="center", fontsize=7)
    a.set_xticks(xs); a.set_xticklabels(labs)
    a.set_ylim(0, 1.09); a.set_ylabel("acceptance")
    a.set_title("Cluster acceptance"); a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")

    a = ax[0, 1]
    a2 = a.twinx()
    r_opt = [rows[l]["res"] * 100 for l in labs]
    r_fix = [fixed[l][2] * 100 for l in labs]
    a.plot(xs, r_opt, "o-", color="#0B5F6B", lw=2, label="core $\\sigma/\\mu$ (own optimum)")
    a.plot(xs, r_fix, "o--", color="#0B5F6B", alpha=0.5, label="core $\\sigma/\\mu$ (fixed WP)")
    for i in range(len(labs)):
        a.annotate(f"{r_opt[i]:.2f}%", (xs[i], r_opt[i]),
                   textcoords="offset points", xytext=(0, 7), fontsize=7,
                   ha="center", color="#0B5F6B")
        a.annotate(f"{r_fix[i]:.1f}%", (xs[i], r_fix[i]),
                   textcoords="offset points", xytext=(0, 7), fontsize=7,
                   ha="center", color="#0B5F6B", alpha=0.7)
    a2.plot(xs, [rows[l]["mu"] for l in labs], "s-", color="#C98500", lw=2,
            label="median $E_{clus}$ (own optimum)")
    a.set_yscale("symlog", linthresh=1.0)
    a.set_xticks(xs); a.set_xticklabels(labs)
    a.set_ylabel("core $\\sigma/\\mu$  [%]", color="#0B5F6B")
    a2.set_ylabel("median $E_{clus}$  [GeV]", color="#C98500")
    a.set_title("Energy resolution and scale")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left"); a.grid(alpha=0.3)

    a = ax[1, 0]
    for l, arr in dcog.items():
        if len(arr):
            a.hist(arr, bins=60, histtype="step", lw=2, color=COLORS[l],
                   label=f"{l} vs 0x  (med {np.median(arr):.2f} mm, "
                         f"q95 {np.percentile(arr,95):.1f} mm)")
    a.set_yscale("log")
    a.set_xlabel("|$\\Delta$COG| vs 0x  [mm]  (truth-matched pairs, fixed WP)")
    a.set_ylabel("events")
    a.set_title("Does MDI move the impact point?")
    if dcog and any(len(v) for v in dcog.values()):
        a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "no truth-matched pairs", ha="center", va="center",
               transform=a.transAxes)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    for l, arr in dee.items():
        if len(arr):
            a.hist(arr * 100, bins=60, histtype="step", lw=2, color=COLORS[l],
                   label=f"{l} vs 0x  (med {np.median(arr)*100:+.2f}%)")
    a.axvline(0, color="k", lw=0.8, alpha=0.5)
    a.set_yscale("log")
    a.set_xlabel("$\\Delta E_{clus}/E_{clus}^{0x}$  [%]  (truth-matched pairs, fixed WP)")
    a.set_ylabel("events")
    a.set_title("Does MDI bias the measured energy?")
    if dee and any(len(v) for v in dee.values()):
        a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RUN, f"bilevel_money_physics.{ext}"), dpi=170)
    print("wrote", os.path.join(RUN, "bilevel_money_physics.png"))


if __name__ == "__main__":
    main()
