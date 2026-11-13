#!/usr/bin/env python3
"""Collect every number and histogram of the v2 campaign into one JSON file.

    python collect_v2.py            -> $RUN/campaign_v2_results.json

For each signal sample it rebuilds the paired events (same 0x signal showers,
background-only hits added crystal by crystal), clusters them at the 0x
optimum found by the inner loop, and records for 0x / 1x / 3x:
  acceptance, energy scale, core width (clustered and unclustered sum),
  the paired change in width with a paired-bootstrap uncertainty, and how
  much background energy enters the cluster.
It also histograms the background-only deposits versus polar angle.
Figures are made from the JSON afterwards.
"""
import glob, json, os, sys
import numpy as np

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
B = "SCEPCal_MainEdep"
LABS = ("0x", "1x", "3x")
MDI = {0.0: "0x", 1.0: "1x", 3.0: "3x"}
NBOOT = 500
CONFIGS = [  # (base, particle, E_true [GeV], theta window [mrad] or None for barrel)
    ("g100MeV_fixed", "gamma", 0.1, None), ("g200MeV_fixed", "gamma", 0.2, None),
    ("g500MeV_fixed", "gamma", 0.5, None), ("g1000MeV_fixed", "gamma", 1.0, None),
    ("g5000MeV_fixed", "gamma", 5.0, None), ("e1GeV_fixed", "e-", 1.0, None),
] + [(f"g{E}MeV_th{a}-{b}", "gamma", E / 1000.0, (a, b))
     for E in (200, 1000) for a, b in ((140, 160), (160, 200), (200, 300), (300, 500))]


def width(a):
    q16, q84 = np.percentile(a, [16, 84])
    return 0.5 * (q84 - q16)


def mix(S, Bk, i):
    """Signal event i plus background event i (cyclic), summed per crystal."""
    cols = [np.asarray(S[k][i]) for k in range(5)]
    if Bk is not None:
        j = i % len(Bk[0])
        cols = [np.concatenate([cols[k], np.asarray(Bk[k][j])]) for k in range(5)]
    if len(cols[0]) == 0:
        z = np.zeros(0)
        return z, z, z, z
    u, first, inv = np.unique(cols[0].astype(np.uint64), return_index=True, return_inverse=True)
    e = np.bincount(inv.ravel(), weights=cols[1].astype(float), minlength=len(u))
    return (e, cols[2].astype(float)[first], cols[3].astype(float)[first],
            cols[4].astype(float)[first])


def cluster(e, x, y, R, Ethr):
    """Seeded radius clustering, identical to seeded_radius_cog."""
    if len(e) == 0:
        return np.nan
    s = int(np.argmax(e))
    sel = (((x - x[s]) ** 2 + (y - y[s]) ** 2) <= R * R) & (e >= Ethr)
    return float(e[sel].sum()) if np.any(sel) else np.nan


def analyse(S, BK, R, Ethr, Et, rng):
    n = len(S[0])
    Ec = {l: np.empty(n) for l in LABS}
    Raw = {l: np.empty(n) for l in LABS}
    for i in range(n):
        for l in LABS:
            e, x, y, _ = mix(S, BK[l], i)
            Ec[l][i] = cluster(e, x, y, R, Ethr)
            Raw[l][i] = e.sum()
    ok = np.isfinite(Ec["0x"]) & np.isfinite(Ec["1x"]) & np.isfinite(Ec["3x"]) & (Ec["0x"] > 0)
    idx = np.where(ok)[0]
    out = {"n_events": int(n), "n_paired": int(len(idx)), "R_mm": R, "Ethr_GeV": Ethr}
    if len(idx) < 50:
        out["error"] = "too few clustered events"
        return out
    boot = {l: np.empty(NBOOT) for l in LABS}
    for b in range(NBOOT):
        s = rng.choice(idx, len(idx))
        for l in LABS:
            boot[l][b] = width(Ec[l][s]) / Et
    bins = np.linspace(0.5, 1.5, 101)
    for l in LABS:
        c, r = Ec[l][idx], Raw[l][idx]
        d = {"acceptance": float(np.mean(np.isfinite(Ec[l]) & (Ec[l] > 0))),
             "scale": float(np.median(c) / Et),
             "width": float(width(c) / Et), "width_err": float(boot[l].std()),
             "raw_scale": float(np.median(r) / Et), "raw_width": float(width(r) / Et),
             "hist_cluster": np.histogram(c / Et, bins)[0].tolist(),
             "hist_raw": np.histogram(r / Et, bins)[0].tolist()}
        if l != "0x":
            dw = boot[l] - boot["0x"]
            de = (Ec[l][idx] - Ec["0x"][idx]) / Et
            d.update({"dwidth": float(d["width"] - width(Ec["0x"][idx]) / Et),
                      "dwidth_err": float(dw.std()),
                      "bkg_in_cluster_median": float(np.median(de)),
                      "bkg_in_cluster_mean": float(np.mean(de)),
                      "frac_events_touched": float(np.mean(de > 1e-6 / Et))})
        out[l] = d
    out["hist_edges"] = [0.5, 1.5, 100]
    return out


def main():
    sys.path.insert(0, RUN)
    import paired_mix as pm
    rng = np.random.default_rng(2026)
    res = {"configs": {}, "forward": {}}

    BK = {"0x": None}
    for l in ("1x", "3x"):
        d = f"{RUN}/sim_outputs/bkgonly_{l}"
        BK[l] = pm.read_calo(pm.ensure_merged(d), B)
        e, x, y, z = (np.concatenate([np.asarray(v, float) for v in BK[l][k]]) for k in (1, 2, 3, 4))
        th = np.arctan2(np.hypot(x, y), np.abs(z)) * 1e3
        edges = np.arange(0.0, 1580.0, 10.0)
        nev = len(BK[l][1])
        per_ev = np.array([np.asarray(v, float).sum() for v in BK[l][1]])
        res["forward"][l] = {
            "n_events": nev, "edges_mrad": [0.0, 1570.0, 10.0],
            "energy_MeV_per_event": (np.histogram(th, edges, weights=e)[0] / nev * 1e3).tolist(),
            "hits_per_event": (np.histogram(th, edges)[0] / nev).tolist(),
            "min_theta_mrad": float(th.min()), "hits_per_event_total": float(len(e) / nev),
            "energy_MeV_per_event_total": float(e.sum() / nev * 1e3),
            "energy_MeV_median": float(np.median(per_ev) * 1e3),
            "energy_MeV_p90": float(np.percentile(per_ev, 90) * 1e3)}
        print(f"[bkgonly {l}] events={nev} hits/evt={len(e)/nev:.0f} "
              f"E/evt mean={e.sum()/nev*1e3:.1f} MeV median={np.median(per_ev)*1e3:.1f} MeV "
              f"min theta={th.min():.0f} mrad")

    for base, part, Et, win in CONFIGS:
        sigdir = f"{RUN}/sim_outputs/{base}_0x"
        if not glob.glob(f"{sigdir}/events_*.root") and not os.path.exists(f"{sigdir}/merged.root"):
            print(f"[{base}] no signal sample, skipped"); continue
        opt = {}
        for f in glob.glob(f"{RUN}/money_v2_{base}/landscape_*.npz"):
            d = np.load(f)
            opt[MDI[round(float(d["geom_mdi_x"]), 3)]] = {
                "score": float(d["best_score"]), "R_mm": float(d["best_R_cluster_mm"]),
                "Ethr_GeV": float(d["best_E_threshold_GeV"]),
                "evals": int(d["opt_history_f"].shape[0])}
        if "0x" not in opt:
            print(f"[{base}] no inner-loop result yet, skipped"); continue
        S = pm.read_calo(pm.ensure_merged(sigdir), B)
        r = analyse(S, BK, opt["0x"]["R_mm"], opt["0x"]["Ethr_GeV"], Et, rng)
        r.update({"particle": part, "E_true_GeV": Et, "theta_window_mrad": win, "optima": opt})
        res["configs"][base] = r
        if "error" in r:
            print(f"[{base}] {r['error']} (paired {r['n_paired']}/{r['n_events']})"); continue
        line = (f"[{base}] R*={r['R_mm']:.0f} mm E*={r['Ethr_GeV']*1e3:.1f} MeV  "
                f"acc={r['0x']['acceptance']:.3f}  width0={r['0x']['width']:.3%}±{r['0x']['width_err']:.3%}  "
                f"raw width {r['0x']['raw_width']:.2%}/{r['1x']['raw_width']:.2%}/{r['3x']['raw_width']:.2%}")
        for l in ("1x", "3x"):
            d = r[l]
            line += (f"\n      {l}: dwidth={d['dwidth']:+.4%} ± {d['dwidth_err']:.4%} "
                     f"({d['dwidth']/max(d['dwidth_err'],1e-12):+.1f}σ)  scale={d['scale']:.3f}  "
                     f"bkg in cluster mean={d['bkg_in_cluster_mean']:+.3%} of E, "
                     f"{d['frac_events_touched']:.0%} of events touched")
        print(line)

    out = os.path.join(RUN, "campaign_v2_results.json")
    json.dump(res, open(out, "w"))
    print("\nwrote", out, f"({os.path.getsize(out)/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
