#!/usr/bin/env python3
"""Collection of campaign v4 (full detectors), independent of collect_v2.py.

    DET=CLD python fulldet/collect_fulldet.py [--pts Z WW ZH tt] [--nboot 300] [--jobs 32]
        -> $RUN/campaign_v4_<det>_results.json and a printed table

For every sample money_v4_<det>_<pt>_<base>/ it re-evaluates the clustering on the mixed events of
each intensity (0x, 1x, 3x, the mix specs the loop wrote) at a set of candidate settings: the DE
optimum and the 12 x 25 landscape-grid optimum of EVERY intensity. Then
  re-tuned  = the best candidate for that intensity (this repairs DE runs that stopped early on a
              flat landscape, e.g. CLD 100 MeV at 0x),
  kept      = the 0x re-tuned setting applied unchanged at 1x and 3x,
and the resolution sigma_core / min(mu, E_cal) (the eres_cal score's inverse; zeros kept) with a
PAIRED bootstrap over signal events: the same resampled event indices at 0x, 1x and 3x (paired
mixing gives every intensity the same showers), so the uncertainty of the background term
c = E_true * sqrt(s_m^2 - s_0^2) is that of the difference, not of the two widths. c is reported
with its bootstrap 16/50/84 % quantiles (negative s_m^2 - s_0^2 enters as 0).
Environment: key4hep, then the bilevel venv (two separate `source` lines). Run as a batch job
(fulldet/collect_fulldet.sbatch). Each sample is saved to money_v4_.../collect_fulldet.json as soon as
it is done and skipped by a later job (--force recomputes), so a job that hits its time limit can
simply be resubmitted. CLD takes ~1 h on one node; ALLEGRO more (its 1.5 m endcap clusters).

VERSION 2 (2026-11-11). The v1 measure sigma_core / min(mu, E_cal) can IMPROVE with background: a small
cluster that misses part of the shower has mu < E_cal, and the average background energy added to it
raises mu and so lowers the ratio (IDEA, 0.2 GeV, Z 1x: 2.61 % re-tuned against 2.72 % without
background, c = 0). v2 uses sigma_core / E_cal with the fixed E_cal of the 0x calibration, i.e. the
absolute spread, for picking the re-tuned setting AND for c: an average offset cannot lower it, and c is
the added fluctuation, the usual meaning of a noise term. The offset is reported separately
(shift_MeV = E_true * (mu_m - mu_0) / E_cal, photon-equivalent). The v1 numbers are recomputed and kept
in "retuned_v1" / "kept_v1". Per-sample caches carry "version": 2 (older ones are recomputed); the
result goes to campaign_v4_<det>_results_v2.json, so the v1 file stays as it was.
"""
import argparse, gc, glob, json, os, re, sys
import numpy as np

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
BASES = ["g100MeV_fixed", "g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed", "g5000MeV_fixed", "e1GeV_fixed",
         "g200MeV_th140-160", "g200MeV_th160-200", "g200MeV_th200-300", "g200MeV_th300-500",
         "g1000MeV_th140-160", "g1000MeV_th160-200", "g1000MeV_th200-300", "g1000MeV_th300-500"]
MIX = (0.0, 1.0, 3.0)
LAB = {0.0: "0x", 1.0: "1x", 3.0: "3x"}


def etrue_gev(base):
    m = re.match(r"[ge](\d+)(MeV|GeV)_", base)
    return float(m.group(1)) * (1e-3 if m.group(2) == "MeV" else 1.0)


def core(a):
    q16, q84 = np.percentile(a, [16.0, 84.0])
    return 0.5 * (q84 - q16)


VERSION = 2


def resol_abs(E, ecal):
    """v2: sigma_core / E_cal over all events (zeros kept): the absolute spread in units of E_cal."""
    s = core(E)
    return s / ecal if s > 0 else np.inf


def resol(E, ecal):
    """sigma_core / min(mu, E_cal) over all events (zeros kept) = 1 / eres_cal score."""
    mu = float(np.median(E)); s = core(E)
    if mu <= 0 or s <= 0:
        return np.inf
    return s / min(mu, ecal)


def lean_mixed_events(paired_mix, sig_path, bkg_path, bname, max_events=None):
    """Same mixing as paired_mix.mixed_events (signal event i + background event i % nb, energies
    summed per cellID, position of the first occurrence, hits ordered by cellID), but each event is
    kept as four numpy arrays instead of one Python object per hit. The clustering only reads
    .E, .x, .y, .z, so the result is identical and the memory is ~50 times smaller (ALLEGRO at
    three ttbar crossings has ~40,000 hits per event)."""
    import types
    S = paired_mix.read_calo(sig_path, bname)
    B = paired_mix.read_calo(bkg_path, bname) if bkg_path else None
    n = len(S[0]) if not max_events else min(len(S[0]), max_events)
    nb = len(B[0]) if B is not None else 0
    out = []
    for i in range(n):
        cols = [np.asarray(S[k][i]) for k in range(5)]
        if B is not None:
            j = i % nb
            cols = [np.concatenate([cols[k], np.asarray(B[k][j])]) for k in range(5)]
        cid = cols[0].astype(np.uint64)
        e = cols[1].astype(float)
        u, first, inv = np.unique(cid, return_index=True, return_inverse=True)
        esum = np.bincount(inv.ravel(), weights=e, minlength=len(u))
        out.append({bname: types.SimpleNamespace(cellID=u, E=esum, x=cols[2].astype(float)[first],
                                                 y=cols[3].astype(float)[first], z=cols[4].astype(float)[first])})
    return out


def selfcheck(tag, nev=150):
    """Lean mixing against paired_mix.mixed_events on the first events of one sample, all intensities."""
    sys.path.insert(0, RUN); sys.path.insert(0, os.path.expanduser("~/bilevel_det_opt"))
    import paired_mix
    from bilevel_opt.algorithms.seeded_radius_cog import SeededRadiusCoG
    d = os.path.join(RUN, tag)
    cfg = json.load(open(os.path.join(d, "money_results_0x_1x_3x.json")))["il_cfg"]
    br = cfg["branches"][0]["name"]; algo = SeededRadiusCoG(**dict(cfg["algorithm"].get("params", {})))
    worst = 0.0
    for m in MIX:
        spec = json.load(open(os.path.join(d, f"mix_{LAB[m]}.mixspec.json")))
        a = paired_mix.mixed_events(spec["signal"], spec["bkg"], br, nev)
        b = lean_mixed_events(paired_mix, spec["signal"], spec["bkg"], br, nev)
        for p in ({"R_cluster_mm": 30.0, "E_threshold_GeV": 0.0}, {"R_cluster_mm": 150.0, "E_threshold_GeV": 1e-4}):
            ea = np.array([r.E_cluster for r in algo.reconstruct(a, p)]); eb = np.array([r.E_cluster for r in algo.reconstruct(b, p)])
            worst = max(worst, float(np.max(np.abs(ea - eb))))
        print(f"[selfcheck] {tag} {LAB[m]}: {len(a)} events, hits/event {np.median([len(x[br].E) for x in a]):.0f} vs "
              f"{np.median([len(x[br].E) for x in b]):.0f}, max |dE_cluster| so far {worst:.3g} GeV", flush=True)
    print("[selfcheck] IDENTICAL" if worst < 1e-12 else "[selfcheck] DIFFERENT - do not use the lean mixing")


def one_sample(args):
    tag, nboot, seed = args
    cache = os.path.join(RUN, tag, "collect_fulldet.json")
    if os.path.exists(cache):                     # finished in an earlier (possibly timed-out) job
        old = json.load(open(cache))
        if old.get("version") == VERSION:
            return old
    sys.path.insert(0, RUN); sys.path.insert(0, os.path.expanduser("~/bilevel_det_opt"))
    import paired_mix
    from bilevel_opt.algorithms.seeded_radius_cog import SeededRadiusCoG
    d = os.path.join(RUN, tag)
    cfg = json.load(open(os.path.join(d, "money_results_0x_1x_3x.json")))["il_cfg"]
    br = cfg["branches"][0]["name"]
    ap_ = dict(cfg["algorithm"].get("params", {}))
    algo = SeededRadiusCoG(**ap_)
    ecal = float(cfg["score"]["params"]["E_true"])
    pnames = [p["name"] for p in cfg["algorithm"]["optimize_params"]]
    summ = {x["geom_values"]["mdi_x"]: x for x in json.load(open(os.path.join(d, "summary.json")))["per_geometry"]}
    # candidates: DE optimum and grid optimum of every intensity
    cands = []
    for f in sorted(glob.glob(os.path.join(d, "landscape_*.npz"))):
        z = np.load(f, allow_pickle=True)
        m = float(z["geom_mdi_x"])
        cands.append(("de", m, {n: float(z[f"best_{n}"]) for n in pnames}))
        if "score" in z.files:
            g = np.asarray(z["score"]); idx = np.unravel_index(int(np.nanargmax(g)), g.shape)
            cands.append(("grid", m, {n: float(z[f"grid_{n}"][idx[k]]) for k, n in enumerate(pnames)}))
    # events and cluster energies for every (intensity, candidate)
    E = {}
    for m in MIX:
        spec = json.load(open(os.path.join(d, f"mix_{LAB[m]}.mixspec.json")))
        ev = lean_mixed_events(paired_mix, spec["signal"], spec["bkg"], br, cfg.get("max_events"))
        for k, (_, _, p) in enumerate(cands):
            reco = algo.reconstruct(ev, p)
            E[(m, k)] = np.array([r.E_cluster if (r.n_hits > 0 and r.E_cluster > 0) else 0.0 for r in reco])
        del ev, reco
        gc.collect()
    n = len(E[(0.0, 0)])
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, n, n) for _ in range(nboot)]
    et = etrue_gev(tag.split("_", 4)[4])
    out = {"tag": tag, "version": VERSION, "E_true_GeV": et, "E_cal_GeV": ecal, "branch": br, "n_events": n,
           "algo": ap_, "candidates": [{"from": c[0], "mdi_x": c[1], **c[2]} for c in cands]}
    for meas, suffix in ((resol_abs, ""), (resol, "_v1")):
        best = {m: min(range(len(cands)), key=lambda k: meas(E[(m, k)], ecal)) for m in MIX}
        for mode in ("retuned", "kept"):
            pick = {m: best[m] for m in MIX} if mode == "retuned" else {m: best[0.0] for m in MIX}
            r = {m: meas(E[(m, pick[m])], ecal) for m in MIX}
            rb = {m: np.array([meas(E[(m, pick[m])][b], ecal) for b in boots]) for m in MIX}
            mu = {m: float(np.median(E[(m, pick[m])])) for m in MIX}
            res = {}
            for m in MIX:
                q = np.percentile(rb[m], [16, 50, 84])
                entry = {"res": r[m], "res_q16": q[0], "res_q84": q[2], "median_GeV": mu[m],
                         **cands[pick[m]][2], "from": f"{cands[pick[m]][0]}@{cands[pick[m]][1]:.0f}x"}
                if m > 0:
                    cb = 1e3 * et * np.sqrt(np.clip(rb[m] ** 2 - rb[0.0] ** 2, 0, None))
                    entry["c_MeV"] = 1e3 * et * float(np.sqrt(max(0.0, r[m] ** 2 - r[0.0] ** 2)))
                    entry["c_q16"], entry["c_q50"], entry["c_q84"] = (float(v) for v in np.percentile(cb, [16, 50, 84]))
                    entry["shift_MeV"] = 1e3 * et * (mu[m] - mu[0.0]) / ecal
                res[LAB[m]] = entry
            out[mode + suffix] = res
    out = json.loads(json.dumps(out, default=float))   # numpy -> plain floats
    json.dump(out, open(cache + ".tmp", "w"), indent=1); os.replace(cache + ".tmp", cache)
    print(f"[collect] done {tag}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", nargs="+", default=["Z", "WW", "ZH", "tt"])
    ap.add_argument("--nboot", type=int, default=300); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="", help="substring filter on the sample tag (testing)")
    ap.add_argument("--force", action="store_true", help="ignore per-sample results of earlier jobs")
    ap.add_argument("--selfcheck", default="", help="money_v4_... tag: compare the lean mixing with paired_mix.mixed_events and exit")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck(a.selfcheck); return
    DET = os.environ.get("DET", "")
    if DET not in ("CLD", "ALLEGRO", "IDEA"):
        sys.exit("usage: DET=CLD|ALLEGRO|IDEA python fulldet/collect_fulldet.py")
    dtag = DET.lower()
    tags = [f"money_v4_{dtag}_{pt}_{b}" for pt in a.pts for b in BASES]
    tags = [t for t in tags if a.only in t]
    have = [t for t in tags if os.path.exists(os.path.join(RUN, t, "summary.json"))]
    for t in sorted(set(tags) - set(have)):
        print("missing", t)
    work = [(t, a.nboot, 1000 + i) for i, t in enumerate(have)]
    if a.force:
        for t, _, _ in work:
            c = os.path.join(RUN, t, "collect_fulldet.json")
            if os.path.exists(c): os.remove(c)
    ndone = sum(os.path.exists(os.path.join(RUN, t, "collect_fulldet.json")) for t, _, _ in work)
    print(f"[collect] {len(work)} samples, {ndone} already collected (per-sample collect_fulldet.json)", flush=True)
    # slowest first (ALLEGRO endcap, 3x-heavy energies) so the pool does not end on one long task
    work.sort(key=lambda w: ("_th" not in w[0], w[0]))
    if a.jobs > 1:
        # ProcessPoolExecutor raises if a worker is killed (e.g. out of memory); multiprocessing.Pool waits forever
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        res = []
        try:
            with ProcessPoolExecutor(a.jobs) as pool:
                for fut in as_completed([pool.submit(one_sample, w) for w in work]):
                    res.append(fut.result())
        except BrokenProcessPool:
            n = sum(os.path.exists(os.path.join(RUN, t, "collect_fulldet.json")) for t, _, _ in work)
            sys.exit(f"[collect] a worker was killed (memory?). {n} of {len(work)} samples are saved; "
                     f"resubmit with fewer --jobs to finish the rest.")
    else:
        res = [one_sample(w) for w in work]
    order = {t: i for i, t in enumerate(tags)}
    res.sort(key=lambda r: order[r["tag"]])
    path = os.path.join(RUN, f"campaign_v4_{dtag}_results_v2.json")
    json.dump({"detector": DET, "version": VERSION, "samples": {r["tag"]: r for r in res}}, open(path, "w"), indent=1)
    print(f"\n== {DET} (v2): sigma_core/E_cal [%] 0x/1x/3x re-tuned ; c [MeV] at 1x, 3x (bootstrap 16-84 %) ;"
          f" offset [MeV] at 3x ; kept ; v1 c at 1x, 3x")
    for r in res:
        R = r["retuned"]
        s = "/".join(f"{100*R[l]['res']:4.1f}" for l in ("0x", "1x", "3x"))
        c = "  ".join(f"{R[l]['c_MeV']:4.0f} [{R[l]['c_q16']:3.0f},{R[l]['c_q84']:3.0f}]" for l in ("1x", "3x"))
        k = "/".join(f"{100*r['kept'][l]['res']:4.1f}" for l in ("0x", "1x", "3x"))
        v1 = "/".join(f"{r['retuned_v1'][l]['c_MeV']:.0f}" for l in ("1x", "3x"))
        print(f"{r['tag'][9:]:34} {s}   c {c}   off {R['3x']['shift_MeV']:5.1f}   kept {k}   v1 c {v1}")
    print("wrote", path)


if __name__ == "__main__":
    main()
