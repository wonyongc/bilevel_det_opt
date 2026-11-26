#!/usr/bin/env python3
"""Summarize one WarpX beam-beam run: luminosity, beamstrahlung, pairs.

    python check_yields.py Z        # directory holding inputs, deck_meta.json, diags/

Everything is WEIGHTED: one macroparticle stands for `weighting` real
particles, so all numbers are per bunch crossing.  Reads the boundary-scraped
particles on all six faces plus what was still in the box at the last step.
Writes <dir>/yields.json and prints a table.
"""
import glob, json, os, sys
import numpy as np
import openpmd_api as io

C, ME_GEV, GEV = 299792458.0, 0.51099895e-3, 1.602176634e-10
FACES = ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi"]


def series(pattern_dir):
    f = sorted(glob.glob(os.path.join(pattern_dir, "openpmd_*")))
    if not f:
        return None
    ext = f[0].split(".")[-1]
    return io.Series(os.path.join(pattern_dir, "openpmd_%T." + ext), io.Access.read_only)


def load(run):
    """-> {species: dict(px,py,pz [GeV/c], w, x,y,z [m], face)} merged over faces and iterations."""
    out = {}

    def add(s, face, last_only=False):
        its = list(s.iterations)
        if last_only:
            its = its[-1:]
        for k in its:
            it = s.iterations[k]
            for sp in it.particles:
                p = it.particles[sp]
                w = p["weighting"][io.Record_Component.SCALAR]
                if w.shape[0] == 0:
                    continue
                arr = {"w": w.load_chunk()}
                for c in "xyz":
                    arr["p" + c] = p["momentum"][c].load_chunk()
                    arr[c] = p["position"][c].load_chunk()
                s.flush()
                d = out.setdefault(sp, {k2: [] for k2 in ("px", "py", "pz", "w", "x", "y", "z", "face")})
                for c in "xyz":
                    d["p" + c].append(np.asarray(arr["p" + c]) * p["momentum"][c].unit_SI * C / GEV)
                    d[c].append(np.asarray(arr[c]) * p["position"][c].unit_SI)
                d["w"].append(np.asarray(arr["w"]))
                d["face"].append(np.full(len(arr["w"]), face))

    for i, face in enumerate(FACES):
        s = series(os.path.join(run, "diags", "scrape1", "particles_at_" + face))
        if s is not None:
            add(s, i)
    s = series(os.path.join(run, "diags", "diag1"))
    if s is not None:
        add(s, 6, last_only=True)                       # leftovers inside the box
    return {k: {a: np.concatenate(v) for a, v in d.items()} for k, d in out.items()}


def wq(x, w, q):
    i = np.argsort(x); cw = np.cumsum(w[i]); return float(np.interp(q * cw[-1], cw, x[i]))


def main(run):
    meta = json.load(open(os.path.join(run, "deck_meta.json")))
    N, Eb, th = meta["N"], meta["ebeam"], meta["th"]
    res = {"point": meta["name"], "preset": meta["preset"]}

    # ---- luminosity and beam state from the reduced diagnostics ----
    d = np.loadtxt(os.path.join(run, "diags", "reducedfiles", "lumi.txt"))
    lum = float(np.sum(0.5 * (d[1:, 2] + d[:-1, 2]) * np.diff(d[:, 1])))
    ipk = int(np.argmax(d[:, 2]))
    res["luminosity_per_crossing_m2"] = lum
    res["luminosity_fsr_m2"] = meta["lumi_bx"]
    res["chi_avg_at_peak"], res["chi_max"] = float(d[ipk, 4]), float(d[:, 5].max())
    res["sigma_y_at_peak_nm"] = float(d[ipk, 9] * 1e9)
    print(f"\n=== {meta['name']}  sqrt(s) = {meta['ecm']} GeV  (preset {meta['preset']}) ===")
    print(f"luminosity per crossing : {lum:.3e} m^-2   FSR {meta['lumi_bx']:.3e}   ratio {lum/meta['lumi_bx']:.2f}")
    print(f"beam chi at peak        : average {d[ipk,4]:.2e}, maximum over run {d[:,5].max():.2e}")
    print(f"beam sigma_y at peak    : {d[ipk,9]*1e9:.1f} nm (whole beam in the box; sigma_y* = {meta['sy']*1e9:.1f})")

    P = load(run)
    ebins = np.logspace(-5, np.log10(Eb), 121)          # GeV
    tbins = np.logspace(-4, np.log10(np.pi / 2), 81)    # rad, folded polar angle to the detector axis
    res["hist"] = {"E_edges_GeV": ebins.tolist(), "theta_edges_rad": tbins.tolist()}

    # ---- beamstrahlung photons ----
    print("\nbeamstrahlung photons (per crossing, both beams):")
    tot = dict(n=0.0, e=0.0, macro=0)
    for sp, sgn in (("pho1", +1), ("pho2", -1)):
        if sp not in P:
            continue
        p = P[sp]; E = np.sqrt(p["px"] ** 2 + p["py"] ** 2 + p["pz"] ** 2); w = p["w"]
        ax = np.array([np.sin(th), 0.0, sgn * np.cos(th)])   # direction of the parent beam
        cosa = (p["px"] * ax[0] + p["pz"] * ax[2]) / E
        ang = np.arccos(np.clip(cosa, -1, 1))
        thd = np.arctan2(np.hypot(p["px"], p["py"]), np.abs(p["pz"]))
        r = {"macro": int(len(w)), "n": float(w.sum()), "n_per_beam_particle": float(w.sum() / N),
             "E_mean_MeV": float(np.average(E, weights=w) * 1e3), "E_median_MeV": wq(E, w, 0.5) * 1e3,
             "E_max_GeV": float(E.max()), "energy_loss_fraction": float((w * E).sum() / (N * Eb)),
             "angle_to_beam_p50_urad": wq(ang, w, 0.5) * 1e6, "angle_to_beam_p99_urad": wq(ang, w, 0.99) * 1e6,
             "n_theta_gt_50mrad": float(w[thd > 0.05].sum()), "n_theta_gt_100mrad": float(w[thd > 0.10].sum()),
             "E_theta_gt_100mrad_GeV": float((w * E)[thd > 0.10].sum())}
        res[sp] = r
        res["hist"][sp + "_E"] = np.histogram(E, ebins, weights=w)[0].tolist()
        tot["n"] += r["n"]; tot["e"] += (w * E).sum(); tot["macro"] += r["macro"]
        print(f"  {sp}: {r['n']:.3e} photons ({r['n_per_beam_particle']:.3f} per beam particle, {r['macro']} macro), "
              f"mean {r['E_mean_MeV']:.2f} MeV, median {r['E_median_MeV']:.3f} MeV, max {r['E_max_GeV']:.2f} GeV")
        print(f"        energy radiated = {r['energy_loss_fraction']*100:.4f} % of the beam energy; "
              f"angle to parent beam: median {r['angle_to_beam_p50_urad']:.0f} urad, 99% within {r['angle_to_beam_p99_urad']:.0f} urad")
        print(f"        beyond 100 mrad from the detector axis: {r['n_theta_gt_100mrad']:.3g} photons, "
              f"{r['E_theta_gt_100mrad_GeV']:.3g} GeV")
    res["photons_total"] = {"n": tot["n"], "energy_GeV": float(tot["e"]), "macro": tot["macro"]}
    print(f"  total: {tot['n']:.3e} photons carrying {tot['e']:.3e} GeV per crossing")

    # ---- incoherent pairs ----
    print("\nincoherent pairs (per crossing; e- and e+ counted separately):")
    print(f"  {'species':8} {'macro':>8} {'particles':>11} {'mean E':>10} {'E>10 MeV':>10} "
          f"{'theta>100mrad':>14} {'pT>5MeV&th>20mrad':>18} {'left in box':>12}")
    allp = dict(n=0.0, e=0.0)
    for sp in meta["products"]:
        if sp not in P:
            print(f"  {sp:8} {0:8d}"); res[sp] = {"macro": 0, "n": 0.0}; continue
        p = P[sp]; w = p["w"]
        pt = np.hypot(p["px"], p["py"]); pp = np.sqrt(pt ** 2 + p["pz"] ** 2)
        E = np.sqrt(pp ** 2 + ME_GEV ** 2); thd = np.arctan2(pt, np.abs(p["pz"]))
        r = {"macro": int(len(w)), "n": float(w.sum()), "E_mean_MeV": float(np.average(E, weights=w) * 1e3),
             "energy_GeV": float((w * E).sum()), "n_E_gt_10MeV": float(w[E > 0.01].sum()),
             "n_theta_gt_100mrad": float(w[thd > 0.1].sum()),
             "n_pt5MeV_theta20mrad": float(w[(pt > 0.005) & (thd > 0.02)].sum()),
             "n_left_in_box": float(w[p["face"] == 6].sum()),
             "weight_mean": float(w.mean()), "weight_max": float(w.max()),
             "exit_face_fraction": {f: float(w[p["face"] == i].sum() / w.sum()) for i, f in enumerate(FACES)}}
        res[sp] = r
        res["hist"][sp + "_E"] = np.histogram(E, ebins, weights=w)[0].tolist()
        res["hist"][sp + "_theta"] = np.histogram(thd, tbins, weights=w)[0].tolist()
        res["hist"][sp + "_pt_theta"] = np.histogram2d(np.log10(np.maximum(pt, 1e-7)), np.log10(np.maximum(thd, 1e-6)),
                                                       bins=[np.linspace(-6, 1, 36), np.linspace(-5, 0.2, 27)],
                                                       weights=w)[0].tolist()
        allp["n"] += r["n"]; allp["e"] += r["energy_GeV"]
        print(f"  {sp:8} {r['macro']:8d} {r['n']:11.4g} {r['E_mean_MeV']:8.1f} MeV {r['n_E_gt_10MeV']:10.4g} "
              f"{r['n_theta_gt_100mrad']:14.4g} {r['n_pt5MeV_theta20mrad']:18.4g} {r['n_left_in_box']:12.3g}")
    res["pairs_total"] = {"n_particles": allp["n"], "energy_GeV": allp["e"]}
    print(f"  total: {allp['n']:.4g} e+/e- carrying {allp['e']:.4g} GeV per crossing")
    big = [(P[s]["w"][P[s]["w"] > 1].sum(), P[s]["w"].sum()) for s in meta["products"] if s in P]
    if big:
        frac = sum(b[0] for b in big) / max(sum(b[1] for b in big), 1e-300)
        res["pairs_total"]["fraction_in_weights_above_1"] = float(frac)
        print(f"  share of pairs carried by macroparticles with weight above 1: {frac*100:.1f} %"
              f"{'' if frac < 0.05 else '   (raise --event-mult or lower --pair-weight)'}")
    json.dump(res, open(os.path.join(run, "yields.json"), "w"))
    print(f"\nwrote {os.path.join(run, 'yields.json')}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r.rstrip("/"))
