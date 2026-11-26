#!/usr/bin/env python3
"""Dump the histograms for the energy-scan figures into one JSON file.

    python escan_figdump.py                       -> $RUN/escan_figdata.json
    python escan_figdump.py escan_lumi2k_Z_1x ... (choose samples; default list below)

Per background-only sample (crystals = SCEPCal_MainEdep, timing = SCEPCal_TimingEdep):
  - energy per crossing vs polar angle theta (folded to 0..pi/2, 10 mrad bins)
  - energy per crossing vs azimuth phi, barrel and endcaps separately (36 bins)
  - distribution of the total crystal energy per crossing (event-to-event spread)
  - number of crystals hit per crossing (above 0, 1 and 10 MeV)
Plus, per working point, the WarpX pair spectra already stored in yields.json.
"""
import glob, json, os, sys
import numpy as np

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
ESCAN = "/pscratch/sd/s/sapta/bilevel/escan/prod"
sys.path.insert(0, RUN)
import paired_mix as pm

DEFAULT = [f"escan_{g}_{p}_{m}" for g in ("lumi2k",) for p in ("Z", "WW", "ZH", "tt") for m in ("1x", "3x")] + \
          [f"escan_{g}_{p}_1x" for g in ("fld", "bpas3", "lumi") for p in ("Z", "tt")]
TB = np.linspace(0, np.pi / 2, 158)            # 10 mrad
PB = np.linspace(-np.pi, np.pi, 37)
EB = np.logspace(1, 5, 61)                     # MeV per crossing, 10 MeV .. 100 GeV
ZEND = 2300.0                                  # mm, barrel / endcap boundary


def one(sample):
    files = sorted(glob.glob(f"{RUN}/sim_outputs/{sample}/events_*.root"))
    if not files:
        return None
    out = {"files": len(files), "events": 0}
    hth = {k: np.zeros(len(TB) - 1) for k in ("crystal", "timing")}
    hph = {k: np.zeros(len(PB) - 1) for k in ("barrel", "endcap")}
    tot, nhit = [], {0: [], 1: [], 10: []}
    for f in files:
        for name, key in (("SCEPCal_MainEdep", "crystal"), ("SCEPCal_TimingEdep", "timing")):
            c, e, x, y, z = pm.read_calo(f, name)
            for a, xx, yy, zz in zip(e, x, y, z):
                a = np.asarray(a, float) * 1e3; xx, yy, zz = (np.asarray(v, float) for v in (xx, yy, zz))
                if len(a) == 0:
                    if key == "crystal":
                        tot.append(0.0); [nhit[t].append(0) for t in nhit]
                    continue
                r = np.hypot(xx, yy); th = np.arctan2(r, np.abs(zz))
                hth[key] += np.histogram(th, TB, weights=a)[0]
                if key == "crystal":
                    ph = np.arctan2(yy, xx); bar = np.abs(zz) < ZEND
                    hph["barrel"] += np.histogram(ph[bar], PB, weights=a[bar])[0]
                    hph["endcap"] += np.histogram(ph[~bar], PB, weights=a[~bar])[0]
                    tot.append(float(a.sum()))
                    # energy per crystal (several hits can share one cell)
                    u, inv = np.unique(np.asarray(c, np.uint64), return_inverse=True)
                    ec = np.bincount(inv.ravel(), weights=a, minlength=len(u))
                    for t in nhit:
                        nhit[t].append(int((ec > t).sum()))
        print(f"  {sample}: {os.path.basename(f)}  ({len(tot)} crossings so far)", flush=True)
    n = max(len(tot), 1)
    out["events"] = len(tot)
    out["theta_edges"] = TB.tolist(); out["phi_edges"] = PB.tolist(); out["Etot_edges_MeV"] = EB.tolist()
    out["E_vs_theta_MeV"] = {k: (v / n).tolist() for k, v in hth.items()}
    out["E_vs_phi_MeV"] = {k: (v / n).tolist() for k, v in hph.items()}
    t = np.asarray(tot)
    out["Etot_hist"] = np.histogram(np.clip(t, EB[0], EB[-1] * 0.999), EB)[0].tolist()
    out["Etot_mean_MeV"] = float(t.mean()); out["Etot_std_MeV"] = float(t.std())
    out["Etot_quantiles_MeV"] = {q: float(np.percentile(t, float(q))) for q in ("5", "16", "50", "84", "95")}
    out["crystals_hit"] = {f"gt{k}MeV": {"mean": float(np.mean(v)), "p16": float(np.percentile(v, 16)),
                                        "p84": float(np.percentile(v, 84))} for k, v in nhit.items()}
    return out


samples = sys.argv[1:] or DEFAULT
R = {"samples": {}, "warpx": {}}
for s in samples:
    print(f"== {s}", flush=True)
    r = one(s)
    if r is None:
        print("   no files, skipped"); continue
    R["samples"][s] = r
for p in ("Z", "WW", "ZH", "tt"):
    y = os.path.join(ESCAN, p, "yields.json")
    if os.path.exists(y):
        d = json.load(open(y))
        R["warpx"][p] = {k: d[k] for k in ("luminosity_per_crossing_m2", "luminosity_fsr_m2", "hist",
                                           "pairs_total", "photons_total") if k in d}
        for k, v in d.items():
            if isinstance(v, dict) and "E_mean_MeV" in v and "n" in v:
                R["warpx"][p][k] = {kk: v[kk] for kk in ("n", "E_mean_MeV", "macro", "n_theta_gt_100mrad") if kk in v}
out = os.path.join(RUN, "escan_figdata.json")
json.dump(R, open(out, "w"))
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
