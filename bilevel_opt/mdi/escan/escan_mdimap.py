#!/usr/bin/env python3
"""Where the pair background lands in the calorimeter: maps for the paper's MDI figures.

    python escan_mdimap.py                          -> $RUN/escan_mdimap.json
    python escan_mdimap.py --nfiles 5 escan_lumi2k_Z_1x ...

Reads the background-only samples directly with uproot (SCEPCal_MainEdep = crystals,
SCEPCal_TimingEdep = timing layer). Per sample, as energy per crossing [MeV]:
  - rz:      side view, |z| vs r, 20 mm cells (crystals and timing layer together)
  - theta:   energy vs polar angle folded to 0..pi/2, 10 mrad bins (crystals, timing)
  - barrel:  unrolled barrel, z vs phi (|z| < 2.3 m), 50 mm x 5 degrees
  - endcap:  x-y face view of each endcap (z > 0 and z < 0), 50 mm cells
  - total:   per-crossing crystal energy (mean, quantiles, histogram)
Plus, per working point, the WarpX pair pT-theta histograms from prod/<pt>/yields.json.
Uses the first --nfiles files (80 crossings each) of every sample; maps are averages.
"""
import argparse, glob, json, os
import numpy as np
import uproot

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
ESCAN = "/pscratch/sd/s/sapta/bilevel/escan/prod"
DEFAULT = [f"escan_lumi2k_{p}_1x" for p in ("Z", "WW", "ZH", "tt")] + \
          ["escan_fld_Z_1x", "escan_bpas3_Z_1x", "escan_lumi2k_Z_3x"]
ZEND = 2300.0
B = {"rz_z": np.arange(0, 3201, 20.0), "rz_r": np.arange(0, 2601, 20.0),
     "th": np.linspace(0, np.pi / 2, 158),
     "bz": np.arange(-2300, 2301, 50.0), "bphi": np.linspace(-np.pi, np.pi, 73),
     "exy": np.arange(-2600, 2601, 50.0),
     "etot": np.logspace(1, 5, 61)}


def find(keys, w):
    m = [k for k in keys if k == w or k.endswith("/" + w)]
    return m[0] if m else None


def read(f, coll):
    """Per-event arrays of hit energy [GeV] and position [mm] (same pattern as where_enter.py)."""
    t = uproot.open(f)["events"]; keys = t.keys(recursive=True)
    br = [find(keys, f"{coll}.{v}") for v in ("energy", "position.x", "position.y", "position.z")]
    if None in br:
        raise SystemExit(f"{f}: branches of {coll} not found")
    a = t.arrays(br, library="np")
    return tuple(a[b] for b in br)


def r4(a):
    a = np.asarray(a, float)
    return [float(f"{v:.4g}") for v in a.ravel()] if a.ndim == 1 else [[float(f"{v:.4g}") for v in row] for row in a]


def one(sample, nfiles):
    files = sorted(glob.glob(f"{RUN}/sim_outputs/{sample}/events_*.root"))[:nfiles]
    if not files:
        return None
    H = {"rz": np.zeros((len(B["rz_z"]) - 1, len(B["rz_r"]) - 1)),
         "th_crystal": np.zeros(len(B["th"]) - 1), "th_timing": np.zeros(len(B["th"]) - 1),
         "barrel": np.zeros((len(B["bz"]) - 1, len(B["bphi"]) - 1)),
         "endcap_pos": np.zeros((len(B["exy"]) - 1,) * 2), "endcap_neg": np.zeros((len(B["exy"]) - 1,) * 2)}
    tot, nev = [], 0
    for f in files:
        for coll, key in (("SCEPCal_MainEdep", "crystal"), ("SCEPCal_TimingEdep", "timing")):
            E, X, Y, Z = read(f, coll)
            if key == "crystal":
                nev += len(E)
            for e, x, y, z in zip(E, X, Y, Z):
                e = np.asarray(e, float) * 1e3; x, y, z = (np.asarray(v, float) for v in (x, y, z))
                if key == "crystal":
                    tot.append(float(e.sum()))
                if len(e) == 0:
                    continue
                r, az = np.hypot(x, y), np.abs(z)
                H["rz"] += np.histogram2d(az, r, [B["rz_z"], B["rz_r"]], weights=e)[0]
                H["th_" + key] += np.histogram(np.arctan2(r, az), B["th"], weights=e)[0]
                if key == "crystal":
                    bar = az < ZEND
                    H["barrel"] += np.histogram2d(z[bar], np.arctan2(y[bar], x[bar]), [B["bz"], B["bphi"]], weights=e[bar])[0]
                    for side, m in (("endcap_pos", ~bar & (z > 0)), ("endcap_neg", ~bar & (z < 0))):
                        H[side] += np.histogram2d(x[m], y[m], [B["exy"], B["exy"]], weights=e[m])[0]
        print(f"  {sample}: {os.path.basename(f)}  ({nev} crossings)", flush=True)
    n = max(nev, 1); t = np.asarray(tot)
    out = {"files": len(files), "crossings": nev}
    out.update({k: r4(v / n) for k, v in H.items()})
    out["total_MeV"] = {"mean": float(t.mean()), "std": float(t.std()),
                        "q": {q: float(np.percentile(t, float(q))) for q in ("5", "16", "50", "84", "95")},
                        "hist": np.histogram(np.clip(t, B["etot"][0], B["etot"][-1] * 0.999), B["etot"])[0].tolist()}
    return out


ap = argparse.ArgumentParser()
ap.add_argument("samples", nargs="*")
ap.add_argument("--nfiles", type=int, default=5)
a = ap.parse_args()
R = {"edges": {k: v.tolist() for k, v in B.items()}, "zend_mm": ZEND, "samples": {}, "warpx": {}}
for s in a.samples or DEFAULT:
    print(f"== {s}", flush=True)
    r = one(s, a.nfiles)
    if r is None:
        print("   no files, skipped"); continue
    R["samples"][s] = r
    print(f"   {r['crossings']} crossings, crystal energy {r['total_MeV']['mean']:.0f} MeV per crossing")
for p in ("Z", "WW", "ZH", "tt"):
    y = os.path.join(ESCAN, p, "yields.json")
    if os.path.exists(y):
        h = json.load(open(y)).get("hist", {})
        R["warpx"][p] = {k: v for k, v in h.items() if k.endswith("_pt_theta") or k.endswith("_E") or k.endswith("_theta")
                         or k.endswith("_edges_GeV") or k.endswith("_edges_rad")}
        print(f"== WarpX {p}: {sorted(k for k in R['warpx'][p] if k.endswith('_pt_theta'))}")
out = os.path.join(RUN, "escan_mdimap.json")
json.dump(R, open(out, "w"))
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
