#!/usr/bin/env python3
"""Which pair particles can reach the crystals?  Per crossing, weighted.

    python pair_tail.py Z WW ZH tt        (run in the WarpX output folder, e.g. escan/prod)

The crystal calorimeter in the standalone geometry covers polar angles
184 mrad < theta < pi - 184 mrad.  In a 2 T field a particle needs a
transverse momentum of roughly 0.3*B*r/2 to reach radius r, i.e. > ~0.13 GeV
for the inner endcap edge (r ~ 0.42 m) and > ~0.6 GeV for the barrel (r ~ 2 m).
Prints, per species, how many particles per crossing are inside the angular
acceptance above several pT thresholds and how much energy they carry, plus
where (which face of the WarpX box) they left and the largest pT seen.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_yields import load, ME_GEV, FACES

TH0 = 0.184
CUTS = (0.0, 0.01, 0.05, 0.13, 0.6)
for run in sys.argv[1:]:
    run = run.rstrip("/")
    meta = json.load(open(os.path.join(run, "deck_meta.json")))
    P = load(run)
    print(f"\n=== {meta['name']}: e+/e- inside {TH0*1e3:.0f} mrad < theta < pi-{TH0*1e3:.0f} mrad, per crossing")
    print(f"  {'species':8} " + " ".join(f"{'pT>'+format(c,'g'):>14}" for c in CUTS) + f" {'E(pT>0.13) GeV':>15} {'max pT':>8}  exit face of pT>0.13 (share)")
    tot = np.zeros(len(CUTS)); etot = 0.0
    for sp in meta["products"]:
        if sp not in P:
            continue
        p = P[sp]; w = p["w"]
        pt = np.hypot(p["px"], p["py"]); E = np.sqrt(pt**2 + p["pz"]**2 + ME_GEV**2)
        th = np.arctan2(pt, p["pz"])
        acc = (th > TH0) & (th < np.pi - TH0)
        n = [float(w[acc & (pt > c)].sum()) for c in CUTS]
        hard = acc & (pt > 0.13)
        e13 = float((w * E)[hard].sum())
        faces = {f: float(w[hard & (p["face"] == i)].sum()) for i, f in enumerate(FACES + ["box"])}
        s = sum(faces.values()) or 1.0
        fs = " ".join(f"{k}:{v/s:.0%}" for k, v in faces.items() if v > 0)
        tot += n; etot += e13
        print(f"  {sp:8} " + " ".join(f"{x:14.4g}" for x in n) + f" {e13:15.4g} {pt.max():8.3f}  {fs}")
    print(f"  {'total':8} " + " ".join(f"{x:14.4g}" for x in tot) + f" {etot:15.4g}")
