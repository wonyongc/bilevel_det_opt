#!/usr/bin/env python3
"""Diagnose one inner-loop sample: cluster-energy distribution at chosen (R, E_thr) points.

    python fulldet/diag_loop.py money_v4_cld_Z_g5000MeV_fixed [--mix 0x] [--nev 400] \\
        [--window 15] [--metric pointing] [--ecal 0.135] [--branch ECalBarrelCollection]

Reads the mix spec written by the loop (mix_<mix>.mixspec.json), builds the mixed events with
paired_mix.py exactly as run_money.py does, and for each parameter point prints: acceptance,
median / q16 / q84 of the cluster energy in MeV (zeros kept, as the eres_cal score sees them),
hits per cluster, the score, and the ten most frequent cluster energies (to catch a degenerate
distribution). Also shows, for the first events, the seed hit chosen by the window vs the
highest-energy hit. Environment: key4hep + the bilevel venv (two separate `source` lines).
"""
import argparse, collections, json, os, sys
import numpy as np

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
ap = argparse.ArgumentParser()
ap.add_argument("money_dir"); ap.add_argument("--mix", default="0x"); ap.add_argument("--nev", type=int, default=400)
ap.add_argument("--window", type=float, default=15.0); ap.add_argument("--metric", default="pointing")
ap.add_argument("--ecal", type=float, default=0.0, help="E_cal [GeV]; 0 = from fulldet/ecal_<det>.json")
ap.add_argument("--branch", default="")
ap.add_argument("--points", default="2.8:0.002,4.6:0.0011,5:0,10:0,20:0,40:0,80:0,150:0,300:0,20:0.0001,40:0.0005")
a = ap.parse_args()

sys.path.insert(0, RUN); sys.path.insert(0, os.path.expanduser("~/bilevel_det_opt"))
import paired_mix
from bilevel_opt.algorithms.seeded_radius_cog import SeededRadiusCoG
from bilevel_opt.algorithms.registry import SCORES

d = os.path.join(RUN, a.money_dir)
tag = os.path.basename(d.rstrip("/"))                     # money_v4_<det>_<pt>_<base>
parts = tag.split("_"); det, base = parts[2], "_".join(parts[4:])
spec = json.load(open(os.path.join(d, f"mix_{a.mix}.mixspec.json")))
branch = a.branch
if not branch:
    cfg = json.load(open(os.path.join(d, "money_results_0x_1x_3x.json")))["il_cfg"] if os.path.exists(os.path.join(d, "money_results_0x_1x_3x.json")) else None
    branch = cfg["branches"][0]["name"] if cfg else {"cld": "ECalBarrelCollection", "allegro": "ECalBarrelModuleThetaMerged"}[det]
ecal = a.ecal
if not ecal:
    ecal = json.load(open(os.path.join(RUN, "fulldet", f"ecal_{det}.json")))["samples"][base]["E_cal_GeV"]
print(f"{tag}: mix {a.mix}, branch {branch}, E_cal {1e3*ecal:.3f} MeV, window {a.window} mm, metric {a.metric}")
print(f"  signal {spec['signal']}\n  bkg    {spec['bkg']}")
events = paired_mix.mixed_events(spec["signal"], spec["bkg"], branch, a.nev)
print(f"  {len(events)} events, hits/event median {np.median([len(e[branch].E) for e in events]):.0f}")

algo = SeededRadiusCoG(branch, seed_window_mm=a.window, metric=a.metric)
ref = SeededRadiusCoG(branch)
score = SCORES["eres_cal"](sigma0=1.0, E_true=ecal)

# seeds: windowed vs highest hit, first 8 events
print("\n  event  n_hits  seed(window): E[MeV] (x,y,z)            seed(max hit): E[MeV] (x,y,z)")
for i, ev in enumerate(events[:8]):
    c = ev[branch]; s = algo._seed(c); m = int(np.argmax(c.E))
    print(f"  {i:5d} {len(c.E):7d}   {1e3*c.E[s]:7.3f} ({c.x[s]:7.0f},{c.y[s]:7.0f},{c.z[s]:7.0f})   "
          f"{1e3*c.E[m]:7.3f} ({c.x[m]:7.0f},{c.y[m]:7.0f},{c.z[m]:7.0f})")

print(f"\n  {'R[mm]':>6} {'thr[MeV]':>8} {'acc':>5} {'median':>8} {'q16':>8} {'q84':>8} {'sig/mu':>7} {'nhit':>5} {'score':>9}   most frequent E_cluster [MeV] (count)")
for pt in a.points.split(","):
    R, thr = (float(v) for v in pt.split(":"))
    reco = algo.reconstruct(events, {"R_cluster_mm": R, "E_threshold_GeV": thr})
    E = np.array([r.E_cluster if r.n_hits > 0 else 0.0 for r in reco]); nh = np.array([r.n_hits for r in reco])
    acc = float(np.mean(E > 0)); mu = np.median(E); q16, q84 = np.percentile(E, [16, 84]); sig = 0.5 * (q84 - q16)
    top = collections.Counter(np.round(E * 1e3, 4)).most_common(4)
    print(f"  {R:6.1f} {1e3*thr:8.3f} {acc:5.2f} {1e3*mu:8.3f} {1e3*q16:8.3f} {1e3*q84:8.3f} {sig/mu if mu > 0 else float('nan'):7.3f} "
          f"{np.median(nh[nh > 0]) if np.any(nh > 0) else 0:5.0f} {score(events, reco):9.3f}   " + ", ".join(f"{e:.4f} ({n})" for e, n in top))
