#!/usr/bin/env python3
"""Polar-angle profile of the crystal background with the crystal count per bin, to remove the
segmentation pattern from the energy-per-mrad figure (fig_mdi_theta).

    python escan_theta_norm.py [--nfiles 5]        -> $RUN/escan_theta_norm.json   (login node, ~10 min)

Why: escan_mdimap.py histograms the hit energy in 10 mrad bins of the polar angle of the hit
position, which is the crystal centre. Bins then contain different numbers of crystal rows, and the
resulting sawtooth is the same in every sample (the bin-to-bin residuals of the four independent
working points are 60-86 % correlated). This script records, in the same bins and separately for
barrel (|z| < 2.3 m) and endcaps:
  - the energy per crossing of each working point (crystals and timing layer), and
  - the number of distinct cells whose centre falls in the bin, counted from every cell that is hit
    at least once in the 1x and 3x background samples of all four working points,
plus the same in 0.5 mrad bins (to see the rows), and how often each cell was seen (coverage).
Environment: key4hep, then the bilevel venv (two separate `source` lines).
"""
import argparse, glob, json, os
import numpy as np
import uproot

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
PTS = ("Z", "WW", "ZH", "tt")
ZEND = 2300.0
TH = np.linspace(0, np.pi / 2, 158)            # the bins of escan_mdimap.py (10.0 mrad)
FINE = np.linspace(0, np.pi / 2, 3143)         # 0.5 mrad
LAYERS = (("SCEPCal_MainEdep", "crystal"), ("SCEPCal_TimingEdep", "timing"))


def find(keys, coll, leaf):
    for k in keys:
        b = k.split("/")[-1]
        if b in (f"{coll}.{leaf}", f"{coll}_{leaf}", f"{coll}_{leaf.replace('.', '_')}"):
            return k
    raise SystemExit(f"branch {coll}.{leaf} not found")


def read(f, coll, tree="events"):
    t = uproot.open(f)[tree]; keys = t.keys(recursive=True)
    return tuple(t[find(keys, coll, v)].array(library="np") for v in ("cellID", "energy", "position.x", "position.y", "position.z"))


ap = argparse.ArgumentParser()
ap.add_argument("--nfiles", type=int, default=5)
ap.add_argument("--prefix", default="escan_lumi2k")
ap.add_argument("--out", default=os.path.join(RUN, "escan_theta_norm.json"))
a = ap.parse_args()

R = {"edges_rad": TH.tolist(), "fine_edges_rad": [float(FINE[0]), float(FINE[-1]), len(FINE) - 1], "zend_mm": ZEND, "samples": {}, "cells": {}}
seen = {k: [] for _, k in LAYERS}              # per layer: list of (cellID, theta, is_barrel, count) per file
for pt in PTS:
    for mode in ("1x", "3x"):
        s = f"{a.prefix}_{pt}_{mode}"
        files = sorted(glob.glob(f"{RUN}/sim_outputs/{s}/events_*.root"))[: a.nfiles]
        if not files:
            print(f"== {s}: no files, skipped"); continue
        H = {k: {"barrel": np.zeros(len(TH) - 1), "endcap": np.zeros(len(TH) - 1), "fine": np.zeros(len(FINE) - 1)} for _, k in LAYERS}
        nev = 0
        for f in files:
            for coll, key in LAYERS:
                C, E, X, Y, Z = read(f, coll)
                if key == "crystal":
                    nev += len(E)
                cid = np.concatenate([np.asarray(c).astype(np.uint64) for c in C]) if len(C) else np.zeros(0, np.uint64)
                if cid.size == 0:
                    continue
                e = np.concatenate([np.asarray(v, float) for v in E]) * 1e3
                x, y, z = (np.concatenate([np.asarray(v, float) for v in V]) for V in (X, Y, Z))
                th = np.arctan2(np.hypot(x, y), np.abs(z)); bar = np.abs(z) < ZEND
                H[key]["barrel"] += np.histogram(th[bar], TH, weights=e[bar])[0]
                H[key]["endcap"] += np.histogram(th[~bar], TH, weights=e[~bar])[0]
                H[key]["fine"] += np.histogram(th, FINE, weights=e)[0]
                u, first, cnt = np.unique(cid, return_index=True, return_counts=True)
                seen[key].append((u, th[first].astype(np.float32), bar[first], cnt))
        n = max(nev, 1)
        if mode == "1x":
            R["samples"][s] = {"crossings": nev, **{k: {r: (v / n).tolist() for r, v in H[k].items()} for k in H}}
        print(f"== {s}: {len(files)} files, {nev} crossings, crystal energy {H['crystal']['barrel'].sum()/n + H['crystal']['endcap'].sum()/n:.0f} MeV per crossing", flush=True)

for key in seen:
    if not seen[key]:
        continue
    cid = np.concatenate([t[0] for t in seen[key]]); th = np.concatenate([t[1] for t in seen[key]])
    bar = np.concatenate([t[2] for t in seen[key]]); cnt = np.concatenate([t[3] for t in seen[key]])
    u, first, inv = np.unique(cid, return_index=True, return_inverse=True)
    tot = np.bincount(inv, weights=cnt).astype(int)
    thu, baru = th[first].astype(float), bar[first]
    R["cells"][key] = {"distinct": int(len(u)), "seen_once": int((tot == 1).sum()), "seen_twice": int((tot == 2).sum()),
                       "median_times_seen": float(np.median(tot)),
                       "barrel": np.histogram(thu[baru], TH)[0].tolist(), "endcap": np.histogram(thu[~baru], TH)[0].tolist(),
                       "fine": np.histogram(thu, FINE)[0].tolist()}
    c = R["cells"][key]
    print(f"== cells, {key}: {c['distinct']} distinct ({int(baru.sum())} barrel, {int((~baru).sum())} endcap), seen once {c['seen_once']}, "
          f"twice {c['seen_twice']}, median {c['median_times_seen']:.0f} times  -> estimated never seen ~ {c['seen_once']**2 / max(2*c['seen_twice'],1):.0f}")
    occ = np.array(c["fine"]); nz = np.nonzero(occ)[0]
    print(f"   0.5 mrad bins with cells: {len(nz)} of {nz[-1]-nz[0]+1} between {1e3*FINE[nz[0]]:.1f} and {1e3*FINE[nz[-1]+1]:.1f} mrad "
          f"(rows are resolved if this fraction is well below 1)")
    n10 = np.array(c["barrel"]) + np.array(c["endcap"]); m = n10 > 0
    print(f"   cells per 10 mrad bin: median {np.median(n10[m]):.0f}, 16-84 % {np.percentile(n10[m],16):.0f}-{np.percentile(n10[m],84):.0f}")
json.dump(R, open(a.out, "w"))
print("wrote", a.out, f"({os.path.getsize(a.out)/1e6:.1f} MB)")
