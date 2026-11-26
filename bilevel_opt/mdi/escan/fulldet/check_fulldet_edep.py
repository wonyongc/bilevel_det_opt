#!/usr/bin/env python3
"""Energy per event in every calorimeter collection of a full-detector sample.

    python check_fulldet_edep.py sim_outputs/escan_cld_Z_1x [--nfiles 2] [--json out.json]

Prints, per collection: hits per event, energy per event (mean, median), the barrel/endcap
split (by the sign-independent |z| of the hits against ZEND), and the mean energy per hit.
This is the full-detector counterpart of escan_edep.py for the standalone crystals and gives
the "ladder" number for the new geometries. Analysis environment:
    source $HOME/key4hep_setup.sh -r 2024-10-03
    source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate     (two separate lines)
uproot with library="np" only (the stack's awkward is broken).
"""
import argparse, glob, json, os, sys
import numpy as np
import uproot

ap = argparse.ArgumentParser()
ap.add_argument("sample"); ap.add_argument("--nfiles", type=int, default=0)
ap.add_argument("--zend", type=float, default=2300.0, help="|z| [mm] separating barrel from endcaps in the table")
ap.add_argument("--json", default="")
ap.add_argument("--tree", default="events")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.sample, "events_*.root")))
if a.nfiles: files = files[:a.nfiles]
if not files: sys.exit(f"no events_*.root in {a.sample}")

# collections = every branch group that has an .energy and a .position.x leaf
def leaves(tree):
    return list(tree.keys())
t0 = uproot.open(files[0])[a.tree]
names = [n.split("/")[-1] for n in leaves(t0)]
# podio TTree files name the leaves "<Coll>.energy"; some writers use "<Coll>_energy": accept both
def strip(n, leaf):
    for sep in (".", "_"):
        if n.endswith(sep + leaf): return n[:-len(sep + leaf)]
    return None
colls = sorted({c for n in names if (c := strip(n, "energy")) and any(strip(m, "position.x") == c for m in names)})
def find(tree, coll, leaf):
    for n in tree.keys():
        if n.split("/")[-1] in (f"{coll}.{leaf}", f"{coll}_{leaf}"): return n
    raise KeyError(f"{coll}.{leaf}")
print(f"{len(files)} file(s), collections with energy and position: {colls}")
# IDEA: the photon-count collections (SCEPCal_*Scounts/Ccounts) and the dual-readout tubes (DRBT*/DRET*) store
# a number of photons in the energy field, not an energy. They are listed as raw counts and left out of the total.
def is_count(c): return c.endswith("counts") or c.startswith(("DRBT", "DRET"))

acc = {c: {"n": [], "e": [], "ebar": [], "eend": [], "nhit": 0, "esum": 0.0} for c in colls}
nev = 0
for f in files:
    t = uproot.open(f)[a.tree]
    for c in colls:
        E = t[find(t, c, "energy")].array(library="np")
        Z = t[find(t, c, "position.z")].array(library="np")
        for e, z in zip(E, Z):
            e = np.asarray(e, float) * (1.0 if is_count(c) else 1000.0)   # GeV -> MeV (counts stay counts)
            z = np.asarray(z, float)
            acc[c]["n"].append(len(e)); acc[c]["e"].append(e.sum())
            acc[c]["ebar"].append(e[np.abs(z) < a.zend].sum()); acc[c]["eend"].append(e[np.abs(z) >= a.zend].sum())
            acc[c]["nhit"] += len(e); acc[c]["esum"] += e.sum()
    nev += int(t.num_entries)
print(f"{nev} events\n")
print(f"{'collection':34} {'hits/ev':>8} {'MeV/ev':>9} {'median':>8} {'barrel':>8} {'endcap':>8} {'MeV/hit':>8}")
out = {"sample": a.sample, "events": nev, "zend_mm": a.zend, "collections": {}}
tot = 0.0
for c in colls:
    d = acc[c]; n = np.array(d["n"]); e = np.array(d["e"])
    row = {"hits_per_event": float(n.mean()), "MeV_per_event": float(e.mean()), "median_MeV": float(np.median(e)),
           "barrel_MeV": float(np.mean(d["ebar"])), "endcap_MeV": float(np.mean(d["eend"])),
           "MeV_per_hit": float(d["esum"] / d["nhit"]) if d["nhit"] else 0.0}
    row["unit"] = "count" if is_count(c) else "MeV"
    out["collections"][c] = row; tot += 0.0 if is_count(c) else row["MeV_per_event"]
    print(f"{c + (' [count]' if is_count(c) else ''):34} {row['hits_per_event']:8.1f} {row['MeV_per_event']:9.1f} {row['median_MeV']:8.1f} {row['barrel_MeV']:8.1f} {row['endcap_MeV']:8.1f} {row['MeV_per_hit']:8.3f}")
print(f"{'all calorimeter collections (MeV)':34} {'':>8} {tot:9.1f}")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1); print("wrote", a.json)
