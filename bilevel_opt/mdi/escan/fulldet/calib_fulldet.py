#!/usr/bin/env python3
"""Deposited-energy calibration of the 14 single-particle samples of one full detector.

    DET=CLD python fulldet/calib_fulldet.py [--nfiles 5]   -> $RUN/fulldet/ecal_cld.json

For every sample <det>_<base>_0x: the median energy per event in the ECAL collection the inner
loop will cluster on (barrel collection for the fixed-theta samples, endcap collection for the
forward windows), with no threshold and no clustering. That median is E_cal, the numerator of
the eres_cal score (fulldet/patch_eres_cal.py) and the normalization of the v4 resolutions.
Also prints the sampling fraction f_s = E_cal / E_true and marks a sample "usable" when it
deposits at least a quarter of what the barrel sample of the same energy deposits (the
140-160 and 160-200 mrad windows may fall below the endcap inner radius; v3 had the same).
Analysis environment:
    source $HOME/key4hep_setup.sh -r 2024-10-03 && source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate
"""
import argparse, glob, json, os, re, sys
import numpy as np
import uproot

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
BR = {"CLD": ("ECalBarrelCollection", "ECalEndcapCollection"),
      "ALLEGRO": ("ECalBarrelModuleThetaMerged", "ECalEndcapTurbine"),
      "IDEA": ("SCEPCal_MainEdep", "SCEPCal_MainEdep")}
BASES = ["g100MeV_fixed", "g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed", "g5000MeV_fixed", "e1GeV_fixed",
         "g200MeV_th140-160", "g200MeV_th160-200", "g200MeV_th200-300", "g200MeV_th300-500",
         "g1000MeV_th140-160", "g1000MeV_th160-200", "g1000MeV_th200-300", "g1000MeV_th300-500"]

ap = argparse.ArgumentParser()
ap.add_argument("--nfiles", type=int, default=5)
ap.add_argument("--tree", default="events")
a = ap.parse_args()
DET = os.environ.get("DET", "")
if DET not in BR:
    sys.exit("usage: DET=CLD|ALLEGRO|IDEA python fulldet/calib_fulldet.py")
DTAG = DET.lower()

def etrue(base):
    m = re.match(r"[ge](\d+)(MeV|GeV)_", base)
    return float(m.group(1)) * (1e-3 if m.group(2) == "MeV" else 1.0)

def find(tree, coll, leaf):
    for n in tree.keys():
        if n.split("/")[-1] in (f"{coll}.{leaf}", f"{coll}_{leaf}"): return n
    raise KeyError(f"{coll}.{leaf}")

out = {}
print(f"{'sample':28} {'collection':28} {'events':>6} {'E_true':>7} {'E_cal[MeV]':>10} {'mean':>8} {'sig/mu':>7} {'f_s':>7}  usable")
for base in BASES:
    d = os.path.join(RUN, "sim_outputs", f"{DTAG}_{base}_0x")
    files = sorted(glob.glob(os.path.join(d, "events_*.root")))[: a.nfiles]
    coll = BR[DET][0] if "_fixed" in base else BR[DET][1]
    E = []
    for f in files:
        t = uproot.open(f)[a.tree]
        for e in t[find(t, coll, "energy")].array(library="np"):
            E.append(float(np.asarray(e, float).sum()))
    E = np.asarray(E)
    if len(E) == 0:
        print(f"{base:28} {coll:28} {'none':>6}"); continue
    et = etrue(base)
    med = float(np.median(E)); q16, q84 = np.percentile(E, [16, 84])
    row = {"collection": coll, "events": int(len(E)), "E_true_GeV": et, "E_cal_GeV": med,
           "mean_GeV": float(E.mean()), "core_width_rel": float(0.5 * (q84 - q16) / med) if med > 0 else None,
           "f_s": med / et, "frac_empty": float((E <= 0).mean())}
    out[base] = row
for base, row in out.items():
    ref = out.get(re.sub(r"_th.*", "_fixed", base), row)
    row["usable"] = bool(row["E_cal_GeV"] > 0 and row["f_s"] >= 0.25 * ref["f_s"])
    sm = f"{row['core_width_rel']:7.3f}" if row["core_width_rel"] is not None else "      -"
    print(f"{base:28} {row['collection']:28} {row['events']:6d} {row['E_true_GeV']:7.3f} {1e3*row['E_cal_GeV']:10.3f} "
          f"{1e3*row['mean_GeV']:8.3f} {sm} {row['f_s']:7.4f}  {'yes' if row['usable'] else 'NO'}")
p = os.path.join(RUN, "fulldet", f"ecal_{DTAG}.json")
json.dump({"detector": DET, "nfiles": a.nfiles, "samples": out}, open(p, "w"), indent=1)
print("wrote", p)
