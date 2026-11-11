#!/usr/bin/env python3
"""Where do the background pairs end, and where does their crystal energy land?

    python where_bkg.py escan_fld_Z_1x escan_bpas3_Z_1x [...]     (reads the first 3 files of each)

Part 1 - the pair particles themselves (generator particles, MCParticles with
generatorStatus 1): where Geant4 stopped tracking them (endpoint), counted and
energy-weighted, in regions of |z| and radius. "left the world" = |z| > 12 m or r > 12 m.
Part 2 - the crystal energy, by |z| and radius of the hit.
"""
import glob, sys
import numpy as np
import uproot

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
ZB = [0, 90, 500, 1190, 1500, 2000, 2300, 3000, 6000, 12000, 1e9]          # mm
RB = [0, 12, 35, 100, 250, 1000, 1e9]                                       # mm
ME = 0.000511


def table(title, z, r, w, unit):
    tot = w.sum()
    print(f"  {title}  (total {tot:.1f} {unit} per event)")
    print("    |z| range [mm]     " + "".join(f"{f'r<{int(b)}' if b < 1e8 else 'r>' + str(int(RB[-2])):>10}" for b in RB[1:]))
    for z0, z1 in zip(ZB[:-1], ZB[1:]):
        m = (z >= z0) & (z < z1)
        if w[m].sum() < 0.005 * tot:
            continue
        row = [w[m & (r >= r0) & (r < r1)].sum() / tot for r0, r1 in zip(RB[:-1], RB[1:])]
        lab = f"{int(z0)}-{int(z1)}" if z1 < 1e8 else f">{int(z0)}"
        print(f"    {lab:>17}  " + "".join(f"{v:10.1%}" if v >= 0.0005 else f"{'':>10}" for v in row))


for s in sys.argv[1:]:
    files = sorted(glob.glob(f"{RUN}/sim_outputs/{s}/events_*.root"))[:3]
    Z, R, E, HZ, HR, HE, nev = [], [], [], [], [], [], 0
    for f in files:
        t = uproot.open(f)["events"]
        want = ["MCParticles.generatorStatus", "MCParticles.endpoint.x", "MCParticles.endpoint.y",
                "MCParticles.endpoint.z", "MCParticles.momentum.x", "MCParticles.momentum.y",
                "MCParticles.momentum.z",
                "SCEPCal_MainEdep.energy", "SCEPCal_MainEdep.position.x", "SCEPCal_MainEdep.position.y",
                "SCEPCal_MainEdep.position.z"]
        keys = t.keys(recursive=True)
        full = {}
        for w in want:                      # branch names differ between podio versions: match by suffix
            m = [k for k in keys if k == w or k.endswith("/" + w)]
            if not m:
                sys.exit(f"branch {w} not found; some available: {[k for k in keys if w.split('.')[0] in k][:8]}")
            full[w] = m[0]
        raw = t.arrays(list(full.values()), library="np")
        a = {w: raw[full[w]] for w in want}
        for i in range(len(a["MCParticles.generatorStatus"])):
            g = a["MCParticles.generatorStatus"][i] == 1
            ex, ey, ez = (np.asarray(a[f"MCParticles.endpoint.{c}"][i], float)[g] for c in "xyz")
            px, py, pz = (np.asarray(a[f"MCParticles.momentum.{c}"][i], float)[g] for c in "xyz")
            Z.append(np.abs(ez)); R.append(np.hypot(ex, ey)); E.append(np.sqrt(px**2 + py**2 + pz**2 + ME**2))
            HE.append(np.asarray(a["SCEPCal_MainEdep.energy"][i], float))
            HZ.append(np.abs(np.asarray(a["SCEPCal_MainEdep.position.z"][i], float)))
            HR.append(np.hypot(np.asarray(a["SCEPCal_MainEdep.position.x"][i], float),
                               np.asarray(a["SCEPCal_MainEdep.position.y"][i], float)))
            nev += 1
    Z, R, E = (np.concatenate(v) for v in (Z, R, E))
    HZ, HR, HE = (np.concatenate(v) for v in (HZ, HR, HE))
    print(f"\n=== {s}: {nev} events, {len(E)/nev:.0f} pair particles per event")
    table("pair particles: where they end, share of NUMBER", Z, R, np.ones_like(E) / nev, "particles")
    table("pair particles: where they end, share of ENERGY", Z, R, E / nev, "GeV")
    table("crystal energy: where it is deposited", HZ, HR, HE * 1e3 / nev, "MeV")
