#!/usr/bin/env python3
"""Which pair particles make the crystal energy, and where did they end?

    python where_from.py escan_fld_Z_1x escan_bpas3_Z_1x [...]     (reads the first 3 files of each)

Every crystal hit carries "contributions": energy deposits, each credited to an
MCParticle that Geant4 kept. Secondaries that were not kept (soft ones, or ones
born outside the tracking region) are credited to their nearest kept ancestor.
For each contribution this script follows the parent links up to the pair
particle it descends from (generatorStatus 1) and tabulates the crystal energy
  A) by the type of the credited particle,
  B) by where the credited particle was born,
  C) by where the ANCESTOR pair particle ended  <- the decisive table: it says
     which part of the machine/detector the crystal energy was launched from.
"""
import glob, sys
import numpy as np
import uproot

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
ZB = [0, 90, 500, 1074, 1190, 1500, 2000, 2300, 3000, 6000, 12000, 1e9]   # mm
RB = [0, 12, 35, 100, 250, 1000, 1e9]                                      # mm
CAL = "SCEPCal_MainEdep"


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


def find(keys, w):
    m = [k for k in keys if k == w or k.endswith("/" + w)]
    return m[0] if m else None


for s in sys.argv[1:]:
    files = sorted(glob.glob(f"{RUN}/sim_outputs/{s}/events_*.root"))[:3]
    acc = {k: [] for k in ("bz", "br", "ez", "er", "e", "pdg", "isgen")}
    hit_tot, nev = 0.0, 0
    for f in files:
        t = uproot.open(f)["events"]
        keys = t.keys(recursive=True)
        want = {"gen": "MCParticles.generatorStatus", "pdg": "MCParticles.PDG",
                "vx": "MCParticles.vertex.x", "vy": "MCParticles.vertex.y", "vz": "MCParticles.vertex.z",
                "ex": "MCParticles.endpoint.x", "ey": "MCParticles.endpoint.y", "ez": "MCParticles.endpoint.z",
                "pb": "MCParticles.parents_begin", "pe": "MCParticles.parents_end",
                "par": "_MCParticles_parents.index",
                "ce": f"{CAL}Contributions.energy", "cp": f"_{CAL}Contributions_particle.index",
                "he": f"{CAL}.energy"}
        full = {k: find(keys, w) for k, w in want.items()}
        miss = [want[k] for k, v in full.items() if v is None]
        if miss:
            sys.exit(f"branches not found: {miss}\n  branches mentioning {CAL}: "
                     f"{[k for k in keys if CAL in k and '/' not in k][:20]}\n  MCParticles relations: "
                     f"{[k for k in keys if '_MCParticles' in k and '/' not in k]}")
        a = t.arrays(list(full.values()), library="np")
        a = {k: a[v] for k, v in full.items()}
        for i in range(len(a["gen"])):
            gen = np.asarray(a["gen"][i]); n = len(gen)
            pb, pe, par = (np.asarray(a[k][i], np.int64) for k in ("pb", "pe", "par"))
            fp = np.full(n, -1, np.int64); has = (pe > pb) & (pb < len(par))
            fp[has] = par[pb[has]]
            fp[(fp < 0) | (fp >= n)] = -1
            anc = np.arange(n)
            for _ in range(500):                      # climb to the pair particle
                up = (gen[anc] != 1) & (fp[anc] >= 0)
                if not up.any():
                    break
                anc = np.where(up, fp[anc], anc)
            ce = np.asarray(a["ce"][i], float); cp = np.asarray(a["cp"][i], np.int64)
            hit_tot += np.asarray(a["he"][i], float).sum(); nev += 1
            ok = (cp >= 0) & (cp < n)
            ce, cp = ce[ok], cp[ok]
            an = anc[cp]
            V = {c: np.asarray(a["v" + c][i], float) for c in "xyz"}
            E = {c: np.asarray(a["e" + c][i], float) for c in "xyz"}
            acc["bz"].append(np.abs(V["z"][cp])); acc["br"].append(np.hypot(V["x"][cp], V["y"][cp]))
            acc["ez"].append(np.where(gen[an] == 1, np.abs(E["z"][an]), -1.0))
            acc["er"].append(np.hypot(E["x"][an], E["y"][an]))
            acc["e"].append(ce); acc["pdg"].append(np.asarray(a["pdg"][i])[cp]); acc["isgen"].append(gen[cp] == 1)
    if nev == 0:
        print(f"\n=== {s}: no events found"); continue
    A = {k: np.concatenate(v) for k, v in acc.items()}
    w = A["e"] * 1e3 / nev                                                      # MeV per event
    print(f"\n=== {s}: {nev} events; crystal energy {hit_tot*1e3/nev:.1f} MeV/event in hits, "
          f"{w.sum():.1f} MeV/event credited to particles via contributions")
    if w.sum() < 0.5 * hit_tot * 1e3 / nev:
        print("  (less than half of the hit energy has contributions; tables below cover only that part)")
    print("  A) credited particle:")
    for lab, m in (("the pair e+/e- itself (or its untracked secondaries)", A["isgen"]),
                   ("kept secondary photon", ~A["isgen"] & (A["pdg"] == 22)),
                   ("kept secondary electron", ~A["isgen"] & (A["pdg"] == 11)),
                   ("kept secondary positron", ~A["isgen"] & (A["pdg"] == -11)),
                   ("kept secondary, other", ~A["isgen"] & (np.abs(A["pdg"]) != 11) & (A["pdg"] != 22))):
        print(f"     {lab:52} {w[m].sum()/w.sum():6.1%}")
    m = ~A["isgen"]
    if m.any():
        table("B) birth point of the credited kept secondaries (share of their energy)", A["bz"][m], A["br"][m], w[m], "MeV")
    m = A["ez"] >= 0
    table("C) END POINT of the ancestor pair particle (share of crystal energy)", A["ez"][m], A["er"][m], w[m], "MeV")
    if (~m).any():
        print(f"     no pair-particle ancestor found for {w[~m].sum()/w.sum():.1%} of the energy")
