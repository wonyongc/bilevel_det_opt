#!/usr/bin/env python3
"""Which particles bring energy INTO the crystals, where were they born, and which
pair particle (and where it ended) did they come from?

    python where_enter.py escan_trace_bpas3_Z_1x [...]       (all files of each sample)

Works from MCParticles alone (the SCEPCal hits carry no particle links). Meant for a
run with every particle kept (run_trace.sbatch: --part.keepAllParticles True, no
tracker-region filter); on a normal sample it only sees particles above 1 MeV born
in the tracking region.
"Crystal region" is inferred from the hit positions themselves (barrel shell and two
endcap discs, with a margin). A particle ENTERS the crystals if it is born outside
that region and ends inside it. Its energy brought in = kinetic energy at birth
(+1.022 MeV for a positron, which annihilates). The sum is compared with the hit
energy as a check.
Tables (share of entering energy):
  A) by particle type   B) by birth point   C) by where the ancestor pair particle ended
"""
import glob, sys
import numpy as np
import uproot

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
ZB = [0, 90, 500, 1074, 1190, 1500, 2000, 2300, 3000, 6000, 12000, 1e9]   # mm
RB = [0, 12, 35, 100, 250, 1000, 1e9]                                      # mm
CAL = "SCEPCal_MainEdep"
ME = 0.000511


def table(title, z, r, w, unit):
    tot = w.sum()
    print(f"  {title}  (total {tot:.1f} {unit} per event)")
    if tot <= 0:
        return
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


want = {"gen": "MCParticles.generatorStatus", "pdg": "MCParticles.PDG", "m": "MCParticles.mass",
        "vx": "MCParticles.vertex.x", "vy": "MCParticles.vertex.y", "vz": "MCParticles.vertex.z",
        "ex": "MCParticles.endpoint.x", "ey": "MCParticles.endpoint.y", "ez": "MCParticles.endpoint.z",
        "px": "MCParticles.momentum.x", "py": "MCParticles.momentum.y", "pz": "MCParticles.momentum.z",
        "pb": "MCParticles.parents_begin", "pe": "MCParticles.parents_end", "par": "_MCParticles_parents.index",
        "he": f"{CAL}.energy", "hx": f"{CAL}.position.x", "hy": f"{CAL}.position.y", "hz": f"{CAL}.position.z"}

for s in sys.argv[1:]:
    files = sorted(glob.glob(f"{RUN}/sim_outputs/{s}/events_*.root"))
    if not files:
        print(f"\n=== {s}: no files"); continue
    # ---- pass 1: crystal region from the hits ----
    HR, HZ = [], []
    for f in files[:2]:
        t = uproot.open(f)["events"]; keys = t.keys(recursive=True)
        a = t.arrays([find(keys, want[k]) for k in ("hx", "hy", "hz")], library="np")
        hx, hy, hz = (np.concatenate(list(v)) if len(v) else np.zeros(0) for v in a.values())
        HR.append(np.hypot(hx, hy)); HZ.append(np.abs(hz))
    HR, HZ = np.concatenate(HR), np.concatenate(HZ)
    bar = HZ < 0.8 * HZ.max()
    RB0 = np.percentile(HR[bar], 0.5) - 150 if bar.any() else 1800.0
    end = HR < RB0
    ZE0 = np.percentile(HZ[end], 0.5) - 150 if end.any() else 2300.0
    RE0 = max(np.percentile(HR[end], 0.5) - 10 if end.any() else 100.0, 0.0)
    RMAX, ZMAX = HR.max() + 300, HZ.max() + 300

    def inside(x, y, z):
        r, az = np.hypot(x, y), np.abs(z)
        return ((r >= RB0) & (r <= RMAX) & (az < ZE0)) | ((az >= ZE0) & (az <= ZMAX) & (r >= RE0) & (r <= RMAX))

    print(f"\n=== {s}: crystal region inferred from hits: barrel r {RB0:.0f}-{RMAX:.0f} mm for |z| < {ZE0:.0f}; "
          f"endcaps |z| {ZE0:.0f}-{ZMAX:.0f} mm, r {RE0:.0f}-{RMAX:.0f} mm")
    acc = {k: [] for k in ("bz", "br", "ez", "er", "e", "pdg", "isgen")}
    hit_tot, nev, npart = 0.0, 0, 0
    for f in files:
        t = uproot.open(f)["events"]; keys = t.keys(recursive=True)
        full = {k: find(keys, w) for k, w in want.items()}
        miss = [want[k] for k, v in full.items() if v is None]
        if miss:
            sys.exit(f"branches not found: {miss}")
        a = t.arrays(list(full.values()), library="np"); a = {k: a[v] for k, v in full.items()}
        for i in range(len(a["gen"])):
            g = lambda k, dt=float: np.asarray(a[k][i], dt)
            gen = g("gen", np.int64); n = len(gen); npart += n; nev += 1
            hit_tot += g("he").sum()
            pb, pe, par = g("pb", np.int64), g("pe", np.int64), g("par", np.int64)
            fp = np.full(n, -1, np.int64); has = (pe > pb) & (pb < len(par)); fp[has] = par[pb[has]]
            fp[(fp < 0) | (fp >= n)] = -1
            anc = np.arange(n)
            for _ in range(1000):
                up = (gen[anc] != 1) & (fp[anc] >= 0)
                if not up.any():
                    break
                anc = np.where(up, fp[anc], anc)
            vx, vy, vz, ex, ey, ez = (g(k) for k in ("vx", "vy", "vz", "ex", "ey", "ez"))
            ent = ~inside(vx, vy, vz) & inside(ex, ey, ez)
            pdg, mass = g("pdg", np.int64), g("m")
            p2 = g("px") ** 2 + g("py") ** 2 + g("pz") ** 2
            ek = np.sqrt(p2 + mass ** 2) - mass + np.where(pdg == -11, 2 * ME, 0.0)
            idx = np.nonzero(ent)[0]; an = anc[idx]
            acc["bz"].append(np.abs(vz[idx])); acc["br"].append(np.hypot(vx[idx], vy[idx]))
            acc["ez"].append(np.where(gen[an] == 1, np.abs(ez[an]), -1.0)); acc["er"].append(np.hypot(ex[an], ey[an]))
            acc["e"].append(ek[idx]); acc["pdg"].append(pdg[idx]); acc["isgen"].append(gen[idx] == 1)
    A = {k: np.concatenate(v) for k, v in acc.items()}
    w = A["e"] * 1e3 / nev
    print(f"  {nev} events, {npart/nev:.0f} MCParticles per event kept; crystal hit energy {hit_tot*1e3/nev:.1f} MeV/event; "
          f"energy carried in by entering particles {w.sum():.1f} MeV/event ({len(w)/nev:.0f} particles/event)")
    print("  A) entering particle:")
    for lab, m in (("pair e+/e- straight from the IP", A["isgen"]),
                   ("secondary photon", ~A["isgen"] & (A["pdg"] == 22)),
                   ("secondary electron", ~A["isgen"] & (A["pdg"] == 11)),
                   ("secondary positron", ~A["isgen"] & (A["pdg"] == -11)),
                   ("secondary, other", ~A["isgen"] & (np.abs(A["pdg"]) != 11) & (A["pdg"] != 22))):
        print(f"     {lab:34} {w[m].sum()/max(w.sum(),1e-30):6.1%}   ({m.sum()/nev:.0f} per event)")
    ph = ~A["isgen"] & (A["pdg"] == 22)
    if ph.any():
        e = A["e"][ph] * 1e3
        print(f"     photons by energy: <0.6 MeV {w[ph][e<0.6].sum()/w.sum():.1%}, 0.6-5 MeV {w[ph][(e>=0.6)&(e<5)].sum()/w.sum():.1%}, "
              f">5 MeV {w[ph][e>=5].sum()/w.sum():.1%}  (share of all entering energy)")
    table("B) BIRTH POINT of the entering particles (share of entering energy)", A["bz"], A["br"], w, "MeV")
    m = A["ez"] >= 0
    table("C) END POINT of their ancestor pair particle (share of entering energy)", A["ez"][m], A["er"][m], w[m], "MeV")
    if (~m).any():
        print(f"     no pair-particle ancestor found for {w[~m].sum()/w.sum():.1%} of the energy")
