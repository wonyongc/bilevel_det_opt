import sys, glob, numpy as np
RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"; sys.path.insert(0, RUN); import paired_mix as pm
print(f"{'sample':18} {'events':>6} {'crystal MeV/evt':>15} {'barrel':>8} {'endcaps':>8} {'timing layer':>13}")
for s in sys.argv[1:]:
    tot, bar, tim, n = 0.0, 0.0, 0.0, 0
    for f in sorted(glob.glob(f"{RUN}/sim_outputs/{s}/events_*.root")):
        c, e, x, y, z = pm.read_calo(f, "SCEPCal_MainEdep")
        for a, d in zip(e, z):
            a = np.asarray(a, float) * 1e3; d = np.abs(np.asarray(d, float))
            tot += a.sum(); bar += a[d < 2300].sum(); n += 1
        c, e, x, y, z = pm.read_calo(f, "SCEPCal_TimingEdep")
        tim += sum(np.asarray(a, float).sum() for a in e) * 1e3
    n = max(n, 1)
    print(f"{s:18} {n:6d} {tot/n:15.1f} {bar/n:8.1f} {(tot-bar)/n:8.1f} {tim/n:13.1f}", flush=True)
