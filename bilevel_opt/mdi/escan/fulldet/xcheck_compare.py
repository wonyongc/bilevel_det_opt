#!/usr/bin/env python3
"""Crystal background of the same Z crossings in several geometries / steerings, side by side.

    python fulldet/xcheck_compare.py                      # the default list below
    python fulldet/xcheck_compare.py sim_outputs/A sim_outputs/B ...

All samples are made from the same HepMC file with the same seeds and event ranges (25 files x 80),
so event k of file j is the same crossing everywhere; differences to the first sample are therefore
also given event by event (paired), which removes most of the crossing-to-crossing spread.
Energies in MeV per crossing; barrel = |z| < 2300 mm, as in the paper. Analysis environment:
    source $HOME/key4hep_setup.sh -r 2024-10-03
    source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate
"""
import glob, os, sys
import numpy as np
import uproot

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
DEFAULT = [("baseline, standalone steering", "escan_lumi2k_Z_1x"),
           ("A baseline, full-det. steering", "xchk_lumi_Z_1x"),
           ("C IDEA no DR/muon, no Pb", "xchk_nodrabs_Z_1x"),
           ("B IDEA no DR/muon", "xchk_nodr_Z_1x"),
           ("full IDEA", "escan_idea_Z_1x")]
ZEND = 2300.0


def find(keys, coll, leaf):
    for k in keys:
        if k.split("/")[-1] in (f"{coll}.{leaf}", f"{coll}_{leaf}"):
            return k
    return None


def load(d):
    files = sorted(glob.glob(os.path.join(RUN, "sim_outputs", d, "events_*.root")) if not os.path.isabs(d)
                   else glob.glob(os.path.join(d, "events_*.root")))
    if not files:
        return None
    out = {k: [] for k in ("bar", "end", "n", "tim")}
    for f in files:
        t = uproot.open(f)["events"]; keys = t.keys(recursive=True)
        E = t[find(keys, "SCEPCal_MainEdep", "energy")].array(library="np")
        Z = t[find(keys, "SCEPCal_MainEdep", "position.z")].array(library="np")
        kt = find(keys, "SCEPCal_TimingEdep", "energy")
        T = t[kt].array(library="np") if kt else [np.zeros(0)] * len(E)
        for e, z, tt in zip(E, Z, T):
            e = np.asarray(e, float) * 1e3; z = np.abs(np.asarray(z, float))
            out["bar"].append(e[z < ZEND].sum()); out["end"].append(e[z >= ZEND].sum())
            out["n"].append(len(e)); out["tim"].append(np.asarray(tt, float).sum() * 1e3)
    return {k: np.array(v) for k, v in out.items()}


args = sys.argv[1:]
samples = [(a, a) for a in args] if args else DEFAULT
res = []
for label, d in samples:
    r = load(d)
    if r is None:
        print(f"  (no files: {d})"); continue
    res.append((label, d, r))
if not res:
    sys.exit("nothing to compare")

sem = lambda x: x.std(ddof=1) / np.sqrt(len(x))
print(f"\n{'sample':32} {'cross.':>6} {'crystals':>14} {'barrel':>13} {'endcap':>13} {'hits':>6} {'timing':>12}")
for label, d, r in res:
    tot = r["bar"] + r["end"]
    print(f"{label:32} {len(tot):6d} {tot.mean():8.1f}±{sem(tot):4.1f} {r['bar'].mean():7.1f}±{sem(r['bar']):4.1f} "
          f"{r['end'].mean():7.1f}±{sem(r['end']):4.1f} {r['n'].mean():6.0f} {r['tim'].mean():6.1f}±{sem(r['tim']):4.1f}")

ref_label, _, ref = res[0]
print(f"\nratio to '{ref_label}' (paired, same crossings; ± from the event-by-event differences)")
print(f"{'sample':32} {'crystals':>13} {'barrel':>13} {'endcap':>13} {'timing':>13}")
for label, d, r in res[1:]:
    if len(r["bar"]) != len(ref["bar"]):
        print(f"{label:32}   different number of crossings ({len(r['bar'])} vs {len(ref['bar'])}): not paired"); continue
    row = []
    for key in ("tot", "bar", "end", "tim"):
        a = (r["bar"] + r["end"]) if key == "tot" else r[key]
        b = (ref["bar"] + ref["end"]) if key == "tot" else ref[key]
        m = b.mean()
        row.append(f"{a.mean()/m:7.3f}±{sem(a - b)/m:5.3f}")
    print(f"{label:32} " + " ".join(row))
