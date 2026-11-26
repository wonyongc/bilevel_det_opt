"""Hit-level paired mixing for the money runs.

Every intensity uses the SAME 0x signal events; background-only hits
(sim_outputs/bkgonly_{1x,3x}) are added crystal-by-crystal by cellID.
Geant4 MainEdep deposits add linearly, so this equals a joint simulation,
but with identical signal showers at every intensity -> shower-to-shower
fluctuations cancel in the 0x/1x/3x comparison.
"""
import glob, json, os, subprocess, sys, types
import numpy as np
sys.path.insert(0, os.path.expanduser("~/bilevel_det_opt"))
from bilevel_opt.edm import SimCalorimeterHit, SimCalorimeterHitCollection

FIELDS = ("cellID", "energy", "position.x", "position.y", "position.z")


def ensure_merged(d):
    merged = os.path.join(d, "merged.root")
    if not os.path.exists(merged):
        files = sorted(glob.glob(os.path.join(d, "events_*.root")))
        assert files, f"no events_*.root in {d}"
        print(f"[mix] hadd {len(files)} files -> {merged}")
        subprocess.run(["hadd", "-f", merged] + files, check=True)
    return merged


def read_calo(path, bname):
    import uproot
    t = uproot.open(path + ":events")
    keys = [f"{bname}.{f}" for f in FIELDS]
    a = t.arrays(keys, library="np")
    return [a[k] for k in keys]


def mixed_events(sig_path, bkg_path, bname, max_events=None):
    S = read_calo(sig_path, bname)
    B = read_calo(bkg_path, bname) if bkg_path else None
    n = len(S[0]) if not max_events else min(len(S[0]), max_events)
    nb = len(B[0]) if B is not None else 0
    out = []
    for i in range(n):
        cols = [np.asarray(S[k][i]) for k in range(5)]
        if B is not None:
            j = i % nb
            cols = [np.concatenate([cols[k], np.asarray(B[k][j])]) for k in range(5)]
        cid = cols[0].astype(np.uint64)
        e = cols[1].astype(float)
        u, first, inv = np.unique(cid, return_index=True, return_inverse=True)
        esum = np.bincount(inv.ravel(), weights=e, minlength=len(u))
        px, py, pz = (cols[k].astype(float) for k in (2, 3, 4))
        hits = [SimCalorimeterHit(types.SimpleNamespace(
                    cellID=int(u[k]), energy=float(esum[k]),
                    position=types.SimpleNamespace(x=float(px[first[k]]),
                                                   y=float(py[first[k]]),
                                                   z=float(pz[first[k]]))))
                for k in range(len(u))]
        out.append({bname: SimCalorimeterHitCollection(hits)})
    return out


def spec_for(lab, samples, outdir, run):
    sig = ensure_merged(samples["0x"][0])
    bkg = None if lab == "0x" else ensure_merged(
        os.path.join(run, "sim_outputs", "bkgonly_" + lab))
    p = os.path.join(outdir, f"mix_{lab}.mixspec.json")
    json.dump({"signal": sig, "bkg": bkg}, open(p, "w"), indent=1)
    return p


def install(il):
    orig = il.load_events

    def load_events(path, tree_name, branches, max_events):
        if str(path).endswith(".mixspec.json"):
            assert len(branches) == 1, "mixing supports one hit branch"
            spec = json.load(open(path))
            print(f"[mix] {os.path.basename(path)}: signal={spec['signal']} bkg={spec['bkg']}")
            return mixed_events(spec["signal"], spec["bkg"], branches[0]["name"], max_events)
        return orig(path, tree_name, branches, max_events)

    il.load_events = load_events
