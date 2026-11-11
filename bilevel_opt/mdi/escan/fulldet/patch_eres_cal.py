#!/usr/bin/env python3
"""Add the calibrated resolution score "eres_cal" to the installed bilevel_opt package.

    source $HOME/key4hep_setup.sh -r 2024-10-03 && source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate
    python fulldet/patch_eres_cal.py          # idempotent; patches the package that `import bilevel_opt` finds

Why: eres_true = acceptance * E_true / sigma_core rewards the smallest ABSOLUTE core width of
the cluster-energy distribution. In a homogeneous crystal calorimeter the cluster energy is the
gun energy, so that is the resolution. In a sampling calorimeter the cluster energy is the
active-layer energy (about 2 % of E_true in the CLD silicon), and the smallest absolute width
comes from a single cell that collects a tenth of the shower (CLD, 200 MeV, 3x: R = 3.8 mm,
"score" 443). eres_cal = min(mu, E_cal) / sigma_core over ALL events (an event without a cluster
enters at 0), with mu the median cluster energy and E_cal the median energy the calorimeter
deposits for that photon without background (fulldet/calib_fulldet.py). Below containment it is
the relative resolution mu/sigma, so a small cluster pays; above it a background pedestal cannot
raise the numerator, as in eres_true. v1 (survivors only, times acceptance) was still beaten by a
one-cell cluster with a ~1 MeV threshold at 100-200 MeV: the survivors form a truncated, narrow
distribution and the linear acceptance factor is too weak a penalty. With the empty events kept
at zero, rejecting events can only widen the distribution.
For the crystals E_cal = E_true and the two scores coincide. Config parameter name stays
"E_true" (MONEY_ETRUE), now carrying E_cal.
Copy the patched files into ~/bilevel_det_opt before committing, as for the other patches.
"""
import os, sys
import bilevel_opt
pkg = os.path.dirname(bilevel_opt.__file__)
alg = os.path.join(pkg, "algorithms", "seeded_radius_cog.py")
reg = os.path.join(pkg, "algorithms", "registry.py")
print("bilevel_opt package:", pkg)

CLS = '''

class EResolutionCalScore:
    """eres_cal v2: score = min(mu, E_cal) / sigma_core over ALL events.

    E_cluster of every event enters, with 0 for an event that produced no cluster;
    mu = median, sigma_core = 0.5*(q84 - q16) of that full array; E_cal = the median energy
    the calorimeter deposits for this gun energy without background (passed as E_true).
    Keeping the empty events at zero closes the threshold hole of the survivors-only
    definition (v1, acceptance * mu/sigma of the survivors): a per-hit threshold that
    rejects events can only widen the distribution (below 84 % acceptance q16 sits at 0),
    so a threshold pays only by removing background hits. Below containment (mu < E_cal)
    the score is the relative resolution, so a cluster that collects a fraction of the
    shower is penalized; at and above containment it is E_cal/sigma, so an additive
    background pedestal cannot raise it (eres_true's property). For a homogeneous
    calorimeter E_cal = E_true. Guards: < 10 events, mu <= 0 or zero width score 0.0.
    """
    name = "eres_cal"

    def __init__(self, sigma0: float = 1.0, E_true: float = 1.0):
        self.sigma0 = float(sigma0)
        self.E_cal = float(E_true)

    def __call__(self, events, reco) -> float:
        if len(reco) < 10:
            return 0.0
        a = np.asarray([r.E_cluster if (r.n_hits > 0 and r.E_cluster > 0) else 0.0 for r in reco], float)
        mu = float(np.median(a))
        q16, q84 = np.percentile(a, [16.0, 84.0])
        sig = 0.5 * (q84 - q16)
        if sig <= 0.0 or mu <= 0.0:
            return 0.0
        return float(min(mu, self.E_cal) / sig)
'''

s = open(alg).read()
if "eres_cal v2" in s:
    print("seeded_radius_cog.py: eres_cal v2 already present")
elif "class EResolutionCalScore" in s:          # v1 (survivors only) -> replace the class, it is the last in the file
    k = s.index("class EResolutionCalScore")
    open(alg, "w").write(s[:k].rstrip("\n") + "\n" + CLS)
    print("seeded_radius_cog.py: EResolutionCalScore v1 replaced by v2")
else:
    if "class EResolutionTrueScore" not in s or "import numpy as np" not in s:
        sys.exit("seeded_radius_cog.py does not look like the expected file; send me its head")
    open(alg, "w").write(s.rstrip("\n") + "\n" + CLS)
    print("seeded_radius_cog.py: EResolutionCalScore added")

r = open(reg).read()
if "EResolutionCalScore" in r and '"eres_cal"' in r:
    print("registry.py: eres_cal already registered")
else:
    old_imp = "EResolutionAccScore, EResolutionTrueScore)"
    old_map = '    "eres_true": EResolutionTrueScore,\n'
    if r.count(old_imp) != 1 or r.count(old_map) != 1:
        sys.exit("registry.py: unexpected layout; send me the file")
    r = r.replace(old_imp, "EResolutionAccScore, EResolutionTrueScore, EResolutionCalScore)")
    r = r.replace(old_map, old_map + '    "eres_cal": EResolutionCalScore,\n')
    open(reg, "w").write(r)
    print("registry.py: eres_cal registered")

# verify
import importlib
import bilevel_opt.algorithms.registry as R
importlib.reload(sys.modules["bilevel_opt.algorithms.seeded_radius_cog"]); importlib.reload(R)
sc = R.SCORES["eres_cal"](sigma0=1.0, E_true=0.004)
class _R:  # minimal reco stand-in
    def __init__(self, e): self.n_hits = 1; self.E_cluster = e
import numpy as np
rng = np.random.default_rng(1)
vals = [sc(None, [_R(e) for e in rng.normal(m, 0.0005, 500)]) for m in (0.001, 0.004, 0.010)]
full = rng.normal(0.004, 0.0005, 500); cut = np.where(full > np.percentile(full, 60), full, 0.0)
vals.append(sc(None, [_R(e) for e in cut]))
print("self-test (E_cal 4 MeV, sigma 0.5 MeV): mu = 1, 4, 10 MeV -> %.1f %.1f %.1f (expect ~2, ~8, ~8); "
      "40 %% acceptance by threshold -> %.1f (expect < 2)" % tuple(vals))
ok = os.path.join(os.environ.get("RUN", "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"), "fulldet", "eres_cal.ok")
try:
    open(ok, "w").write("eres_cal v2 " + pkg + "\n"); print("wrote", ok, "(campaign_v4.sh preflight marker)")
except OSError as e:
    print("marker not written:", e)
