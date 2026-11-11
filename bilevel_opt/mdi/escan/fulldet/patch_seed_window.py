#!/usr/bin/env python3
"""Two optional algorithm parameters for the inner-loop clustering, for the full detectors.

    source /global/cfs/cdirs/m4956/sapta/venvs/bilevel/bin/activate     (after key4hep_setup)
    python fulldet/patch_seed_window.py          # idempotent; patches the package `import bilevel_opt` finds

SeededRadiusCoG gets, with defaults that reproduce the original behaviour exactly:
  seed_window_mm (default 0): 0 = the seed is the highest-energy hit of the event (original).
      > 0 = the seed is the hit with the largest energy SUM within seed_window_mm of it (transverse
      distance on the sphere through the hits' median radius, computed with a k-d tree). An
      isolated energetic background hit (a pair particle crossing one silicon cell) is then not
      a seed, a shower spread over many cells is. The seed does not depend on (R, E_thr), so it
      is found once per event and cached on the hit collection (keyed by the window).
  metric ("xy" or "pointing", default "xy"): the distance that R_cluster_mm cuts on. "xy" is the
      original transverse-plane distance; "pointing" is the distance in the plane perpendicular to
      the line from the IP to the seed (every hit scaled radially onto the sphere through the
      seed), so the depth of a barrel stack (40 silicon layers over 20 cm in r for CLD) does not
      count towards R, and R means the same thing in the barrel and in the endcaps.
Why: in the first CLD loops at 100-200 MeV the seed was the maximum background hit at 1x/3x
(the shower's largest cell holds ~0.3 MeV, energetic background hits exceed 1 MeV), so the
"cluster" measured an extreme-value statistic of the background, and the barrel optimum R of
200-300 mm was the depth of the stack projected on x-y.
run_money.py passes them from MONEY_SEEDWIN and MONEY_METRIC (patch_money_branch.py).
Copy the patched file into ~/bilevel_det_opt before committing, as for the other patches.
"""
import os, sys
import bilevel_opt
pkg = os.path.dirname(bilevel_opt.__file__)
alg = os.path.join(pkg, "algorithms", "seeded_radius_cog.py")
print("bilevel_opt package:", pkg)

CLS = '''class SeededRadiusCoG:
    """
    Clustering: find the seed, collect hits within radius R of it that pass an energy
    threshold, compute the energy-weighted center-of-gravity.

    Constructor params (from config algorithm.params):
        branch (str)           — which branch to cluster on
        seed_window_mm (float) — 0 (default): seed = highest-energy hit. > 0: seed = the hit
                                 with the largest energy sum within this transverse window
                                 (follows a shower, not an isolated energetic hit); independent
                                 of the optimized params, cached per event
        metric (str)           — "xy" (default): R cuts on the x-y distance to the seed;
                                 "pointing": on the distance perpendicular to the IP->seed
                                 line (depth along the shower axis does not count)

    Optimized params:
        R_cluster_mm (float)     — radius cut around seed
        E_threshold_GeV (float)  — minimum energy per hit (0 = no threshold)
    """
    name = "seeded_radius_cog"

    def __init__(self, branch: str, seed_window_mm: float = 0.0, metric: str = "xy"):
        self.branch = branch
        self.seed_window_mm = float(seed_window_mm)
        if metric not in ("xy", "pointing"):
            raise ValueError(f"metric must be 'xy' or 'pointing', got {metric!r}")
        self.metric = metric

    @staticmethod
    def _geom(ev):
        """Positions (n,3) and radii (n,), cached on the collection."""
        g = getattr(ev, "_geom", None)
        if g is None:
            P = np.stack([np.asarray(ev.x, float), np.asarray(ev.y, float), np.asarray(ev.z, float)], axis=1)
            r = np.linalg.norm(P, axis=1)
            r[r == 0.0] = 1.0
            g = (P, r)
            ev._geom = g
        return g

    def _seed(self, ev) -> int:
        cache = getattr(ev, "_seed_cache", None)
        if cache is None:
            cache = ev._seed_cache = {}
        s = cache.get(self.seed_window_mm)
        if s is not None:
            return s
        if self.seed_window_mm <= 0.0 or ev.E.size < 2:
            s = int(np.argmax(ev.E))
        else:
            from scipy.spatial import cKDTree
            P, r = self._geom(ev)
            R0 = float(np.median(r))
            Q = P * (R0 / r)[:, None]                      # all hits on one sphere: angular window
            nb = cKDTree(Q).query_ball_point(Q, self.seed_window_mm)
            E = np.asarray(ev.E, float)
            local = np.fromiter((E[idx].sum() for idx in nb), float, count=len(nb))
            s = int(np.argmax(local))
        cache[self.seed_window_mm] = s
        return s

    def reconstruct(self, events, params: Dict[str, float]) -> List[ClusterReco]:
        R2 = float(params["R_cluster_mm"]) ** 2
        E_thr = float(params.get("E_threshold_GeV", 0.0))
        out: List[ClusterReco] = []
        for event in events:
            ev = event[self.branch]
            if ev.E.size == 0:
                out.append(ClusterReco(0.0, np.nan, np.nan, 0))
                continue
            seed = self._seed(ev)
            if self.metric == "pointing":
                P, r = self._geom(ev)
                Q = P * (r[seed] / r)[:, None]
                d2 = np.sum((Q - Q[seed]) ** 2, axis=1)
            else:
                dx = ev.x - ev.x[seed]
                dy = ev.y - ev.y[seed]
                d2 = dx * dx + dy * dy
            sel = (d2 <= R2) & (ev.E >= E_thr)
            if not np.any(sel):
                out.append(ClusterReco(0.0, np.nan, np.nan, 0))
                continue
            Esel = ev.E[sel]
            Ecl = float(np.sum(Esel))
            nh = int(Esel.size)
            if Ecl <= 0.0:
                out.append(ClusterReco(Ecl, np.nan, np.nan, nh))
                continue
            xc = float(np.sum(Esel * ev.x[sel]) / Ecl)
            yc = float(np.sum(Esel * ev.y[sel]) / Ecl)
            out.append(ClusterReco(Ecl, xc, yc, nh))
        return out


'''

s = open(alg).read()
if "seed_window_mm" in s:
    print("seeded_radius_cog.py: seed_window_mm / metric already present")
else:
    i = s.find("class SeededRadiusCoG:")
    j = s.find("class SNREnergyScore:")
    if i < 0 or j < 0 or j < i:
        sys.exit("seeded_radius_cog.py: unexpected layout; send me the file")
    open(alg, "w").write(s[:i] + CLS + s[j:])
    print("seeded_radius_cog.py: SeededRadiusCoG replaced (seed_window_mm, metric)")

# self-test: a shower of 60 hits around (2150, 0, 0) spread over 20 cm of depth, plus 2000
# background hits of 4 keV and one isolated 2 MeV background hit far away
import importlib, types
import numpy as np
import bilevel_opt.algorithms.seeded_radius_cog as M
importlib.reload(M)
from bilevel_opt.edm import SimCalorimeterHit, SimCalorimeterHitCollection
rng = np.random.default_rng(0)
def hit(x, y, z, e):
    return SimCalorimeterHit(types.SimpleNamespace(cellID=0, energy=e, position=types.SimpleNamespace(x=x, y=y, z=z)))
hits = [hit(2150 + d, rng.normal(0, 6), rng.normal(0, 6), 0.0003 * rng.exponential()) for d in rng.uniform(0, 200, 60)]
hits += [hit(x, y, z, 4e-6) for x, y, z in zip(rng.uniform(-2300, 2300, 2000), rng.uniform(-2300, 2300, 2000), rng.uniform(-2000, 2000, 2000))]
hits += [hit(-1000, 1900, 500, 0.002)]
ev = {"B": SimCalorimeterHitCollection(hits)}
Esh = sum(h.E for h in hits[:60])
old = M.SeededRadiusCoG("B")
new = M.SeededRadiusCoG("B", seed_window_mm=15.0, metric="pointing")
r_old = old.reconstruct([ev], {"R_cluster_mm": 20.0, "E_threshold_GeV": 0.0})[0]
r_new = new.reconstruct([ev], {"R_cluster_mm": 20.0, "E_threshold_GeV": 0.0})[0]
print(f"self-test: shower energy {1e3*Esh:.3f} MeV; R = 20 mm: original seed/metric collects "
      f"{1e3*r_old.E_cluster:.3f} MeV at x = {r_old.x_cog:.0f} (the 2 MeV background hit), "
      f"windowed seed + pointing metric collects {1e3*r_new.E_cluster:.3f} MeV at x = {r_new.x_cog:.0f} "
      f"(expect ~{1e3*Esh:.3f} at x ~ 2250)")
ok = os.path.join(os.environ.get("RUN", "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"), "fulldet", "seed_window.ok")
try:
    open(ok, "w").write("seed_window v1 " + pkg + "\n"); print("wrote", ok, "(campaign_v4.sh preflight marker)")
except OSError as e:
    print("marker not written:", e)
