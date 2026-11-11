#!/usr/bin/env python3
"""Transfer test on the measured inner-loop landscapes (campaign v3).

    python money_transfer.py [money_dump.json]   -> money_transfer.json + console table

For every barrel sample, working point and background level (1x, 3x), a surrogate of the
inner-loop score is built from the points the real optimizer evaluated (about 600) plus the
12 x 25 grid, by linear interpolation in (R, log10 E_thr). Four strategies are run on it:

  de_lin   differential evolution as in the inner loop (R in [0,300] mm, E_thr in [0,0.6] GeV linear)
  de_log   the same, with the threshold searched in log10 (1e-5 .. 0.6 GeV)
  warm_nm  Nelder-Mead in (R, log E_thr) started at the optimum of an already-solved point
  warm_de  differential evolution (log), initial population drawn around that optimum

Warm starts come from (a) the lower background level of the same working point (0x -> 1x, 1x -> 3x),
or (b) another working point with about the same background energy (Z 3x -> ZH 1x / tt 1x,
ZH 3x -> tt 3x). Metric: evaluations until the best score so far is within EPS of the surrogate
maximum. The recorded history of the real run (de_lin, one seed) is reported alongside.
"""
import json, os, sys, warnings
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import differential_evolution, minimize

warnings.filterwarnings("ignore")
SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/MDI/money_dump.json"
D = json.load(open(SRC))
BASES = ["g100MeV_fixed", "g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed", "g5000MeV_fixed", "e1GeV_fixed"]
PTS = ["Z", "WW", "ZH", "tt"]
LVL = {0.0: "0x", 1.0: "1x", 3.0: "3x"}
EPS = float(os.environ.get("EPS", "0.02"))
NSEED = int(os.environ.get("NSEED", "30"))
BUDGET = 640
LE0, LE1 = -5.0, np.log10(0.6)


def to_uv(R, E):
    return np.column_stack([np.asarray(R) / 300.0, (np.log10(np.maximum(E, 1e-5)) - LE0) / (LE1 - LE0)])


class Surrogate:
    """Kernel-smoothed (Gaussian, h = 0.025 in the unit square) average of the evaluated points:
    the measured scores carry ~2-4 % statistical noise, which a linear interpolant would keep as
    spurious local maxima."""
    H = 0.025

    def __init__(self, rec):
        R, E = np.array(rec["grid_R_cluster_mm"]), np.array(rec["grid_E_threshold_GeV"])
        gR, gE = np.meshgrid(R, E, indexing="ij")
        hx, hf = np.array(rec["opt_history_x"]), np.array(rec["opt_history_f"])
        self.P = np.vstack([to_uv(gR.ravel(), gE.ravel()), to_uv(hx[:, 0], hx[:, 1])])
        self.F = np.concatenate([np.array(rec["score"]).ravel(), hf])
        # reference maximum: the smoothed score at the points actually evaluated (no extrapolation
        # into regions the inner loop never sampled)
        f = np.concatenate([self(self.P[i:i + 2000]) for i in range(0, len(self.P), 2000)])
        self.fmax = float(f.max()); self.best_uv = self.P[np.argmax(f)]
        self.hist_f = hf
        self.hist_s = self(to_uv(hx[:, 0], hx[:, 1]))          # smoothed value along the real run

    def __call__(self, uv):
        uv = np.atleast_2d(uv)
        d2 = ((uv[:, None, :] - self.P[None, :, :]) ** 2).sum(-1)
        w = np.exp(-0.5 * d2 / self.H ** 2) + 1e-300
        return (w * self.F).sum(1) / w.sum(1)


def run(strategy, S, seed, start_uv=None):
    rng = np.random.default_rng(seed)
    trace = []

    def f_uv(uv):
        v = S(uv); trace.extend(np.atleast_1d(v).tolist()); return v

    if strategy == "de_lin":         # linear threshold, as in the inner loop
        def obj(x):
            x = np.atleast_2d(x.T) if x.ndim == 2 else x[None]
            return -f_uv(to_uv(x[:, 0], x[:, 1]))
        differential_evolution(obj, [(0, 300), (0, 0.6)], popsize=15, maxiter=20, tol=0, polish=False,
                               seed=seed, init="latinhypercube", vectorized=True, updating="deferred")
    elif strategy == "de_log":
        def obj(x):
            x = np.atleast_2d(x.T) if x.ndim == 2 else x[None]
            return -f_uv(x)
        differential_evolution(obj, [(0, 1), (0, 1)], popsize=15, maxiter=20, tol=0, polish=False,
                               seed=seed, init="latinhypercube", vectorized=True, updating="deferred")
    elif strategy == "warm_de":
        pop = np.clip(start_uv + rng.normal(0, [0.1, 0.08], size=(30, 2)), 0, 1); pop[0] = start_uv
        def obj(x):
            x = np.atleast_2d(x.T) if x.ndim == 2 else x[None]
            return -f_uv(x)
        differential_evolution(obj, [(0, 1), (0, 1)], popsize=15, maxiter=20, tol=0, polish=False,
                               seed=seed, init=pop, vectorized=True, updating="deferred")
    elif strategy == "warm_nm":
        x0 = np.array(start_uv, float); first = True
        while len(trace) < BUDGET:
            if not first:                                        # restart near the best point so far
                x0 = np.clip(best_x + rng.normal(0, [0.08, 0.06]), 0, 1)
            first = False
            res = minimize(lambda x: -f_uv(np.clip(x, 0, 1))[0], x0, method="Nelder-Mead",
                           options=dict(maxfev=120, xatol=1e-3, fatol=1e-4,
                                        initial_simplex=np.clip(x0 + np.array([[0, 0], [0.12, 0], [0, 0.1]]), 0, 1)))
            best_x = np.clip(res.x, 0, 1)
    tr = np.maximum.accumulate(np.array(trace[:BUDGET]))
    return tr


def evals_to(tr, target):
    hit = np.nonzero(tr >= target)[0]
    return int(hit[0]) + 1 if len(hit) else None


def sources():
    """(target_key, source_key, kind) with keys (pt, base, level)."""
    out = []
    for b in BASES:
        for p in PTS:
            out.append(((p, b, "1x"), (p, b, "0x"), "background step"))
            out.append(((p, b, "3x"), (p, b, "1x"), "background step"))
        out += [(("ZH", b, "1x"), ("Z", b, "3x"), "matched background"),
                (("tt", b, "1x"), ("Z", b, "3x"), "matched background"),
                (("tt", b, "3x"), ("ZH", b, "3x"), "matched background")]
    return out


# ---------- surrogates ----------
SUR = {}
for p in PTS:
    for b in BASES:
        for f, rec in D[f"money_v3_{p}_{b}"].items():
            SUR[(p, b, LVL[float(rec["geom_mdi_x"])])] = (Surrogate(rec), rec)

def one_row(args):
    tgt, src, kind = args
    S, rec = SUR[tgt]; Ssrc, rsrc = SUR[src]
    start = to_uv([rsrc["best_R_cluster_mm"]], [rsrc["best_E_threshold_GeV"]])[0]
    target = (1 - EPS) * S.fmax
    row = {"target": list(tgt), "source": list(src), "kind": kind, "fmax": S.fmax,
           "start_value_frac": float(S(start)[0] / S.fmax),
           "real_run": evals_to(np.maximum.accumulate(S.hist_s), target)}
    for st in ("de_lin", "de_log", "warm_nm", "warm_de"):
        if kind == "matched background" and st in ("de_lin", "de_log"):
            continue                                   # cold strategies do not depend on the source
        n = [evals_to(run(st, S, 1000 + s, start), target) for s in range(NSEED)]
        row[st] = n
    return row


def med(v):
    v = [x if x is not None else BUDGET + 1 for x in v]; return f"{int(np.median(v)):4d} ({sum(x <= BUDGET for x in v)}/{len(v)})"


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPROC", "4"))) as pool:
        res = pool.map(one_row, sources())
    for row in res:
        tgt, src = row["target"], row["source"]
        print(f"{tgt[0]:2} {tgt[1]:15} {tgt[2]} <- {src[0]:2} {src[2]}  start {row['start_value_frac']:.3f}  real {row['real_run']}  "
              + "  ".join(f"{st} {med(row[st])}" for st in ("de_lin", "de_log", "warm_nm", "warm_de") if st in row))

    json.dump({"eps": EPS, "nseed": NSEED, "budget": BUDGET, "rows": res}, open("money_transfer.json", "w"))
    print("wrote money_transfer.json")
