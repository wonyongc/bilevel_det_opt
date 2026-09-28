"""SeededRadiusCoG  algorithm and SNREnergyScore scorer.

Toy algorithm for demonstrating the bilevel optimization pipeline.

Parameters optimized by the inner loop:
    R_cluster_mm     — clustering radius around the seed hit [mm]
    E_threshold_GeV  — minimum hit energy to include in cluster [GeV]

Physics motivation: R_cluster_mm controls how much of the shower is captured.
Too small misses energy; too large picks up noise from neighboring showers.
E_threshold_GeV suppresses low-energy noise hits and shower tails that
degrade the energy-weighted position and energy resolution. The optimal
(R, E_threshold) pair depends on the detector granularity (geometry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class ClusterReco:
    E_cluster: float
    x_cog: float
    y_cog: float
    n_hits: int


class SeededRadiusCoG:
    """
    Clustering: find highest-energy hit (seed), collect hits within radius R
    that pass an energy threshold, compute energy-weighted center-of-gravity.

    Constructor params (from config algorithm.params):
        branch (str) — which branch to cluster on

    Optimized params:
        R_cluster_mm (float)     — radius cut around seed
        E_threshold_GeV (float)  — minimum energy per hit (0 = no threshold)
    """
    name = "seeded_radius_cog"

    def __init__(self, branch: str):
        self.branch = branch

    def reconstruct(self, events, params: Dict[str, float]) -> List[ClusterReco]:
        R2 = float(params["R_cluster_mm"]) ** 2
        E_thr = float(params.get("E_threshold_GeV", 0.0))
        out: List[ClusterReco] = []
        for event in events:
            ev = event[self.branch]
            if ev.E.size == 0:
                out.append(ClusterReco(0.0, np.nan, np.nan, 0))
                continue
            seed = int(np.argmax(ev.E))
            dx = ev.x - ev.x[seed]
            dy = ev.y - ev.y[seed]
            sel = ((dx * dx + dy * dy) <= R2) & (ev.E >= E_thr)
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


class SNREnergyScore:
    """Score = mean( E_cluster / (sigma0 * sqrt(N_hits)) ) over events."""
    name = "snr_energy"

    def __init__(self, sigma0: float):
        self.sigma0 = float(sigma0)

    def __call__(self, events, reco) -> float:
        vals = [
            r.E_cluster / (self.sigma0 * math.sqrt(r.n_hits))
            for r in reco if r.n_hits > 0
        ]
        return float(np.mean(vals)) if vals else float("-inf")


class SNREnergyAccScore:
    """Score = (1/N_events) * sum( E_cluster / (sigma0*sqrt(n_hits)) ).

    Same per-event statistic as snr_energy, but averaged over ALL events,
    with empty events contributing zero -- i.e. snr_energy multiplied by
    the acceptance.  Removes the degenerate maximum at extreme thresholds
    that keeps only a handful of seed-energy-tail events (acc ~ 1e-3 at
    the snr_energy optimum).  All-empty configurations score 0.0 (not
    -inf), which keeps the DE landscape finite and smooth.
    """
    name = "snr_energy_acc"

    def __init__(self, sigma0: float):
        self.sigma0 = float(sigma0)

    def __call__(self, events, reco) -> float:
        if not reco:
            return float("-inf")
        total = sum(
            r.E_cluster / (self.sigma0 * math.sqrt(r.n_hits))
            for r in reco if r.n_hits > 0
        )
        return float(total / len(reco))


class EResolutionAccScore:
    """Score = acceptance * mu / sigma_core of the E_cluster distribution.

    Truth-anchored for monochromatic gun samples: mu = median E_cluster,
    sigma_core = 0.5*(q84 - q16).  Unlike snr_energy(_acc), collecting
    background energy does not pay: it broadens the distribution and is
    penalized through sigma.  Guards: <10 surviving events or a degenerate
    zero-width spike score 0.0 (closes the tail-cherry-picking hole).
    sigma0 accepted for config compatibility; unused.
    """
    name = "eres_acc"

    def __init__(self, sigma0: float = 1.0):
        self.sigma0 = float(sigma0)

    def __call__(self, events, reco) -> float:
        E = [r.E_cluster for r in reco if r.n_hits > 0 and r.E_cluster > 0]
        if len(reco) == 0 or len(E) < 10:
            return 0.0
        a = np.asarray(E)
        acc = len(E) / len(reco)
        mu = float(np.median(a))
        q16, q84 = np.percentile(a, [16.0, 84.0])
        sig = 0.5 * (q84 - q16)
        if sig <= 0.0:
            return 0.0
        return float(acc * mu / sig)


class EResolutionTrueScore:
    """Score = acceptance * E_true / sigma_core(E_cluster).

    Absolute core width normalized by the TRUE gun energy. Unlike eres_acc
    (mu/sigma), an additive background pedestal cannot raise the score: it
    shifts mu but not E_true, and its fluctuations widen sigma.  Bias
    (median/E_true) is a diagnostic, not scored: a known pedestal can be
    subtracted by calibration, its fluctuations cannot.
    """
    name = "eres_true"

    def __init__(self, sigma0: float = 1.0, E_true: float = 1.0):
        self.sigma0 = float(sigma0)
        self.E_true = float(E_true)

    def __call__(self, events, reco) -> float:
        E = [r.E_cluster for r in reco if r.n_hits > 0 and r.E_cluster > 0]
        if len(reco) == 0 or len(E) < 10:
            return 0.0
        a = np.asarray(E)
        acc = len(E) / len(reco)
        q16, q84 = np.percentile(a, [16.0, 84.0])
        sig = 0.5 * (q84 - q16)
        if sig <= 0.0:
            return 0.0
        return float(acc * self.E_true / sig)
