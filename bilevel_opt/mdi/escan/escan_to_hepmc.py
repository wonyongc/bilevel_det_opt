#!/usr/bin/env python3
"""WarpX beam-beam run -> HepMC3 background events, one event per bunch crossing.

    python escan_to_hepmc.py Z --mult 1 --nevents 2000      # -> Z/bkg_Z_1x.hepmc
    python escan_to_hepmc.py Z --mult 3 --nevents 2000      # -> Z/bkg_Z_3x.hepmc

Contents: the incoherent e+e- pairs.  Beamstrahlung photons are NOT written:
there are ~3e10 per crossing and all stay within ~0.3 mrad of the outgoing
beams (see check_yields.py), i.e. inside the beam pipe.

Normalisation: every macroparticle stands for `weighting` real particles.  An
event holds Poisson(mult * sum of weights) particles, each drawn with
probability proportional to its weight, which is the same as keeping every
macroparticle independently with probability mult * weight.  mult = 1 is one
real crossing; mult = 3 is three crossings on top of each other.

All particles are placed at the origin (they leave the simulated region within
a few mm of the collision point, well inside the beam pipe).  Momenta are in
the detector frame: z bisects the two beams, the beams lie in the x-z plane.
File layout matches what ddsim already reads in this project
(E / U GEV MM / W / P lines, no V records).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_yields import load, ME_GEV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="directory of one WarpX run (holds deck_meta.json and diags/)")
    ap.add_argument("--mult", type=float, default=1.0, help="bunch crossings per event")
    ap.add_argument("--nevents", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--emin", type=float, default=0.0, help="drop particles below this energy [GeV]")
    ap.add_argument("--out")
    a = ap.parse_args()
    run = a.run.rstrip("/")
    meta = json.load(open(os.path.join(run, "deck_meta.json")))
    tag = f"{a.mult:g}x"
    out = a.out or os.path.join(run, f"bkg_{meta['name']}_{tag}.hepmc")

    P = load(run)
    cols = {k: [] for k in ("px", "py", "pz", "w", "pdg")}
    for sp in meta["products"]:
        if sp not in P:
            continue
        for k in ("px", "py", "pz", "w"):
            cols[k].append(P[sp][k])
        cols["pdg"].append(np.full(len(P[sp]["w"]), 11 if sp.startswith("ele") else -11))
    if not cols["w"]:
        sys.exit("no pair particles found in " + run)
    px, py, pz, w, pdg = (np.concatenate(cols[k]) for k in ("px", "py", "pz", "w", "pdg"))
    E = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + ME_GEV ** 2)
    keep = E >= a.emin
    px, py, pz, w, pdg, E = px[keep], py[keep], pz[keep], w[keep], pdg[keep], E[keep]
    cdf = np.cumsum(w)
    mean_n = a.mult * cdf[-1]
    reuse = a.nevents * a.mult * float(np.max(w))
    print(f"{meta['name']}: {len(w)} pair macroparticles, {cdf[-1]:.1f} real e+/e- per crossing, "
          f"mean weight {w.mean():.4f}, max {w.max():.3f}")
    print(f"writing {a.nevents} events with {mean_n:.0f} particles on average ({a.mult:g} crossing(s) each)")
    print(f"the heaviest macroparticle is used about {reuse:.1f} times across the {a.nevents} events "
          f"({'fine' if reuse < 0.05 * a.nevents else 'HIGH: lower --pair-weight in the WarpX run'})")

    rng = np.random.default_rng(a.seed)
    ntot, etot = 0, 0.0
    with open(out, "w") as f:
        f.write("HepMC::Version 3.02.05\nHepMC::Asciiv3-START_EVENT_LISTING\n")
        for ev in range(a.nevents):
            n = rng.poisson(mean_n)
            i = np.searchsorted(cdf, rng.random(n) * cdf[-1])
            f.write(f"E {ev} 1 {n}\nU GEV MM\nW 1.0000000000000000000000e+00\n")
            lines = [f"P {k + 1} 0 {pdg[j]} {px[j]:.10e} {py[j]:.10e} {pz[j]:.10e} {E[j]:.10e} {ME_GEV:.10e} 1\n"
                     for k, j in enumerate(i)]
            f.write("".join(lines))
            ntot += n; etot += float(E[i].sum())
        f.write("HepMC::Asciiv3-END_EVENT_LISTING\n")
    print(f"wrote {out}: {ntot / a.nevents:.0f} particles and {etot / a.nevents:.1f} GeV per event, "
          f"{os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
