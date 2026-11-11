#!/usr/bin/env python3
"""FCC-ee collision parameters, FSR 2025 baseline (GHC lattice).

Source: FCC Feasibility Study Report Vol. 2 (arXiv:2505.00274v1), Table 1.2
"FCC-ee collider parameters for the GHC lattice".  Values are copied from the
table; everything under "derived" is computed here.
Run:  python fsr2025_params.py   -> prints the table, writes fsr2025_params.json
"""
import json, math

C, RE, ALPHA, ME_GEV = 299792458.0, 2.8179403262e-15, 1 / 137.035999084, 0.51099895e-3
CIRC_M, THETA_HALF = 90658.509, 15e-3            # circumference; half crossing angle [rad]

TABLE = {  # beam energy GeV, bunches, N, eps_x [m], eps_y [m], beta_x* [m], beta_y* [m], sigma_z SR/BS [m], sigma_delta BS, lumi/IP [1e34]
    "Z":  dict(ecm=91.2,  ebeam=45.6,  nb=11200, N=2.18e11, epsx=0.71e-9, epsy=2.1e-12,  betx=0.110, bety=0.7e-3, sigz_sr=5.15e-3, sigz=15.2e-3, sigd=0.115e-2, lumi=144.0),
    "WW": dict(ecm=160.0, ebeam=80.0,  nb=1856,  N=1.38e11, epsx=2.16e-9, epsy=2.0e-12,  betx=0.220, bety=1.0e-3, sigz_sr=3.46e-3, sigz=5.28e-3, sigd=0.105e-2, lumi=20.0),
    "ZH": dict(ecm=240.0, ebeam=120.0, nb=300,   N=1.69e11, epsx=0.66e-9, epsy=1.0e-12,  betx=0.240, bety=1.0e-3, sigz_sr=3.26e-3, sigz=5.59e-3, sigd=0.176e-2, lumi=7.5),
    "tt": dict(ecm=365.0, ebeam=182.5, nb=60,    N=1.58e11, epsx=1.65e-9, epsy=1.32e-12, betx=0.900, bety=1.4e-3, sigz_sr=1.91e-3, sigz=2.34e-3, sigd=0.186e-2, lumi=1.45),
}

def derive(p):
    g = p["ebeam"] / ME_GEV
    sx, sy = math.sqrt(p["epsx"] * p["betx"]), math.sqrt(p["epsy"] * p["bety"])
    phi = THETA_HALF * p["sigz"] / sx                       # Piwinski angle
    lum = p["N"] ** 2 * p["nb"] * (C / CIRC_M) / (4 * math.pi * sx * sy * math.sqrt(1 + phi ** 2)) * 1e-4
    ups = 5 / 6 * RE ** 2 * g * p["N"] / (ALPHA * p["sigz"] * (sx + sy))   # average beamstrahlung parameter
    return dict(gamma=g, sigx=sx, sigy=sy, piwinski=phi,
                overlap_len=sx / THETA_HALF,                # length of the collision region along z
                sigxp=math.sqrt(p["epsx"] / p["betx"]), sigyp=math.sqrt(p["epsy"] / p["bety"]),
                lumi_geom_1e34=lum / 1e34, upsilon_avg=ups,
                hourglass=p["bety"] / (sx / THETA_HALF))

if __name__ == "__main__":
    out = {"source": "FSR Vol. 2 (arXiv:2505.00274v1) Table 1.2, GHC lattice",
           "theta_half_rad": THETA_HALF, "points": {}}
    print(f"{'':4}{'sigx um':>9}{'sigy nm':>9}{'sigz mm':>9}{'Piwinski':>9}{'overlap mm':>11}{'bety/ovl':>9}"
          f"{'L geom':>8}{'L FSR':>7}{'Upsilon':>10}")
    for k, p in TABLE.items():
        d = derive(p); out["points"][k] = {**p, "derived": d}
        print(f"{k:4}{d['sigx']*1e6:9.2f}{d['sigy']*1e9:9.1f}{p['sigz']*1e3:9.2f}{d['piwinski']:9.2f}"
              f"{d['overlap_len']*1e3:11.3f}{d['hourglass']:9.2f}{d['lumi_geom_1e34']:8.1f}{p['lumi']:7.1f}{d['upsilon_avg']:10.2e}")
    json.dump(out, open("fsr2025_params.json", "w"), indent=1)
