"""Energy per mrad of polar angle on the crystal-tower grid (no binning-segmentation aliasing).
towers(): group the occupied 0.5 mrad bins of the cell histogram into towers (front + rear row),
profile(): sum the energy over groups of m consecutive towers and divide by the angle they span."""
import json
import numpy as np

def load(path):
    R = json.load(open(path)); lo, hi, n = R["fine_edges_rad"]
    fe = 1e3 * np.linspace(lo, hi, int(n) + 1)
    return R, fe, 0.5 * (fe[1:] + fe[:-1])

def towers(Nf, fc, gap=4):
    """-> list of index arrays (fine bins of one tower), tower centres [mrad]. A new tower starts where
    the distance to the previous occupied fine bin is >= gap bins (2.0 mrad): front and rear rows of
    one tower are 0.5-1.5 mrad apart, towers 3.5 mrad."""
    occ = np.nonzero(Nf)[0]
    idx = np.split(occ, np.nonzero(np.diff(occ) >= gap)[0] + 1)
    cen = np.array([np.average(fc[g], weights=Nf[g]) for g in idx])
    return idx, cen

def profile(Ef, idx, cen, m=3):
    """-> group centres, group widths [mrad], energy per mrad."""
    Et = np.array([Ef[g].sum() for g in idx])
    edges = np.concatenate([[cen[0] - 0.5 * (cen[1] - cen[0])], 0.5 * (cen[1:] + cen[:-1]), [cen[-1] + 0.5 * (cen[-1] - cen[-2])]])
    k = (len(cen) // m) * m
    ge = edges[: k + 1 : m]
    E = Et[:k].reshape(-1, m).sum(1)
    return 0.5 * (ge[1:] + ge[:-1]), np.diff(ge), E / np.diff(ge)
