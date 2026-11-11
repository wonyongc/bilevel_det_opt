#!/usr/bin/env python3
"""Dump events from a HepMC3 ASCII file into the explorer's event-display JSON.

usage:  python dump_events_hepmc.py <file.hepmc> [--events 0,1,2] [--name "Pair background, Z 1x"]
                                   [--kind background|signal|probe] [--out events.json]

Format written (what the explorer's "load" button reads):
  {"events": [{"id": ..., "name": ..., "kind": ..., "desc": ...,
               "particles": [[pdg, px, py, pz, x0, y0, z0], ...]}]}      momenta in GeV, vertices in mm
Only final-state particles (status 1) are kept; neutrinos are dropped. Momentum units are read
from the file's "U" line (GEV/MEV), lengths from the same line (MM/CM).
"""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("hepmc"); ap.add_argument("--events", default="0")
ap.add_argument("--name", default=None); ap.add_argument("--kind", default="background")
ap.add_argument("--out", default="events.json"); ap.add_argument("--maxp", type=int, default=20000)
a = ap.parse_args()
want = sorted(int(x) for x in a.events.split(","))
NU = {12, 14, 16}

events, cur, vtx, n_ev, pscale, lscale = [], None, {}, -1, 1.0, 1.0
def flush():
    if cur is None: return
    parts = []
    for pdg, px, py, pz, vid in cur["p"]:
        if abs(pdg) in NU: continue
        x, y, z = vtx.get(vid, (0.0, 0.0, 0.0))
        parts.append([pdg, round(px * pscale, 6), round(py * pscale, 6), round(pz * pscale, 6),
                      round(x * lscale, 3), round(y * lscale, 3), round(z * lscale, 3)])
        if len(parts) >= a.maxp: break
    events.append({"id": f"{os.path.basename(a.hepmc)}_ev{cur['n']}",
                   "name": (a.name or os.path.basename(a.hepmc)) + f" (event {cur['n']})",
                   "kind": a.kind, "desc": f"{len(parts)} final-state particles from {os.path.basename(a.hepmc)}, event {cur['n']}",
                   "particles": parts})

with open(a.hepmc) as f:
    for line in f:
        t = line.split()
        if not t: continue
        if t[0] == "U":
            pscale = 1e-3 if t[1].upper() == "MEV" else 1.0
            lscale = 10.0 if t[2].upper() == "CM" else 1.0
        elif t[0] == "E":
            flush(); n_ev += 1
            if n_ev > want[-1]: cur = None; break
            cur = {"n": n_ev, "p": []} if n_ev in want else None; vtx = {}
        elif cur is not None and t[0] == "V":
            # V <id> <status> [children] @ x y z t
            if "@" in t:
                i = t.index("@"); vtx[int(t[1])] = (float(t[i + 1]), float(t[i + 2]), float(t[i + 3]))
        elif cur is not None and t[0] == "P":
            # P <id> <mother vertex> <pdg> <px> <py> <pz> <E> <m> <status>
            if int(t[9]) == 1:
                cur["p"].append((int(t[3]), float(t[4]), float(t[5]), float(t[6]), int(t[2])))
    flush()
json.dump({"events": events}, open(a.out, "w"), separators=(",", ":"))
print(f"wrote {a.out}: {[ (e['id'], len(e['particles'])) for e in events ]}")
