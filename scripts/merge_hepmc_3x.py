#!/usr/bin/env python3
"""Merge each 3 consecutive events of a HepMC3 ASCII overlay into one event
(3x intensity). Assumes the flat structure of mdi_background_10k.hepmc:
E n 1 <np> / U / W / P <id> 0 <pdg> ... with no V records."""
import sys
src, dst = sys.argv[1], sys.argv[2]
header, events, cur = [], [], None
for line in open(src):
    if line.startswith("E "):
        if cur is not None: events.append(cur)
        cur = [line]
    elif cur is None:
        header.append(line)
    else:
        cur.append(line)
if cur: events.append(cur)
n_out = len(events) // 3
print(f"source: {len(events)} events -> {n_out} merged 3x events")
with open(dst, "w") as out:
    out.writelines(header)
    for i in range(n_out):
        blocks = events[3*i:3*i+3]
        plines = [ln for b in blocks for ln in b[1:] if ln.startswith("P ")]
        out.write(f"E {i} 1 {len(plines)}\n")
        for ln in blocks[0][1:]:
            if not ln.startswith("P "):
                out.write(ln)
        for newid, ln in enumerate(plines, start=1):
            f = ln.split()
            f[1] = str(newid)
            out.write(" ".join(f) + "\n")
print(f"wrote {dst}")
