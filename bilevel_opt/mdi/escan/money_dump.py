#!/usr/bin/env python3
"""Collect the inner-loop landscapes and optimizer histories of campaign v3 (and v2) into one JSON.

    python money_dump.py                  -> $RUN/money_dump.json

For every money_v3_<pt>_<base>/landscape_*.npz (barrel samples, all four working points) and the
matching money_v2_<base>/ directory, every array in the npz is stored as a list (arrays above
200k elements are skipped and reported). The first file's keys and shapes are printed, so the
plotting side can be written against the real layout.
"""
import glob, json, os
import numpy as np

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
BASES = ["g100MeV_fixed", "g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed", "g5000MeV_fixed", "e1GeV_fixed"]
DIRS = [f"money_v3_{p}_{b}" for p in ("Z", "WW", "ZH", "tt") for b in BASES] + [f"money_v2_{b}" for b in BASES]


def conv(v):
    a = np.asarray(v)
    if a.dtype.kind in "fiub":
        if a.size > 200_000:
            return None
        return np.round(a.astype(float), 7).tolist()
    if a.dtype.kind in "US":
        return a.tolist()
    try:
        return json.loads(json.dumps(a.tolist()))
    except Exception:
        return str(a)


out, shown = {}, False
for d in DIRS:
    files = sorted(glob.glob(os.path.join(RUN, d, "landscape_*.npz")))
    if not files:
        print(f"MISSING {d}"); continue
    out[d] = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        if not shown:
            print(f"layout of {d}/{os.path.basename(f)}:")
            for k in z.files:
                a = z[k]; print(f"   {k:28} {str(a.dtype):10} {a.shape}" + (f"  = {a.tolist()}" if a.size <= 4 else ""))
            shown = True
        rec = {}
        for k in z.files:
            c = conv(z[k])
            if c is None:
                print(f"   skipped {d}/{os.path.basename(f)}:{k} (too large)")
            else:
                rec[k] = c
        out[d][os.path.basename(f)] = rec
    print(f"{d}: {len(files)} files")
other = sorted({os.path.basename(p) for d in DIRS for p in glob.glob(os.path.join(RUN, d, "*"))} - {""})
print("files present in the money directories:", [o for o in other if not o.startswith("landscape_")][:20])
p = os.path.join(RUN, "money_dump.json")
json.dump(out, open(p, "w"))
print(f"wrote {p}  ({os.path.getsize(p)/1e6:.1f} MB)")
