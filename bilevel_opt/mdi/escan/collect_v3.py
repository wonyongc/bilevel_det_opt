#!/usr/bin/env python3
"""Collect campaign v3 for one working point, with exactly the code of collect_v2.py.

    python collect_v3.py Z        -> $RUN/campaign_v3_Z_results.json

collect_v2.py is read and run with three substitutions, so the analysis is
identical to v2 and only the inputs change:
  background  sim_outputs/bkgonly_<m>      -> sim_outputs/escan_lumi2k_<pt>_<m>
  inner loops money_v2_<base>              -> money_v3_<pt>_<base>
  output      campaign_v2_results.json     -> campaign_v3_<pt>_results.json
Each substitution must occur at least once, otherwise it stops.
"""
import os, sys

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
if len(sys.argv) != 2 or sys.argv[1] not in ("Z", "WW", "ZH", "tt"):
    sys.exit("usage: python collect_v3.py Z|WW|ZH|tt")
PT = sys.argv[1]
src = open(os.path.join(RUN, "collect_v2.py")).read()
for a, b in (("bkgonly_", f"escan_lumi2k_{PT}_"), ("money_v2_", f"money_v3_{PT}_"),
             ("campaign_v2_results.json", f"campaign_v3_{PT}_results.json")):
    n = src.count(a)
    print(f"[collect_v3] {a!r} -> {b!r}: {n} occurrence(s)")
    if n == 0:
        sys.exit(f"collect_v2.py has no {a!r}; send me the lines that set the paths")
    src = src.replace(a, b)
sys.argv = [os.path.join(RUN, "collect_v2.py")]
exec(compile(src, f"collect_v2.py (v3, {PT})", "exec"), {"__name__": "__main__", "__file__": sys.argv[0]})
