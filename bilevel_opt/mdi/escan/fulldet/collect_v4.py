#!/usr/bin/env python3
"""Collect campaign v4 (full detector) for one working point, with the code of collect_v2.py.

    DET=CLD python fulldet/collect_v4.py Z   -> $RUN/campaign_v4_cld_Z_results.json

collect_v2.py is read and run with substitutions, as collect_v3.py does:
  background   sim_outputs/bkgonly_<m>        -> sim_outputs/escan_<det>_<pt>_<m>
  signal       sim_outputs/<base>_0x          -> sim_outputs/<det>_<base>_0x      (every base of the 14 samples)
  inner loops  money_v2_<base>                -> money_v4_<det>_<pt>_<base>
  hit branch   SCEPCal_MainEdep               -> the ECAL barrel collection of the detector
  output       campaign_v2_results.json       -> campaign_v4_<det>_<pt>_results.json
The forward samples are clustered on the ECAL endcap collection in the inner loop; if collect_v2.py
reads hits itself for the raw sums, those samples need the endcap branch there too — the script
stops and says so if it finds more than one place where the branch name is used, so that the
substitution can be made per sample instead of globally.
"""
import os, re, sys

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
DET = os.environ.get("DET", "")
BR = {"CLD": ("ECalBarrelCollection", "ECalEndcapCollection"),
      "ALLEGRO": ("ECalBarrelModuleThetaMerged", "ECalEndcapTurbine"),
      "IDEA": ("SCEPCal_MainEdep", "SCEPCal_MainEdep")}
if len(sys.argv) != 2 or sys.argv[1] not in ("Z", "WW", "ZH", "tt") or DET not in BR:
    sys.exit("usage: DET=CLD|ALLEGRO|IDEA python fulldet/collect_v4.py Z|WW|ZH|tt")
PT = sys.argv[1]; DTAG = DET.lower()
BASES = ["g100MeV_fixed", "g200MeV_fixed", "g500MeV_fixed", "g1000MeV_fixed", "g5000MeV_fixed", "e1GeV_fixed",
         "g200MeV_th140-160", "g200MeV_th160-200", "g200MeV_th200-300", "g200MeV_th300-500",
         "g1000MeV_th140-160", "g1000MeV_th160-200", "g1000MeV_th200-300", "g1000MeV_th300-500"]

src = open(os.path.join(RUN, "collect_v2.py")).read()
subs = [("bkgonly_", f"escan_{DTAG}_{PT}_"), ("money_v2_", f"money_v4_{DTAG}_{PT}_"),
        ("campaign_v2_results.json", f"campaign_v4_{DTAG}_{PT}_results.json")]
for a, b in subs:
    n = src.count(a)
    print(f"[collect_v4] {a!r} -> {b!r}: {n} occurrence(s)")
    if n == 0:
        sys.exit(f"collect_v2.py has no {a!r}; send me the lines that set the paths")
    src = src.replace(a, b)
# signal sample directories: "<base>_0x" -> "<det>_<base>_0x", only where not already prefixed
n_sig = 0
for base in BASES:
    pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(base) + r'_0x')
    src, k = pat.subn(f"{DTAG}_{base}_0x", src); n_sig += k
print(f"[collect_v4] signal directories prefixed with {DTAG!r}: {n_sig} occurrence(s)")
nb = src.count("SCEPCal_MainEdep")
print(f"[collect_v4] 'SCEPCal_MainEdep': {nb} occurrence(s)")
if DET != "IDEA" and nb > 1:
    sys.exit("collect_v2.py uses the hit branch in more than one place: the forward samples need the "
             "endcap collection there. Send me collect_v2.py and I will make the substitution per sample.")
src = src.replace("SCEPCal_MainEdep", BR[DET][0])
sys.argv = [os.path.join(RUN, "collect_v2.py")]
exec(compile(src, f"collect_v2.py (v4, {DET} {PT})", "exec"), {"__name__": "__main__", "__file__": sys.argv[0]})
