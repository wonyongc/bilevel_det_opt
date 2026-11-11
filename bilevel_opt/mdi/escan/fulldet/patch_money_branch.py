#!/usr/bin/env python3
"""Let the inner loop read a hit collection other than SCEPCal_MainEdep, and bound the
per-hit threshold per detector.

    python patch_money_branch.py            # patches $RUN/run_money.py in place (idempotent)

After the patch, run_money.py reads
  MONEY_BRANCH   the hit-collection name (default SCEPCal_MainEdep) in the two places it is
                 used: the "branches" list of the inner-loop config and the clustering
                 parameter "branch";
  MONEY_SEEDWIN  seed window [mm] (default 0 = highest-energy hit) and
  MONEY_METRIC   "xy" (default) or "pointing" — see patch_seed_window.py;
  MONEY_ETHRMAX  the upper bound of the per-hit energy threshold in GeV (default 0.6, the
                 crystal value). Sampling calorimeters deposit only a few keV per cell in the
                 active layer (CLD silicon: 4 keV/hit at Z), so the crystal range 0-0.6 GeV
                 would leave the optimizer and the 25-point grid with a single useful point.
Every existing campaign is unchanged by the defaults. paired_mix.py needs no change: it takes
the branch name from that config. Copy the patched file into ~/bilevel_det_opt/scripts/ before
committing, as for the other patches.
"""
import os, sys

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
p = os.path.join(RUN, "run_money.py")
s = open(p).read()
PATCHES = [
    ("MONEY_BRANCH",
     '"branches": [{"name": "SCEPCal_MainEdep", "hit_type": "SimCalorimeterHit"}],',
     '"branches": [{"name": os.environ.get("MONEY_BRANCH", "SCEPCal_MainEdep"), "hit_type": "SimCalorimeterHit"}],'),
    ("MONEY_BRANCH",
     '"name": "seeded_radius_cog", "params": {"branch": "SCEPCal_MainEdep"},',
     '"name": "seeded_radius_cog", "params": {"branch": os.environ.get("MONEY_BRANCH", "SCEPCal_MainEdep")},'),
    ("MONEY_ETHRMAX",
     '"bounds": [0.0, 0.6], "plot_grid_points": 25},',
     '"bounds": [0.0, float(os.environ.get("MONEY_ETHRMAX", "0.6"))], "plot_grid_points": 25},'),
    ("MONEY_SEEDWIN",
     '"name": "seeded_radius_cog", "params": {"branch": os.environ.get("MONEY_BRANCH", "SCEPCal_MainEdep")},',
     '"name": "seeded_radius_cog", "params": {"branch": os.environ.get("MONEY_BRANCH", "SCEPCal_MainEdep"), '
     '"seed_window_mm": float(os.environ.get("MONEY_SEEDWIN", "0")), "metric": os.environ.get("MONEY_METRIC", "xy")},'),
]
changed = []
for key, old, new in PATCHES:
    if new in s:
        continue
    n = s.count(old)
    if n == 0 and key in s:        # already carried by a later form of the same line
        continue
    if n != 1:
        sys.exit(f"run_money.py: expected exactly one occurrence of {old!r}, found {n}; send me the il_cfg block")
    s = s.replace(old, new); changed.append(key)
if changed:
    open(p, "w").write(s)
    print("run_money.py: patched", ", ".join(sorted(set(changed))))
else:
    print("run_money.py: MONEY_BRANCH, MONEY_ETHRMAX, MONEY_SEEDWIN/MONEY_METRIC already patched")
for key in ("MONEY_BRANCH", "MONEY_ETHRMAX", "MONEY_SEEDWIN", "MONEY_METRIC"):
    print(f"  {key}: {s.count(key)} use(s)")
