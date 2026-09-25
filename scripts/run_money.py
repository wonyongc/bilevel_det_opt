#!/usr/bin/env python3
"""Money-plot driver: seeded_radius_cog optimum vs MDI overlay intensity."""
import glob, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/bilevel_det_opt"))
import bilevel_opt.innerloop as il

RUN = "/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole"
OUT = os.path.join(RUN, "money" + os.environ.get("MONEY_TAG", "")); os.makedirs(OUT, exist_ok=True)
_BASE = os.environ.get("MONEY_SAMPLES_BASE", "")
if _BASE:   # e.g. MONEY_SAMPLES_BASE=e200MeV_fixed -> <base>_{0x,1x,3x}
    SAMPLES = {m: (RUN + "/sim_outputs/" + _BASE + "_" + m, x)
               for m, x in (("0x", 0.0), ("1x", 1.0), ("3x", 3.0))}
else:       # legacy 1 GeV layout
    SAMPLES = {
        "0x": (RUN + "/sim_outputs/e1GeV_fixed_0x", 0.0),
        "1x": (RUN + "/sim_outputs/e1GeV_fixed",    1.0),
        "3x": (RUN + "/sim_outputs/e1GeV_fixed_3x", 3.0),
    }
labels = sys.argv[1:] or ["1x"]
il._make_plots = lambda *a, **k: print("[money] built-in plots skipped")

entries = []
for lab in labels:
    d, x = SAMPLES[lab]
    merged = os.path.join(d, "merged.root")
    if not os.path.exists(merged):
        files = sorted(glob.glob(os.path.join(d, "events_*.root")))
        assert files, f"no events_*.root in {d}"
        print(f"[money] hadd {len(files)} files -> {merged}")
        subprocess.run(["hadd", "-f", merged] + files, check=True)
    entries.append({"run": lab, "geom_values": {"mdi_x": x}, "path": merged})

tag = "_".join(labels)
man = os.path.join(OUT, f"manifest_{tag}.json")
json.dump(entries, open(man, "w"), indent=1)

il_cfg = {
  "tree_name": "events",
  "branches": [{"name": "SCEPCal_MainEdep", "hit_type": "SimCalorimeterHit"}],
  "max_events": None,
  "geometry_labels": {"mdi_x": "MDI overlay intensity [x nominal]"},
  "heatmap": {"axes": ["mdi_x"]},
  "algorithm": {
    "name": "seeded_radius_cog", "params": {"branch": "SCEPCal_MainEdep"},
    "optimize_method": "differential_evolution",
    "optimize_params": [
      {"name": "R_cluster_mm",    "label": "Cluster radius R [mm]",
       "bounds": [0.0, 200.0], "plot_grid_points": 12},
      {"name": "E_threshold_GeV", "label": "Energy threshold [GeV]",
       "bounds": [0.0, 0.6], "plot_grid_points": 25},
    ],
  },
  "score": {"name": os.environ.get("MONEY_SCORE", "snr_energy"), "params": {"sigma0": 1.0}},
}

res = il.run_inner_loop(man, il_cfg, OUT)

def enc(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if hasattr(o, "__dict__"): return o.__dict__
    return str(o)

out = os.path.join(OUT, f"money_results_{tag}.json")
json.dump({"labels": labels, "il_cfg": il_cfg, "results": res},
          open(out, "w"), indent=1, default=enc)
print("[money] wrote", out)
