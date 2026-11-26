"""
One-click bilevel detector optimization pipeline.

Usage:
    python pipeline.py --config config.yaml [options]

Stages:
    mdi    - Run WarpX to generate MDI background HepMC overlays (cached per scenario).
    outer  - Source key4hep, generate geometry-scan runs, submit SLURM jobs.
    inner  - Run inner-loop optimization on ROOT outputs from the geometry scan.
    all    - Run all stages sequentially (default).

The outer loop produces ROOT files. Set run_mode in config (or --run-mode on CLI):
    local     - run ddsim jobs sequentially in the current shell.
    slurm     - submit SLURM batch jobs.
    generate  - only write run scripts (default).

MDI overlay (optional): if cfg["mdi"]["enabled"] is true, the MDI stage runs WarpX
to generate beam-induced backgrounds at the IP and overlays them onto every ddsim
job via --inputFiles. Multiple scenarios under cfg["mdi"]["scenarios"] become an
additional axis of the outer-loop Cartesian product.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    sys.exit(f"PyYAML is required. Install with `pip install pyyaml` ({exc})")

from bilevel_opt.outerloop import generate_runs
from bilevel_opt.util import manifest_path, results_dir, run_local, runs_dir, setup_env, submit_jobs
from bilevel_opt import mdi as mdi_pkg


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bilevel detector optimization: one-click pipeline."
    )
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--run", default=None, help="Run only the named run (default: all runs).")
    p.add_argument(
        "--stage",
        choices=["mdi", "outer", "inner", "all"],
        default="all",
        help="Which stage(s) to run (default: all).",
    )

    mdi = p.add_argument_group("MDI overlay")
    mdi.add_argument("--no-mdi", action="store_true",
                     help="Force-disable MDI even if config enables it.")
    mdi.add_argument("--force-mdi", action="store_true",
                     help="Re-run WarpX even if a cached HepMC exists.")
    mdi.add_argument("--mdi-scenario", default=None,
                     help="Restrict to one MDI scenario label (e.g. 'E45.6').")

    outer = p.add_argument_group("outer loop")
    outer.add_argument("--run-mode", choices=["local", "slurm", "generate"], default=None,
                       help="Override config run_mode (local | slurm | generate).")

    inner = p.add_argument_group("inner loop")
    inner.add_argument("--manifest", default=None,
                       help="Override path to manifest.json (default: from outer loop).")
    inner.add_argument("--max-events", type=int, default=None,
                       help="Override max_events from config.")
    inner.add_argument("--outdir", default=None,
                       help="Override inner-loop output directory.")

    return p


def merge_run_config(global_config: dict, run_block: dict) -> dict:
    """Merge a run block's outer_loop/inner_loop into global config.

    Paths are scoped: runs/<name>/sim_outputs and runs/<name>/results.
    """
    name = run_block["name"]
    merged = {**global_config}
    merged["outer_loop"] = run_block["outer_loop"]
    merged["inner_loop"] = run_block["inner_loop"]
    merged["runtime"] = {
        **global_config["runtime"],
        "runs_dir": str(Path(global_config["runtime"]["runs_dir"]) / name),
    }
    return merged


def run_mdi_stage(config: dict, args: argparse.Namespace) -> list:
    """Expand MDI scenarios, run WarpX (or use cache), and return the list.

    Returns a list of (MDIScenario, MDIArtifacts) tuples. If MDI is disabled,
    returns a single no-op tuple so callers can iterate uniformly.
    """
    if args.no_mdi:
        config.setdefault("mdi", {})["enabled"] = False

    # MDI cache is keyed by detector + physics params; outerloop's runs_dir
    # is the per-run subdir, but cache should be shared across runs. Use the
    # top-level runs_dir for caching.
    top_runs_dir = Path(config["runtime"]["runs_dir"])

    # Synthesize a minimal "run_cfg" shim: build_mdi_scenarios needs
    # outer_loop.detector_label, which lives in the merged config.
    run_cfg_shim = {
        "outer_loop": config.get("outer_loop", {}),
        "name": config.get("_run_name", "default"),
    }

    scenarios = mdi_pkg.build_mdi_scenarios(
        config, run_cfg_shim,
        runs_dir=top_runs_dir, force=args.force_mdi,
    )

    if args.mdi_scenario:
        scenarios = [(sc, art) for sc, art in scenarios
                     if sc.label == args.mdi_scenario]
        if not scenarios:
            sys.exit(f"--mdi-scenario={args.mdi_scenario!r} matched no scenarios")

    labels = [sc.label for sc, _ in scenarios]
    print(f"[mdi] Prepared {len(scenarios)} scenario(s): {labels}")
    return scenarios


def run_outer(config: dict, args: argparse.Namespace,
              mdi_scenarios: list | None = None) -> str:
    """Run all outer-loop steps. Returns the manifest path for the inner loop.

    If mdi_scenarios is provided, the outer loop iterates over them as an
    additional axis (geometry × MDI × seeds × particles). The list is attached
    to the config so generate_runs can read it without a signature change.
    """
    mode = args.run_mode or config["run_mode"]
    if mdi_scenarios:
        config["_mdi_scenarios"] = mdi_scenarios
    generate_runs(config, run_mode=mode)
    if mode == "local":
        run_local(runs_dir(config) / "run_all.sh")
    elif mode == "slurm":
        submit_jobs(runs_dir(config))
    return str(manifest_path(config))


def run_inner(config: dict, args: argparse.Namespace, mf: str) -> dict:
    """Run the inner loop on ROOT files listed in the manifest."""
    il_cfg = dict(config["inner_loop"])
    if args.max_events is not None:
        il_cfg["max_events"] = args.max_events
    outdir = args.outdir or str(results_dir(config))
    return run_inner_loop(manifest_file=mf, il_cfg=il_cfg, outdir=outdir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = make_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    global_config = yaml.safe_load(cfg_path.read_text())

    setup_env(global_config)

    run_blocks = global_config["runs"]
    if args.run:
        run_blocks = [r for r in run_blocks if r["name"] == args.run]
        if not run_blocks:
            names = [r["name"] for r in global_config["runs"]]
            sys.exit(f"Run '{args.run}' not found. Available: {names}")

    for run_block in run_blocks:
        run_name = run_block["name"]
        config = merge_run_config(global_config, run_block)
        config["_run_name"] = run_name

        print()
        print("#" * 60)
        print(f"  RUN: {run_name}")
        print("#" * 60)

        mf = args.manifest
        mdi_scenarios = None

        if args.stage in ("mdi", "outer", "all"):
            print("=" * 60)
            print(f"  MDI STAGE: WarpX background generation [{run_name}]")
            print("=" * 60)
            mdi_scenarios = run_mdi_stage(config, args)
            if args.stage == "mdi":
                continue

        if args.stage in ("outer", "all"):
            print("=" * 60)
            print(f"  OUTER LOOP: geometry sweep generation [{run_name}]")
            print("=" * 60)
            mf_from_outer = run_outer(config, args, mdi_scenarios=mdi_scenarios)
            if mf is None:
                mf = mf_from_outer

        if args.stage in ("inner", "all"):
            from bilevel_opt.innerloop import run_inner_loop  # lazy: needs ROOT
            if mf is None:
                mf = str(manifest_path(config))
            print()
            print("=" * 60)
            print(f"  INNER LOOP: algorithm parameter optimization [{run_name}]")
            print("=" * 60)
            summary = run_inner(config, args, mf)

            best = summary["global_best"]
            gv = ", ".join(f"{k}={v}" for k, v in best["geom_values"].items())
            bp = ", ".join(f"{k}={v:.4g}" for k, v in best["best_params"].items())
            print()
            print("=" * 60)
            print(f"  RESULT [{run_name}]: {gv}, {bp}, "
                  f"score={best['best_score']:.6g}")
            print("=" * 60)


if __name__ == "__main__":
    main()
