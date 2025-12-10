# bilevel_det_opt

Scripts to run outer loop geometry scans with key4hep or a minimal spack environment. Detector repositories must be specified.

## Contents
- `scripts/setup_spack.sh`: bootstrap or validate `key4hep-spack` and `spack` at
  pinned commits (key4hep-spack `57e59a3`, spack `6cb16c3`), applying
  `key4hep-spack/.cherry-pick` when present. Creates in-repo submodules or uses
  user-specified external paths.
- `spack_envs/minimal_dd4hep`: minimal spack environment definition for dd4hep if not using a centralized key4hep installation.
- `outerloop.py`: main driver that reads a YAML config, ensures the key4hep/spack
  environment is installed, builds the detector, generates steering/compact
  files for geometry sweeps, writes slurm scripts, and submits them.
- `config.example.yaml`: starting point for paths and run parameters with
  `environment.mode=key4hep` by default; spack settings live under `spack.*`
  with `spack_env_dir` pointing to the bundled env.

## Quickstart
1. Copy the example config and adjust paths to your detector clone and scratch
   locations:
   ```bash
   cp config.example.yaml myconfig.yaml
   $EDITOR myconfig.yaml
   ```
2. Choose environment mode in your config (`environment.mode=key4hep` by
   default). For spack mode, set `spack.spack_env_dir` to the target env path
   (default points to the bundled env), then make sure `scripts/setup_spack.sh`
   is executable and run it (clones spack and key4hep-spack if missing unless `use_external: true`):
   ```bash
   ./scripts/setup_spack.sh  # optional overrides via --spack-dir/--key4hep-dir
   ```
3. Run the workflow (use `--dry-run` to inspect commands first):
   ```bash
   python3 outerloop.py --config myconfig.yaml --dry-run
   python3 outerloop.py --config myconfig.yaml --make-threads 8          # executes
   python3 outerloop.py --config myconfig.yaml --submit --make-threads 8 # also sbatch jobs
   ```

## Workflow overview
- Environment modes:
  - `key4hep` (default): spack steps are skipped; runtime/build environments are
    set up by sourcing `key4hep_setup` from the config.
  - `spack`: `setup_spack.sh` enforces pinned commits and applies patches, then
    `outerloop.py` renders the bundled `spack_envs/minimal_dd4hep` templates into
    your target env dir (path from `spack.spack_env_dir`), runs
    `spack concretize/install`, and refreshes modulefiles. Modulefiles are
    stored under the env’s `modules/`. A lock file in `.state/` avoids
    re-running unless `--force-install` is passed.
- Detector build: CMake Release build/install with the chosen environment
  (lock `.state/detector_built.lock`, override via `--force-build`).
- Geometry sweeps: For each geometry value and seed/particle, the script:
  - Copies the compact XML and updates the requested `<constant>` value.
  - Fills the steering template placeholders (`seedplaceholder`, etc.).
  - Writes a slurm script that loads the configured modules/spack env and runs
    `ddsim --steeringFile <generated.py>`.
  - Submits via `sbatch` when `--submit` (or `submit: true` in YAML) is used.

## YAML highlights
- `environment.mode`: choose `key4hep` (default) or `spack`.
- `key4hep.*`: `setup_script` to source when in key4hep mode.
- `spack.*`: repo roots, env dir, and options for spack/key4hep-spack (env dir
  defaults to `spack_envs/minimal_dd4hep/env`).
- `detectors.*`: per-detector config (repo, source script, steering template,
  compact XML). Build/install paths default to `<repo>/build` and
  `<repo>/install`, with include/lib inferred accordingly.
- `runtime.*`: where to emit run artifacts (`runs_dir`), `ddsim` executable, and
  `scheduler` options (e.g. `scheduler.slurm`). Optional `env_commands` lets you
  provide a manual environment block.
- `batch.*`: simulation inputs and `geometry.parameter` sweep definition (list
  of values or start/stop/step with `unit`). `detector_label` is used for
  naming and default memory selection (`scepcal` -> 12G, otherwise 32G).

## Notes
- The scripts never modify the detector sources; compact XML edits are written
  to the run directory.
- Modulefile copy and spack installs are optional if locks already exist; see
  `.state/` for markers.
- Adjust paths in `config.example.yaml` to other clusters/detectors as needed;
  the workflow keeps detector-specific inputs isolated in the config.
