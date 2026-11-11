# bilevel_det_opt

Bilevel optimization of detector geometry and reconstruction algorithm parameters. The **outer loop** sweeps detector geometry constants via dd4hep simulations (locally or on SLURM), and the **inner loop** optimizes reconstruction algorithm parameters per geometry point using scipy, producing a joint loss landscape.

## Prerequisites

- [key4hep](https://key4hep.github.io/) available via CVMFS or a local build
- ROOT with PyROOT (comes with key4hep)

## Installation

```bash
pip install -e .
```

This installs the `bilevel_opt` CLI command and Python package.

## File structure

```
bilevel_opt/                 — Python package
  pipeline.py                — entry point, chains outer + inner loops
  outerloop.py               — geometry sweep: compact XML, ddsim commands, manifests
  innerloop.py               — per-geometry algorithm optimization + landscape plots
  edm.py                     — EDM wrappers for edm4hep ROOT objects (hit types)
  util.py                    — shared utilities, config accessors, job execution
  algorithms/
    __init__.py              — re-exports ALGORITHMS and SCORES registries
    registry.py              — name -> class mappings
    seeded_radius_cog.py     — example algorithm + scorer
pyproject.toml               — package metadata + CLI entry point
config.example.yaml          — starting-point configuration
```

## Quickstart

1. Copy and edit the config:
   ```bash
   cp config.example.yaml myconfig.yaml
   $EDITOR myconfig.yaml
   ```

2. Generate run scripts (without executing):
   ```bash
   bilevel_opt --config myconfig.yaml --run-mode generate
   ```

3. Run the full pipeline:
   ```bash
   # Run ddsim locally, then optimize
   bilevel_opt --config myconfig.yaml

   # Or run stages independently
   bilevel_opt --config myconfig.yaml --stage outer
   bilevel_opt --config myconfig.yaml --stage inner
   ```

4. Run a specific named run:
   ```bash
   bilevel_opt --config myconfig.yaml --run test_2x2params
   ```

Also available as `python -m bilevel_opt --config myconfig.yaml`.


## MDI energy scan and full detectors

`bilevel_opt/mdi/escan/` holds the WarpX pair-background energy scan at the four FCC-ee working points
(decks, HepMC3 conversion, standalone crystal geometries, inner-loop campaign v3, figures), and
`bilevel_opt/mdi/escan/fulldet/` the same chain in the full CLD_o2_v08, ALLEGRO_o1_v03 and IDEA_o2_v01
detector models (campaign v4 and its bootstrap collection). See `README_escan.md` and
`fulldet/README_fulldet.md`.

## How it works

### Outer loop

1. **Environment** — optionally sources key4hep via `setup_script` (controlled by `run_setup`).
2. **Geometry sweep** (`outerloop.py`) — for the Cartesian product of all geometry parameter values x seeds x particles:
   - Copies the dimensions XML (`dimensions_xml`) with updated geometry constants
   - Copies the top-level compact XML (`compact_xml`) with the `<include>` rewritten to point to the new dimensions file
   - Builds the `ddsim` command using native dd4hep CLI options (`--compactFile`, `--gun.particle`, `--gun.momentumMin/Max`, `--random.seed`, etc.)
   - Writes a `manifest.json` mapping each ROOT output file to its geometry parameter values and ddsim command
   - Depending on `run_mode`: writes per-job `.slurm` files (`slurm`), a single `run_all.sh` (`local`), or just the manifest (`generate`)
3. **Execution** — submits SLURM jobs or runs the shell script locally.

Files are named `run0000.xml`, `run0000.root`, etc. The manifest maps each to its descriptive metadata.

Seeds are indices into a table of large primes used as ddsim random seeds for reproducibility.

### Inner loop

For each entry in the manifest:

1. **Load events** — reads ROOT files with configurable tree, branches, and hit types. Each event is a dict mapping branch name to its EDM hit collection, so algorithms can access multiple subdetectors.
2. **Optimize** — minimizes `-score(algo.reconstruct(events, params))` over the algorithm parameter space using the configured scipy method. All objective evaluations are recorded in an optimization history.
3. **Grid landscape** — for 1-2 algorithm parameters, evaluates the score on a grid for visualization (configurable via `force_grid_landscape` for 3+ params).
4. **Output** — saves `summary.json`, `summary.csv`, per-geometry NPZ files (including optimizer history), landscape tensor, and plots.

### Algorithm plugins

Algorithms and scores live in `bilevel_opt/algorithms/` and are registered by name in `registry.py`. The inner loop instantiates them from the config.

**Algorithm interface:**
```python
class MyAlgorithm:
    name = "my_algorithm"
    def __init__(self, branch: str, ...):  # constructor params from config
        ...
    def reconstruct(self, events: list[dict], params: dict) -> reco:
        # events[i] is a dict mapping branch name -> hit collection
        # params are the values being optimized
        ...
```

**Score interface:**
```python
class MyScore:
    name = "my_score"
    def __init__(self, sigma0: float, ...):  # constructor params from config
        ...
    def __call__(self, events, reco) -> float:
        ...
```

The included example (`seeded_radius_cog`) clusters calorimeter hits within a radius `R_cluster_mm` of the highest-energy seed hit, with an energy threshold `E_threshold_GeV` to suppress noise. The score (`snr_energy`) computes a signal-to-noise proxy: `mean(E_cluster / (sigma0 * sqrt(N_hits)))`.

## Outputs

Per named run, outputs go to `runs/<name>/`:

- `run*.xml`, `run*_dimensions.xml` — per-run compact XML copies
- `sim_outputs/` — ROOT files from ddsim + `manifest.json`
- `slurm/` — SLURM scripts and logs (when `run_mode: slurm`)
- `run_all.sh` — local run script (when `run_mode: local`)
- `results/` — inner loop outputs:
  - `summary.json` / `summary.csv` — per-geometry optimal parameters and global best
  - `landscape_*.npz` — per-geometry landscapes with optimizer history (`opt_history_x`, `opt_history_f`)
  - `loss_landscape_tensor.npz` — score grid over algorithm parameter space (1-2 algo params)
  - `landscape_heatmap.png` — 2D heatmap (1 geom param x 1 algo param) or best-score heatmap (2 geom params)
  - `landscape_3d.png` — 3D surface plot (2 geom params)
  - `best_score_vs_<geom_param>.png` — optimized score vs each geometry parameter
  - `best_<algo_param>_vs_<geom_param>.png` — optimal algorithm parameter(s) vs each geometry parameter

## Config reference

### Global

| Key | Description |
|-----|-------------|
| `run_mode` | `local` (run ddsim in shell), `slurm` (sbatch), or `generate` (scripts only) |
| `key4hep.run_setup` | Whether to run the setup script (`true`/`false`) |
| `key4hep.setup_script` | Shell command to set up the runtime environment |
| `detectors.<label>` | Per-detector: `repo_root`, `source_script`, `steering_file`, `compact_xml`, `dimensions_xml` |
| `runtime.runs_dir` | Base directory for all run outputs |
| `runtime.ddsim_executable` | ddsim binary name or path |
| `runtime.scheduler.slurm` | SLURM options: `cpus_per_task`, `mem`, `time`, `partition`, etc. |

### Runs

Each entry in the `runs` list has:

| Key | Description |
|-----|-------------|
| `name` | Run identifier (used for output directory naming) |
| `outer_loop.detector_label` | Which detector from `detectors` to use |
| `outer_loop.seeds` | `start`/`end` range of seed indices |
| `outer_loop.geometry` | List of geometry sweeps, each with `parameter`, `start`/`stop`/`step` (or `values`), `unit` |
| `outer_loop.sim` | ddsim options: `particles`, `events`, `momentum_GeV`, `plusminus_percent`, `theta_min`, `theta_max` |
| `inner_loop.tree_name` | ROOT tree name |
| `inner_loop.branches` | List of `{name, hit_type}` for ROOT branches to read |
| `inner_loop.geometry_labels` | Display labels for geometry params (dict) |
| `inner_loop.heatmap` | Plot config: `axes` (1-2 geom params to plot), `fixed` (pin others for 3+ params) |
| `inner_loop.algorithm.name` | Algorithm from registry |
| `inner_loop.algorithm.params` | Constructor kwargs for the algorithm (e.g. `branch`) |
| `inner_loop.algorithm.optimize_method` | scipy method: `differential_evolution`, `L-BFGS-B`, `dual_annealing`, `shgo`, `brute`, `bounded` |
| `inner_loop.algorithm.optimize_params` | List of params to optimize, each with `name`, `label`, `bounds`, `plot_grid_points` |
| `inner_loop.algorithm.force_grid_landscape` | Force full grid evaluation for 3+ algo params (default: false) |
| `inner_loop.algorithm.brute_grid_sizes` | Grid density per param (only for `brute` method) |
| `inner_loop.score.name` | Score from registry |
| `inner_loop.score.params` | Constructor kwargs for the score (e.g. `sigma0`) |
| `inner_loop.max_events` | Max events per ROOT file (`null` for all) |

## CLI options

```
bilevel_opt --config CONFIG [options]

  --run NAME          Run only the named run (default: all)
  --stage STAGE       outer | inner | all (default: all)
  --run-mode MODE     Override config run_mode
  --manifest PATH     Override manifest path for inner loop
  --max-events N      Override max_events from config
  --outdir DIR        Override inner-loop output directory
```
