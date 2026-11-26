"""
WarpX-driven MDI background runner.

Runs WarpX once per (detector, beam_energy, scenario) combination, caches the
HepMC3 output keyed by a hash of the relevant config, and hands the path back
to the outer loop. Caching matters: a single run of the bilevel pipeline may
trigger hundreds of ddsim jobs, but they all need the same background overlay,
so we don't want to re-run a 30-minute WarpX simulation for each one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import warpx_input
from . import to_hepmc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MDIConfig:
    """Subset of the user config relevant to MDI generation.

    Mirrors the `mdi:` block in config.yaml. Kept as a dataclass so the rest
    of the pipeline doesn't have to dig through nested dicts.
    """
    enabled: bool = False
    warpx_executable: str = "warpx.2d"          # or warpx.3d for full geometry
    n_macroparticles: int = 100_000
    beam_energy_GeV: float = 45.6                # FCC-ee Z-pole default
    bunch_charge_nC: float = 24.0                # FCC-ee Z bunch charge
    crossing_angle_mrad: float = 30.0            # IDEA / FCC-ee MDI default
    sigma_x_um: float = 6.4                      # horizontal beam size at IP
    sigma_y_um: float = 0.028                    # vertical
    sigma_z_mm: float = 12.1                     # bunch length
    process: str = "beamstrahlung_pairs"         # | "sync_radiation" | "both"
    n_bx_overlay: int = 1                        # bunch crossings to overlay per signal evt
    cache_dir: str = "mdi_cache"                 # under runtime.runs_dir
    extra_warpx_args: list[str] = field(default_factory=list)
    # --- v2 deck fields (warpx_input v2) ---
    fidelity: str = "smoke"                      # "smoke" | "formenti"
    enable_ipc: bool = False                     # LL/BH/BW pair channels
    energy_spread: float = 0.0                   # fractional; formenti dflt 1.34e-3
    solenoid_T: float = 0.0                      # formenti default -2.0
    max_step: int = 0                            # 0 = preset default
    chi_min: float = 0.0                         # 0 = preset default
    launcher: list[str] = field(default_factory=list)  # e.g. ["srun","-n","4","-G","4"]

    @classmethod
    def from_dict(cls, d: dict | None) -> "MDIConfig":
        if not d:
            return cls(enabled=False)
        # Filter to known fields so unknown keys don't blow up
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)

    def cache_key(self, detector_label: str) -> str:
        """Stable hash of the inputs that affect WarpX output.

        Excludes things like cache_dir and warpx_executable path, which don't
        change the physics. Includes detector_label so different detector
        geometries (with different IP optics assumptions) don't collide.
        """
        payload = {
            "detector": detector_label,
            "n_macroparticles": self.n_macroparticles,
            "beam_energy_GeV": self.beam_energy_GeV,
            "bunch_charge_nC": self.bunch_charge_nC,
            "crossing_angle_mrad": self.crossing_angle_mrad,
            "sigma_x_um": self.sigma_x_um,
            "sigma_y_um": self.sigma_y_um,
            "sigma_z_mm": self.sigma_z_mm,
            "process": self.process,
            "extra_warpx_args": sorted(self.extra_warpx_args),
            "fidelity": self.fidelity,
            "enable_ipc": self.enable_ipc,
            "energy_spread": self.energy_spread,
            "solenoid_T": self.solenoid_T,
            "max_step": self.max_step,
            "chi_min": self.chi_min,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Artifact handle returned to the outer loop
# ---------------------------------------------------------------------------

@dataclass
class MDIArtifacts:
    """What the outer loop needs from MDI generation."""
    hepmc_path: Path | None       # None if MDI disabled
    n_bx_overlay: int = 1
    metadata: dict = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.hepmc_path is not None


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def build_mdi_backgrounds(
    cfg: dict,
    run_cfg: dict,
    *,
    runs_dir: Path,
    force: bool = False,
) -> MDIArtifacts:
    """Generate (or fetch from cache) MDI background HepMC for one named run.

    Single-scenario entry point: uses the top-level `mdi` config block
    directly, ignoring any `mdi.scenarios` sweep. For sweeps, call
    `build_mdi_scenarios` instead.

    Returns MDIArtifacts with hepmc_path=None if MDI is disabled in config.
    """
    mdi_cfg = MDIConfig.from_dict(cfg.get("mdi"))
    if not mdi_cfg.enabled:
        log.info("MDI disabled in config; outer loop will run signal-only.")
        return MDIArtifacts(hepmc_path=None)

    detector_label = run_cfg["outer_loop"]["detector_label"]
    return _build_for_cfg(
        mdi_cfg=mdi_cfg,
        detector_label=detector_label,
        runs_dir=runs_dir,
        force=force,
    )


def build_mdi_scenarios(
    cfg: dict,
    run_cfg: dict,
    *,
    runs_dir: Path,
    force: bool = False,
) -> list[tuple]:
    """Expand mdi.scenarios into N concrete WarpX runs (cached) and return them.

    Each entry is (MDIScenario, MDIArtifacts). The outer loop iterates over
    this list as an additional axis of its Cartesian product.

    If MDI is disabled, returns a single (no_mdi_scenario, disabled_artifacts)
    tuple so callers can iterate uniformly without conditionals.

    The expansion uses the same caching as the single-scenario path:
    MDIConfig.cache_key includes the swept fields, so each scenario gets its
    own cache directory automatically.
    """
    # Lazy import to avoid circular dependency at module load
    from .scenarios import expand_scenarios

    scenarios = expand_scenarios(cfg.get("mdi"))
    detector_label = run_cfg["outer_loop"]["detector_label"]

    log.info("Expanding %d MDI scenario(s) for run %r",
             len(scenarios), run_cfg.get("name"))

    out = []
    for sc in scenarios:
        if not sc.cfg.enabled:
            out.append((sc, MDIArtifacts(hepmc_path=None)))
            continue
        artifacts = _build_for_cfg(
            mdi_cfg=sc.cfg,
            detector_label=detector_label,
            runs_dir=runs_dir,
            force=force,
        )
        # Stash scenario provenance in the artifact metadata so it flows
        # all the way to manifest.json
        artifacts.metadata.setdefault("scenario", {})
        artifacts.metadata["scenario"]["label"] = sc.label
        artifacts.metadata["scenario"]["params"] = sc.params
        out.append((sc, artifacts))
    return out


# ---------------------------------------------------------------------------
# Shared per-config build (single point of WarpX invocation)
# ---------------------------------------------------------------------------

def _build_for_cfg(
    *,
    mdi_cfg: MDIConfig,
    detector_label: str,
    runs_dir: Path,
    force: bool,
) -> MDIArtifacts:
    """Generate (or cache-hit) one MDIArtifacts for a fully resolved MDIConfig."""
    cache_root = runs_dir / mdi_cfg.cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)

    key = mdi_cfg.cache_key(detector_label)
    cache_dir = cache_root / f"{detector_label}_{key}"
    hepmc_path = cache_dir / "mdi_background.hepmc"
    meta_path = cache_dir / "metadata.json"

    if hepmc_path.exists() and not force:
        log.info("MDI cache hit: %s", hepmc_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return MDIArtifacts(
            hepmc_path=hepmc_path,
            n_bx_overlay=mdi_cfg.n_bx_overlay,
            metadata=meta,
        )

    log.info("MDI cache miss; running WarpX in %s", cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate WarpX input deck from config
    deck_path = cache_dir / "warpx_inputs"
    warpx_input.write_deck(mdi_cfg, deck_path)

    # 2. Run WarpX
    raw_dump = _run_warpx(mdi_cfg, deck_path, cache_dir)

    # 3. Convert WarpX particle dump -> HepMC3
    to_hepmc.convert(
        warpx_dump=raw_dump,
        out_path=hepmc_path,
        beam_energy_GeV=mdi_cfg.beam_energy_GeV,
        process_label=mdi_cfg.process,
    )

    # 4. Persist metadata for cache validation + provenance
    meta = {
        "config": asdict(mdi_cfg),
        "detector": detector_label,
        "cache_key": key,
        "warpx_dump": str(raw_dump),
        "hepmc_path": str(hepmc_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return MDIArtifacts(
        hepmc_path=hepmc_path,
        n_bx_overlay=mdi_cfg.n_bx_overlay,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# WarpX invocation
# ---------------------------------------------------------------------------

def _run_warpx(mdi_cfg: MDIConfig, deck_path: Path, work_dir: Path) -> Path:
    """Run WarpX and return the path to the particle dump."""
    if not (shutil.which(mdi_cfg.warpx_executable)
            or Path(mdi_cfg.warpx_executable).is_file()):
        raise FileNotFoundError(
            f"WarpX executable {mdi_cfg.warpx_executable!r} not found on PATH. "
            "Either install WarpX (https://warpx.readthedocs.io/) or set "
            "mdi.warpx_executable in the config."
        )

    cmd = [
        *mdi_cfg.launcher,
        mdi_cfg.warpx_executable,
        str(deck_path),
        *mdi_cfg.extra_warpx_args,
    ]
    log.info("Running WarpX: %s", " ".join(cmd))
    log_path = work_dir / "warpx.log"
    with log_path.open("w") as fh:
        result = subprocess.run(cmd, cwd=work_dir, stdout=fh, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"WarpX exited with code {result.returncode}; see {log_path}"
        )

    # WarpX with diag1.format=openpmd writes diags/diag1/openpmd_<step>.bp5/.
    # Return the diag1/ directory itself; the converter passes the openpmd
    # series pattern to openpmd_api which walks all iterations from there.
    diag_dir = work_dir / "diags" / "diag1"
    if not diag_dir.exists():
        raise RuntimeError(
            f"WarpX completed but no diagnostic output found at {diag_dir}. "
            f"Check {log_path}."
        )
    has_openpmd = any(diag_dir.glob("openpmd_*.bp*")) or any(diag_dir.glob("openpmd_*.h5"))
    if not has_openpmd:
        # Fall back to AMReX plotfile layout
        plotfiles = sorted(p for p in diag_dir.iterdir() if p.is_dir())
        if not plotfiles:
            raise RuntimeError(f"No plotfiles produced in {diag_dir}")
        return plotfiles[-1]
    return diag_dir
