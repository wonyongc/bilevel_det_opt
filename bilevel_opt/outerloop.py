"""Step 3: geometry sweep — generate compact XML and SLURM files.

With MDI overlay (optional): if config["_mdi_scenarios"] is set by the
pipeline, generate_runs iterates over MDI scenarios as an additional axis.
Each (geometry, scenario) cell gets:
  - swept MDI parameters merged into geom_values (so the inner loop and
    its heatmaps can plot them as axes with zero code changes)
  - --inputFiles <hepmc> appended to ddsim (DD4hep's native overlay path)
  - scenario label folded into the run tag and slug
"""

from __future__ import annotations

import itertools
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bilevel_opt.util import detector_cfg, env_setup_cmd, log, manifest_path, output_dir, runs_dir
from bilevel_opt.geometry.sibling_resolver import link_siblings, summarize


# ---------------------------------------------------------------------------
# Seed primes — seed index maps to a large prime for ddsim --random.seed
# ---------------------------------------------------------------------------
SEED_PRIMES = [
    2000038169, 2000038193, 2000038277, 2000038289, 2000038297,
    2000038303, 2000038351, 2000038357, 2000038399, 2000038429,
    2000038433, 2000038459, 2000038471, 2000038487, 2000038519,
    2000038541, 2000038561, 2000038567, 2000038571, 2000038583,
    2000038619, 2000038631, 2000038657, 2000038673, 2000038693,
    2000038709, 2000038717, 2000038723, 2000038739, 2000038741,
    2000038769, 2000038787, 2000038801, 2000038841, 2000038847,
    2000038861, 2000038877, 2000038879, 2000038913, 2000038949,
    2000038979, 2000038981, 2000039003, 2000039011, 2000039017,
    2000039029, 2000039051, 2000039057, 2000039077, 2000039099,
    2000039141, 2000039159, 2000039179, 2000039189, 2000039213,
    2000039231, 2000039233, 2000039291, 2000039339, 2000039347,
    2000039387, 2000039413, 2000039423, 2000039449, 2000039453,
    2000039467, 2000039527, 2000039539, 2000039563, 2000039593,
    2000039623, 2000039627, 2000039651, 2000039663, 2000039669,
    2000039689, 2000039711, 2000039729, 2000039753, 2000039759,
    2000039791, 2000039831, 2000039843, 2000039863, 2000039869,
    2000039887, 2000039903, 2000039917, 2000039969, 2000040043,
    2000040059, 2000040061, 2000040101, 2000040103, 2000040131,
    2000040139, 2000040167, 2000040181, 2000040193, 2000040197,
]


# ---------------------------------------------------------------------------
# Geometry config
# ---------------------------------------------------------------------------
@dataclass
class Seeds:
    start: int
    end: int

    def values(self) -> List[int]:
        return list(range(self.start, self.end + 1))


@dataclass
class GeometrySweep:
    parameter: str
    unit: Optional[str]
    values: List[Decimal]

    def formatted_values(self) -> List[str]:
        return [
            f"{val.normalize()}*{self.unit}" if self.unit else f"{val.normalize()}"
            for val in self.values
        ]

    def slug(self, formatted_value: str) -> str:
        return formatted_value.replace("*", "").replace(".", "p").replace("/", "-").replace(" ", "")


def parse_seeds(config: dict) -> Seeds:
    seeds_cfg = config["outer_loop"]["seeds"]
    return Seeds(int(seeds_cfg["start"]), int(seeds_cfg["end"]))


def _parse_one_geometry(geom_cfg: dict) -> GeometrySweep:
    unit = geom_cfg.get("unit")
    values_cfg = geom_cfg.get("values")
    if values_cfg is not None:
        values = [Decimal(str(v)) for v in values_cfg]
    else:
        start = Decimal(str(geom_cfg["start"]))
        stop = Decimal(str(geom_cfg["stop"]))
        step = Decimal(str(geom_cfg["step"]))
        if step == 0:
            raise ValueError("geometry step cannot be zero")
        values, current = [], start
        while current <= stop + Decimal("1e-12"):
            values.append(current)
            current += step
    return GeometrySweep(parameter=geom_cfg["parameter"], unit=unit, values=values)


def parse_geometries(config: dict) -> List[GeometrySweep]:
    geom_cfg = config["outer_loop"]["geometry"]
    if isinstance(geom_cfg, dict):
        return [_parse_one_geometry(geom_cfg)]
    return [_parse_one_geometry(g) for g in geom_cfg]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _env_block(config: dict) -> str:
    """Shell environment lines embedded in each generated SLURM script."""
    env_cmds = config["runtime"].get("env_commands")
    if env_cmds:
        return "\n".join(env_cmds)

    det = detector_cfg(config)
    det_repo = Path(det["repo_root"]).expanduser()
    det_install = Path(det.get("install_dir") or (det_repo / "install"))

    lines = [env_setup_cmd(config)]
    if det.get("source_script"):
        lines.append(f"source {det['source_script']}")
    ld = det.get("ld_library_path") or f"{det_install / 'lib'}"
    inc = det.get("root_include_path") or f"{det_install / 'include'}"
    lines.append(f"export LD_LIBRARY_PATH={ld}:$LD_LIBRARY_PATH")
    lines.append(f"export ROOT_INCLUDE_PATH={inc}:$ROOT_INCLUDE_PATH")
    return "\n".join(lines)


def _update_dimensions(
    base_xml: Path,
    target_xml: Path,
    updates: List[Tuple[GeometrySweep, str]],
) -> None:
    """Copy the dimensions XML and set geometry constants."""
    tree = ET.parse(base_xml)
    root = tree.getroot()
    for geom, value in updates:
        found = False
        for const in root.findall(".//constant"):
            if const.attrib.get("name") == geom.parameter:
                const.set("value", value)
                found = True
        if not found:
            raise ValueError(f"Parameter {geom.parameter} not found in {base_xml}")
    target_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target_xml)
    desc = ", ".join(f"{g.parameter}={v}" for g, v in updates)
    log(f"Created dimensions XML {target_xml} with {desc}")


def _update_compact_include(
    base_xml: Path,
    target_xml: Path,
    old_dimensions: Path,
    new_dimensions: Path,
) -> None:
    """Copy the top-level compact XML and rewrite the <include> to point to the
    new dimensions file. Then symlink all other referenced siblings into the
    run dir so ddsim can resolve them.
    """
    tree = ET.parse(base_xml)
    root = tree.getroot()
    old_ref = str(old_dimensions)
    old_name = old_dimensions.name
    found = False
    for inc in root.findall(".//include"):
        ref = inc.attrib.get("ref", "")
        if ref == old_ref or ref.endswith(old_name):
            inc.set("ref", str(new_dimensions))
            found = True
    if not found:
        raise ValueError(f"No <include> referencing {old_name} found in {base_xml}")
    target_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target_xml)
    log(f"Created compact XML {target_xml} with include -> {new_dimensions}")
 
    # NEW: resolve and symlink all remaining sibling refs into the run dir,
    # so ddsim can find them when loaded from target_xml.parent.
    manifest = link_siblings(
        generated_compact=target_xml,
        source_compact=base_xml,
        run_dir=target_xml.parent,
    )
    log(summarize(manifest))

def _ddsim_cmd(
    config: dict,
    steering_file: Path,
    compact_xml: Path,
    output_file: Path,
    particle: str,
    momentum_gev: float,
    plusminus_frac: float,
    n_events: int,
    theta_min: float,
    theta_max: float,
    seed: int,
    mdi_hepmc: Optional[Path] = None,
) -> str:
    """Build the ddsim command line using native dd4hep options.

    If mdi_hepmc is provided, the HepMC overlay is appended via --inputFiles.
    dd4hep merges the gun-generated primaries with the HepMC primaries
    event-by-event, producing signal + MDI-background mixed events.
    """
    ddsim = config["runtime"]["ddsim_executable"]
    parts = [
        ddsim,
        f"--steeringFile {steering_file}",
        f"--compactFile {compact_xml}",
        f"--outputFile {output_file}",
        f"-N {n_events}",
        f"--gun.particle {particle}",
        f"--random.seed {seed}",
        f"--gun.thetaMin {math.radians(theta_min)}*rad",
        f"--gun.thetaMax {math.radians(theta_max)}*rad",
    ]
    mom_min = momentum_gev * (1 - plusminus_frac)
    mom_max = momentum_gev * (1 + plusminus_frac)
    parts.append(f"--gun.momentumMin {mom_min}*GeV")
    parts.append(f"--gun.momentumMax {mom_max}*GeV")
    if plusminus_frac > 0:
        parts.append("--gun.distribution uniform")
    if mdi_hepmc is not None:
        parts.append(f"--inputFiles {mdi_hepmc}")
    return " \\\n    ".join(parts)


def _slurm_text(config: dict, job_name: str, ddsim_cmd: str, slurm_dir: Path) -> str:
    slurm_cfg = config["runtime"]["scheduler"]["slurm"]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={slurm_dir}/{job_name}.out",
        f"#SBATCH --error={slurm_dir}/{job_name}.err",
        f"#SBATCH --cpus-per-task={slurm_cfg['cpus_per_task']}",
        f"#SBATCH --mem={slurm_cfg['mem']}",
        f"#SBATCH --time={slurm_cfg['time']}",
    ]
    for key, flag in [
        ("nodes", "nodes"), ("ntasks", "ntasks"),
        ("partition", "partition"), ("constraint", "constraint"),
        ("mail_type", "mail-type"), ("mail_user", "mail-user"),
        ("account", "account"), ("qos", "qos"),
    ]:
        if key in slurm_cfg:
            lines.append(f"#SBATCH --{flag}={slurm_cfg[key]}")
    lines += ["", _env_block(config), "", ddsim_cmd, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MDI scenario helpers
# ---------------------------------------------------------------------------
def _resolve_mdi_axis(config: dict) -> list:
    """Return the list of (scenario, artifacts) tuples to iterate over.

    Falls back to a single no-op scenario if MDI wasn't set up, so the main
    Cartesian product is uniform regardless of MDI being enabled.
    """
    scenarios = config.get("_mdi_scenarios")
    if scenarios:
        return scenarios
    # No MDI: use a single placeholder. Lazy-import to avoid circular dep.
    from bilevel_opt.mdi import MDIScenario, MDIConfig, MDIArtifacts
    return [(
        MDIScenario(cfg=MDIConfig(enabled=False), label="no_mdi", params={}),
        MDIArtifacts(hepmc_path=None),
    )]


def _scenario_slug(label: str) -> str:
    """Filesystem-safe scenario label fragment for tags and dirs."""
    return label.replace(".", "p").replace("/", "-").replace(" ", "")


# ---------------------------------------------------------------------------
# Main outer-loop
# ---------------------------------------------------------------------------
def generate_runs(config: dict, run_mode: str) -> None:
    """Generate compact XML, manifest, and run scripts for every geometry combo × MDI scenario × seed × particle.

    run_mode controls which scripts are written:
        slurm    — one .slurm file per job
        local    — one run_all.sh with all ddsim commands
        generate — manifest only (no run scripts)

    If config["_mdi_scenarios"] is set, the outer product includes an MDI axis.
    Each scenario's swept parameters are merged into geom_values so the inner
    loop's existing heatmap/scatter machinery can plot them as axes.
    """
    ol = config["outer_loop"]
    sim = ol["sim"]
    det = detector_cfg(config)

    rd = runs_dir(config)
    od = output_dir(config)
    slurm_dir = rd / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    compact_template = Path(det["compact_xml"]).expanduser()
    dimensions_template = Path(det["dimensions_xml"]).expanduser()
    steering_file = Path(det["steering_file"]).expanduser()

    geometries = parse_geometries(config)
    seeds = parse_seeds(config).values()
    particles = sim["particles"]
    detector_label = ol["detector_label"]

    momentum = sim["momentum_GeV"]
    plusminus_percent = sim["plusminus_percent"]
    n_events = sim["events"]
    theta_min = sim["theta_min"]
    theta_max = sim["theta_max"]

    # Resolve MDI scenarios (empty list -> single no-op)
    mdi_scenarios = _resolve_mdi_axis(config)
    has_real_mdi = any(art.enabled for _, art in mdi_scenarios)

    # Build per-sweep value lists: [(Decimal, formatted_str), ...]
    sweep_points = [
        list(zip(g.values, g.formatted_values())) for g in geometries
    ]

    manifest_entries: list = []
    run_idx = 0
    for combo in itertools.product(*sweep_points):
        geom_slug = "_".join(
            f"{g.parameter}-{g.slug(fval)}" for g, (_, fval) in zip(geometries, combo)
        )
        base_geom_values: Dict[str, float] = {
            g.parameter: float(dec) for g, (dec, _) in zip(geometries, combo)
        }
        xml_updates: List[Tuple[GeometrySweep, str]] = [
            (g, fval) for g, (_, fval) in zip(geometries, combo)
        ]

        # Geometry XML files are shared across MDI scenarios; only generate once.
        dims_copy_geom = rd / f"geom_{geom_slug}_dimensions.xml"
        compact_copy_geom = rd / f"geom_{geom_slug}.xml"
        _update_dimensions(dimensions_template, dims_copy_geom, xml_updates)
        _update_compact_include(compact_template, compact_copy_geom,
                                dimensions_template, dims_copy_geom)

        for scenario, mdi_art in mdi_scenarios:
            # Merge MDI swept params into geom_values so the inner loop can
            # plot them on heatmap axes alongside geometry params. This is
            # the integration's single key trick: the inner loop is untouched.
            geom_values = {**base_geom_values, **scenario.params}
            scen_slug = _scenario_slug(scenario.label) if has_real_mdi else ""

            for seed in seeds:
                for particle in particles:
                    tag = f"run{run_idx:04d}"
                    desc_parts = [
                        detector_label, particle,
                        f"{momentum}x{plusminus_percent}GeV",
                        geom_slug,
                    ]
                    if scen_slug:
                        desc_parts.append(scen_slug)
                    desc_parts += [f"s{seed}", f"N{n_events}"]
                    description = "_".join(desc_parts)

                    output_file = od / f"{tag}.root"

                    if seed < 0 or seed >= len(SEED_PRIMES):
                        raise IndexError(
                            f"Seed index {seed} out of range (0–{len(SEED_PRIMES) - 1})"
                        )

                    ddsim_cmd = _ddsim_cmd(
                        config=config,
                        steering_file=steering_file,
                        compact_xml=compact_copy_geom,
                        output_file=output_file,
                        particle=particle,
                        momentum_gev=momentum,
                        plusminus_frac=float(plusminus_percent) / 100.0,
                        n_events=n_events,
                        theta_min=theta_min,
                        theta_max=theta_max,
                        seed=SEED_PRIMES[seed],
                        mdi_hepmc=mdi_art.hepmc_path if mdi_art.enabled else None,
                    )

                    entry = {
                        "run": tag,
                        "description": description,
                        "geom_values": geom_values,
                        "particle": particle,
                        "seed": seed,
                        "path": str(output_file),
                        "ddsim_cmd": ddsim_cmd,
                    }
                    if has_real_mdi:
                        entry["mdi_scenario"] = scenario.label
                        entry["mdi_params"] = scenario.params
                        entry["mdi_overlay"] = (
                            str(mdi_art.hepmc_path) if mdi_art.enabled else None
                        )
                    manifest_entries.append(entry)

                    if run_mode == "slurm":
                        slurm_path = slurm_dir / f"{tag}.slurm"
                        slurm_content = _slurm_text(config, ddsim_cmd=ddsim_cmd,
                                                    job_name=tag, slurm_dir=slurm_dir)
                        slurm_path.write_text(slurm_content)
                        log(f"Wrote slurm {slurm_path}")

                    run_idx += 1

    # Write manifest
    mp = manifest_path(config)
    mp.write_text(json.dumps(manifest_entries, indent=2))
    log(f"Wrote manifest {mp} ({len(manifest_entries)} entries)")

    # Write local run script
    if run_mode == "local":
        lines = ["#!/bin/bash", "set -e", ""]
        if config["key4hep"]["run_setup"]:
            lines += [_env_block(config), ""]
        for entry in manifest_entries:
            lines.append(entry["ddsim_cmd"])
            lines.append("")
        sh_path = rd / "run_all.sh"
        sh_path.write_text("\n".join(lines))
        log(f"Wrote local run script {sh_path}")
