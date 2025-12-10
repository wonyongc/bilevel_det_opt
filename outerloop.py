#!/usr/bin/env python3
"""
Prepares spack/key4hep and detector repos, submits
geometry-scan batch jobs on on slurm.
"""

from __future__ import annotations

import argparse
import json
import shlex
import stat
import subprocess
import sys
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import yaml
except ImportError as exc:
    sys.exit(
        "PyYAML is required. Install with `pip install pyyaml` "
        f"(import error: {exc})."
    )


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
        formatted = []
        for val in self.values:
            val_str = f"{val.normalize()}" if val == val.to_integral() else f"{val.normalize()}"
            formatted.append(f"{val_str}*{self.unit}" if self.unit else val_str)
        return formatted

    def slug(self, formatted_value: str) -> str:
        slug = formatted_value.replace("*", "").replace(".", "p").replace("/", "-")
        return slug.replace(" ", "")


class Workflow:
    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.root = Path(__file__).resolve().parent
        self.env_cfg = config.get("environment", {})
        self.env_mode = self.env_cfg.get("mode", "key4hep")
        self.spack_cfg = config.get("spack", {})
        self.key4hep_cfg = config.get("key4hep", {})
        self.detectors_cfg = config.get("detectors", {})
        self.spack_repo_dir = Path(
            self.spack_cfg.get("spack_repo_root", self.root / "spack")
        ).expanduser()
        self.state_dir = (self.root / config.get("state_dir", ".state")).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.spack_env_template = Path(
            self.spack_cfg.get(
                "spack_env_template", self.root / "spack_envs" / "minimal_dd4hep"
            )
        ).expanduser()
        self.dry_run = args.dry_run
        self.force_install = args.force_install
        self.force_build = args.force_build
        self.submit_enabled = args.submit or config.get("submit", False)
        self.make_threads = args.make_threads or os.cpu_count() or 8

    # ---------- basic utilities ------------------------------------------------
    def log(self, msg: str) -> None:
        print(f"[bilevel] {msg}")

    def run_bash(self, command: str, cwd: Optional[Path] = None) -> None:
        printable = command.replace("\n", " ").strip()
        self.log(f"run: {printable}")
        if self.dry_run:
            return
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd) if cwd else None,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Command failed ({result.returncode}): {printable}")

    def write_lock(self, name: str, payload: dict) -> None:
        lock_path = self.state_dir / name
        payload = {"timestamp": datetime.utcnow().isoformat(), **payload}
        lock_path.write_text(json.dumps(payload, indent=2))
        self.log(f"Wrote lock {lock_path}")

    def has_lock(self, name: str) -> bool:
        return (self.state_dir / name).exists()

    def detector_label(self) -> str:
        return self.config.get("batch", {}).get("detector_label", "scepcal")

    def detector_cfg(self) -> dict:
        label = self.detector_label()
        detectors = self.detectors_cfg
        if label not in detectors:
            raise KeyError(f"Detector config for '{label}' not found under detectors.")
        return detectors[label]

    # ---------- configuration helpers -----------------------------------------
    def seeds(self) -> Seeds:
        seeds_cfg = self.config.get("batch", {}).get("seeds", {})
        return Seeds(int(seeds_cfg.get("start", 1)), int(seeds_cfg.get("end", 1)))

    def geometry(self) -> GeometrySweep:
        geom_cfg = self.config.get("batch", {}).get("geometry", {})
        unit = geom_cfg.get("unit")
        values_cfg = geom_cfg.get("values")
        if values_cfg is not None:
            values = [Decimal(str(v)) for v in values_cfg]
        else:
            try:
                start = Decimal(str(geom_cfg["start"]))
                stop = Decimal(str(geom_cfg["stop"]))
                step = Decimal(str(geom_cfg["step"]))
            except (InvalidOperation, KeyError) as exc:
                raise ValueError("geometry sweep requires values list or start/stop/step") from exc
            if step == 0:
                raise ValueError("geometry step cannot be zero")
            values = []
            current = start
            # Allow small floating point slack
            while current <= stop + Decimal("1e-12"):
                values.append(current)
                current += step
        return GeometrySweep(parameter=geom_cfg["parameter"], unit=unit, values=values)

    # ---------- setup steps ----------------------------------------------------
    def ensure_spack_checkouts(self) -> None:
        # Clone/verify spack and key4hep-spack at pinned commits when in spack mode.
        if self.env_mode == "key4hep":
            self.log("Environment mode key4hep: skipping spack/key4hep-spack checkout.")
            return

        script_path = self.root / "scripts" / "setup_spack.sh"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing setup script at {script_path}")

        args: List[str] = []
        if self.spack_cfg.get("spack_repo_root"):
            args += ["--spack-dir", self.spack_cfg["spack_repo_root"]]
        if self.spack_cfg.get("key4hep_repo_root"):
            args += ["--key4hep-dir", self.spack_cfg["key4hep_repo_root"]]
        if self.spack_cfg.get("spack_url"):
            args += ["--spack-url", self.spack_cfg["spack_url"]]
        if self.spack_cfg.get("key4hep_url"):
            args += ["--key4hep-url", self.spack_cfg["key4hep_url"]]
        if self.spack_cfg.get("use_external"):
            args.append("--use-external")

        cmd = " ".join([shlex.quote(str(script_path))] + [shlex.quote(a) for a in args])
        self.run_bash(cmd)

    def install_spack_env(self) -> None:
        # Render env templates and run spack install/module refresh (spack mode only).
        if self.env_mode == "key4hep":
            self.log("Environment mode key4hep: skipping spack env install.")
            return

        lock_name = "spack_env_installed.lock"
        if self.has_lock(lock_name) and not self.force_install:
            self.log("Spack env lock present; skipping spack install.")
            return

        self.prepare_spack_env()

        gcc_module = self.spack_cfg.get("gcc_toolset_module")
        env_activate = [
            f"source {self.spack_cfg['spack_setup_script']}",
            f"spack env activate -d {self.spack_cfg['spack_env_dir']}",
        ]
        cmd_parts: List[str] = []
        if gcc_module:
            cmd_parts.append(f"module load {gcc_module}")
        cmd_parts.extend(env_activate)
        cmd_parts += [
            "spack concretize -f",
            "spack install",
            "spack module tcl refresh",
        ]
        self.run_bash(" && ".join(cmd_parts))

        env_dir = Path(self.spack_cfg["spack_env_dir"]).expanduser()
        module_src = self.spack_cfg.get("modulefile_source")
        module_dest = self.spack_cfg.get("modulefile_target")
        src_path = Path(module_src).expanduser()
        dest_path = Path(module_dest).expanduser()
        if src_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.dry_run:
                dest_path.write_text(src_path.read_text())
            self.log(f"Copied modulefile {src_path} -> {dest_path}")
        else:
            self.log(f"Modulefile source {src_path} not found; skipping copy.")

        self.write_lock(
            lock_name,
            {
                "spack_env": self.spack_cfg["spack_env_dir"],
                "spack_setup": self.spack_cfg["spack_setup_script"],
            },
        )

    def prepare_spack_env(self) -> None:
        # Render spack env templates with repo-relative defaults.
        env_dir = Path(
            self.spack_cfg.get(
                "spack_env_dir", self.root / "spack_envs" / "minimal_dd4hep" / "env"
            )
        ).expanduser()
        self.spack_cfg.setdefault("spack_env_dir", str(env_dir))
        env_dir.mkdir(parents=True, exist_ok=True)

        template_spack = self.spack_env_template / "spack.yaml"
        if not template_spack.exists():
            raise FileNotFoundError(f"Spack env template missing: {template_spack}")

        spack_opt_root = self.spack_cfg.get("spack_opt_root") or str(
            (env_dir / ".spack" / "opt").expanduser()
        )
        module_root = self.spack_cfg.get("module_root") or str(
            (env_dir / "modules").expanduser()
        )
        install_cfg = self.config.get("install", {})
        detector_repo = Path(install_cfg.get("detector_repo", "")).expanduser()
        detector_install = Path(install_cfg.get("install_dir", detector_repo / "install")).expanduser()
        module_use_paths = self.spack_cfg.get("module_use_paths") or [module_root]
        module_name = self.spack_cfg.get("module_name", "SCEPCal/1.0")
        spack_setup_script = self.spack_cfg.get("spack_setup_script") or str(
            self.spack_repo_dir / "share" / "spack" / "setup-env.sh"
        )
        modulefile_target = self.spack_cfg.get("modulefile_target") or str(
            Path(module_root) / "SCEPCal" / "1.0"
        )
        modulefile_source = self.spack_cfg.get("modulefile_source") or str(env_dir / "SCEPCal_modulefile")

        self.spack_cfg.setdefault("spack_opt_root", spack_opt_root)
        self.spack_cfg.setdefault("module_root", module_root)
        self.spack_cfg.setdefault("module_use_paths", module_use_paths)
        self.spack_cfg.setdefault("module_name", module_name)
        self.spack_cfg.setdefault("spack_setup_script", spack_setup_script)
        self.spack_cfg.setdefault("modulefile_target", modulefile_target)
        self.spack_cfg.setdefault("modulefile_source", modulefile_source)

        replacements = {
            "@SPACK_OPT_ROOT@": spack_opt_root,
            "@MODULE_ROOT@": module_root,
            "@MODULE_USE_BASE@": module_use_paths[0] if module_use_paths else "",
            "@MODULE_USE_PLATFORM@": module_use_paths[1] if len(module_use_paths) > 1 else "",
            "@DETECTOR_MODULE@": module_name,
            "@SPACK_SETUP@": spack_setup_script,
            "@SPACK_ENV_DIR@": str(env_dir),
            "@VIEW_ROOT@": str(env_dir / ".spack-env" / "view"),
            "@DETECTOR_INSTALL_LIB@": self.spack_cfg.get(
                "modulefile_ld_library_path", str(detector_install / "lib")
            ),
            "@DETECTOR_INSTALL_INCLUDE@": self.spack_cfg.get(
                "modulefile_root_include_path", str(detector_install / "include")
            ),
        }

        def render_template(src: Path, dest: Path) -> None:
            content = src.read_text()
            for key, value in replacements.items():
                content = content.replace(key, value)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not self.dry_run:
                dest.write_text(content)
                if dest.suffix == ".sh":
                    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            self.log(f"Rendered {dest} from template {src}")

        templates = [
            ("spack.yaml", True),
            ("packages.yaml", False),
            ("minimal_dd4hep.sh", True),
            ("SCEPCal_modulefile", True),
        ]
        for filename, force_render in templates:
            src = self.spack_env_template / filename
            if not src.exists():
                continue
            dest = env_dir / filename
            if dest.exists() and not (self.force_install or force_render):
                continue
            render_template(src, dest)

    def build_detector(self) -> None:
        # Configure, build, and install the detector once per lock unless forced.
        lock_name = "detector_built.lock"
        if self.has_lock(lock_name) and not self.force_build:
            self.log("Detector build lock present; skipping cmake.")
            return

        detector_cfg = self.detector_cfg()
        detector_repo = Path(detector_cfg.get("repo_root", "")).expanduser()
        build_dir = Path(detector_cfg.get("build_dir", detector_repo / "build")).expanduser()
        install_dir = Path(detector_cfg.get("install_dir", detector_repo / "install")).expanduser()
        cmake_type = "Release"
        make_threads = int(self.make_threads)

        if not detector_repo.exists():
            raise FileNotFoundError(f"Detector repo not found: {detector_repo}")
        build_dir.mkdir(parents=True, exist_ok=True)
        install_dir.mkdir(parents=True, exist_ok=True)

        cmd_parts: List[str] = []
        if self.env_mode == "spack":
            if not self.spack_cfg:
                raise KeyError("Spack config is required when environment.mode=spack")
            gcc_module = self.spack_cfg.get("gcc_toolset_module")
            if gcc_module:
                cmd_parts.append(f"module load {gcc_module}")
            cmd_parts.extend(
                [
                    f"source {self.spack_cfg['spack_setup_script']}",
                    f"spack env activate -d {self.spack_cfg['spack_env_dir']}",
                ]
            )
        else:
            key4hep_setup = self.key4hep_cfg.get("setup_script")
            if not key4hep_setup:
                raise KeyError("key4hep.setup_script is required when environment.mode=key4hep")
            cmd_parts.append(f"source {key4hep_setup}")

        cmake_cmd = (
            f"cd {shlex.quote(str(build_dir))} && "
            f"cmake -Wno-dev -DCMAKE_BUILD_TYPE={cmake_type} "
            f"-DCMAKE_INSTALL_PREFIX={shlex.quote(str(install_dir))} .. && "
            f"make install -j{make_threads}"
        )
        cmd_parts.append(cmake_cmd)
        self.run_bash(" && ".join(cmd_parts), cwd=detector_repo)

        self.write_lock(
            lock_name,
            {
                "detector_repo": str(detector_repo),
                "build_dir": str(build_dir),
                "install_dir": str(install_dir),
            },
        )

    # ---------- run generation -------------------------------------------------
    def env_block(self) -> str:
        # Construct environment setup for slurm scripts or use manual overrides.
        runtime_cfg = self.config.get("runtime", {})
        env_cmds: Optional[List[str]] = runtime_cfg.get("env_commands")
        if env_cmds:
            return "\n".join(env_cmds)

        if self.env_mode == "key4hep":
            key4hep_setup = self.key4hep_cfg.get("setup_script")
            if not key4hep_setup:
                raise KeyError("key4hep.setup_script is required when environment.mode=key4hep")
            lines = [f"source {key4hep_setup}"]
            detector_cfg = self.detector_cfg()
            if detector_cfg.get("source_script"):
                lines.append(f"source {detector_cfg['source_script']}")
            detector_repo = Path(detector_cfg.get("repo_root", "")).expanduser()
            detector_install = Path(detector_cfg.get("install_dir", detector_repo / "install"))
            ld_path = detector_cfg.get("ld_library_path", str(detector_install / "lib"))
            include_path = detector_cfg.get("root_include_path", str(detector_install / "include"))
            if ld_path:
                lines.append(f"export LD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH")
            if include_path:
                lines.append(f"export ROOT_INCLUDE_PATH={include_path}:$ROOT_INCLUDE_PATH")
            return "\n".join(lines)

        module_paths = self.spack_cfg.get("module_use_paths", [])
        module_name = self.spack_cfg.get("module_name", "SCEPCal/1.0")

        spack_env_dir = Path(self.spack_cfg.get("spack_env_dir", ""))
        view_root = spack_env_dir / ".spack-env" / "view"

        lines = ["module purge"]
        for path in module_paths:
            lines.append(f"module use {path}")
        if module_name:
            lines.append(f"module load {module_name}")
        if self.spack_cfg.get("spack_setup_script"):
            lines.append(f"source {self.spack_cfg['spack_setup_script']}")
        if self.spack_cfg.get("spack_env_dir"):
            lines.append(f"spack env activate {self.spack_cfg['spack_env_dir']}")
        lines.append(
            f"source {self.spack_cfg.get('thisroot', str(view_root / 'bin' / 'thisroot.sh'))}"
        )
        lines.append(
            f"source {self.spack_cfg.get('thisdd4hep', str(view_root / 'bin' / 'thisdd4hep.sh'))}"
        )
        detector_cfg = self.detector_cfg()
        if detector_cfg.get("source_script"):
            lines.append(f"source {detector_cfg['source_script']}")
        detector_repo = Path(detector_cfg.get("repo_root", "")).expanduser()
        detector_install = Path(detector_cfg.get("install_dir", detector_repo / "install"))
        ld_path = detector_cfg.get("ld_library_path", str(detector_install / "lib"))
        include_path = detector_cfg.get("root_include_path", str(detector_install / "include"))
        if ld_path:
            lines.append(f"export LD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH")
        if include_path:
            lines.append(f"export ROOT_INCLUDE_PATH={include_path}:$ROOT_INCLUDE_PATH")
        return "\n".join(lines)

    def update_compact(self, base_xml: Path, target_xml: Path, geom: GeometrySweep, value: str) -> None:
        tree = ET.parse(base_xml)
        root = tree.getroot()
        updated = False
        for const in root.findall(".//constant"):
            if const.attrib.get("name") == geom.parameter:
                const.set("value", value)
                updated = True
        if not updated:
            raise ValueError(f"Parameter {geom.parameter} not found in {base_xml}")
        target_xml.parent.mkdir(parents=True, exist_ok=True)
        if not self.dry_run:
            tree.write(target_xml)
        self.log(f"Created compact XML {target_xml} with {geom.parameter}={value}")

    def load_template(self, path: Path) -> str:
        return path.read_text()

    def fill_steering_template(self, template: str, replacements: dict) -> str:
        filled = template
        for key, value in replacements.items():
            filled = filled.replace(key, str(value))
        return filled

    def build_filter_suffix(self, filt: str, cut1: str, cut2: str) -> str:
        if filt in ("edep", "wvmin", "wvmax"):
            return f"{filt}{cut1}"
        if filt == "wvbtwn":
            return f"{filt}{cut1}-{cut2}"
        return "nofilt"

    def slurm_text(
        self,
        runtime_cfg: dict,
        job_name: str,
        steering_path: Path,
        log_dir: Path,
    ) -> str:
        scheduler_cfg = runtime_cfg.get("scheduler", {})
        slurm_cfg = scheduler_cfg.get("slurm", {})
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={log_dir}/{job_name}.out",
            f"#SBATCH --error={log_dir}/{job_name}.err",
            f"#SBATCH --nodes={slurm_cfg.get('nodes', 1)}",
            f"#SBATCH --ntasks={slurm_cfg.get('ntasks', 1)}",
            f"#SBATCH --cpus-per-task={slurm_cfg.get('cpus_per_task', 1)}",
            f"#SBATCH --mem={slurm_cfg.get('mem', self.default_mem())}",
            f"#SBATCH --time={slurm_cfg.get('time', '60:00:00')}",
        ]
        if slurm_cfg.get("partition"):
            lines.append(f"#SBATCH --partition={slurm_cfg['partition']}")
        if slurm_cfg.get("constraint"):
            lines.append(f"#SBATCH --constraint={slurm_cfg['constraint']}")
        if slurm_cfg.get("mail_type"):
            lines.append(f"#SBATCH --mail-type={slurm_cfg['mail_type']}")
        if slurm_cfg.get("mail_user"):
            lines.append(f"#SBATCH --mail-user={slurm_cfg['mail_user']}")

        lines.append("")
        lines.append(self.env_block())
        lines.append("")
        ddsim_exec = runtime_cfg.get("ddsim_executable", "ddsim")
        lines.append(f"{ddsim_exec} --steeringFile {steering_path}")
        lines.append("")
        return "\n".join(lines)

    def default_mem(self) -> str:
        detector = self.config.get("batch", {}).get("detector_label", "scepcal")
        return "12G" if detector == "scepcal" else "32G"

    def generate_runs(self) -> List[Path]:
        # Create compact/steering/slurm files for all geometry/seed/particle combinations.
        runtime_cfg = self.config.get("runtime", {})
        batch_cfg = self.config.get("batch", {})
        detector_cfg = self.detector_cfg()

        runs_dir = Path(runtime_cfg.get("runs_dir", self.root / "runs")).expanduser()
        output_dir = runs_dir / "output"
        log_dir = runs_dir / "logs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        steering_template_path = Path(detector_cfg["steering_template"]).expanduser()
        compact_template_path = Path(detector_cfg["compact_xml"]).expanduser()

        geom = self.geometry()
        seeds = self.seeds().values()
        particles: Iterable[str] = batch_cfg.get("particles", ["neutron"])
        detector_label = batch_cfg.get("detector_label", "scepcal")

        filter_name = str(batch_cfg.get("filter", "") or "")
        filtcut1 = str(batch_cfg.get("filtercut1", ""))
        filtcut2 = str(batch_cfg.get("filtercut2", ""))
        filter_suffix = self.build_filter_suffix(filter_name, filtcut1, filtcut2)

        momentum = batch_cfg.get("momentum_GeV", 50)
        plusminus_percent = batch_cfg.get("plusminus_percent", 0)
        plusminus_fraction = float(plusminus_percent) / 100.0
        n_events = batch_cfg.get("events", 1000)
        theta_min = batch_cfg.get("theta_min", 60)
        theta_max = batch_cfg.get("theta_max", 120)

        template_content = self.load_template(steering_template_path)

        slurm_files: List[Path] = []
        for formatted_value, seed, particle in [
            (val, seed, particle)
            for val in geom.formatted_values()
            for seed in seeds
            for particle in particles
        ]:
            geom_slug = geom.slug(formatted_value)
            run_prefix = (
                f"{detector_label}_{particle}_"
                f"{momentum}x{plusminus_percent}GeV_"
                f"{geom.parameter}-{geom_slug}_"
                f"s{seed}_N{n_events}_{filter_suffix}"
            )

            steering_path = runs_dir / f"{run_prefix}.py"
            slurm_path = runs_dir / f"{run_prefix}.slurm"
            compact_copy_path = runs_dir / f"{run_prefix}_{compact_template_path.name}"
            output_file = output_dir / f"{run_prefix}.root"

            self.update_compact(compact_template_path, compact_copy_path, geom, formatted_value)

            replacements = {
                "seedplaceholder": seed,
                "filtcut1placeholder": filtcut1,
                "filtcut2placeholder": filtcut2,
                "filterplaceholder": filter_name,
                "particleplaceholder": particle,
                "neventsplaceholder": n_events,
                "momentumplaceholder": momentum,
                "plusminusplaceholder": plusminus_fraction,
                "thetaminplaceholder": theta_min,
                "thetamaxplaceholder": theta_max,
                "compactfileplaceholder": str(compact_copy_path),
                "outputfileplaceholder": str(output_file),
            }
            steering_content = self.fill_steering_template(template_content, replacements)
            if not self.dry_run:
                steering_path.write_text(steering_content)
            self.log(f"Wrote steering {steering_path}")

            slurm_content = self.slurm_text(runtime_cfg, run_prefix, steering_path, log_dir)
            if not self.dry_run:
                slurm_path.write_text(slurm_content)
            self.log(f"Wrote slurm {slurm_path}")
            slurm_files.append(slurm_path)

        return slurm_files

    # ---------- submission -----------------------------------------------------
    def submit_jobs(self, slurm_files: List[Path]) -> None:
        # Submit generated slurm scripts if enabled
        if not self.submit_enabled:
            self.log("Submission disabled; skipping sbatch.")
            return
        for slurm_file in slurm_files:
            cmd = f"sbatch {slurm_file}"
            self.run_bash(cmd, cwd=slurm_file.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate detector build and batch submission for geometry sweeps."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--submit", action="store_true", help="Submit jobs via sbatch.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--force-install", action="store_true", help="Ignore spack env lock and reinstall."
    )
    parser.add_argument(
        "--force-build", action="store_true", help="Ignore detector build lock and rebuild."
    )
    parser.add_argument(
        "--make-threads",
        type=int,
        help="Number of parallel threads for detector build (default: CPU count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    config = yaml.safe_load(cfg_path.read_text())

    workflow = Workflow(config, args)
    workflow.ensure_spack_checkouts()
    workflow.install_spack_env()
    workflow.build_detector()
    slurm_files = workflow.generate_runs()
    workflow.submit_jobs(slurm_files)


if __name__ == "__main__":
    main()
