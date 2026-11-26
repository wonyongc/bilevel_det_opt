"""
Convert WarpX particle dump (AMReX plotfile) -> HepMC3 ASCII.

ddsim accepts HepMC2/3 via --inputFiles. We package the secondary particles
produced during the IP collision (beamstrahlung photons, coherent pairs,
incoherent pairs) as final-state particles in a synthetic HepMC event whose
"interaction vertex" is the IP at (0, 0, 0, 0).

We chunk into one HepMC event per N particles so that ddsim's overlay
machinery can consume them on a per-bunch-crossing basis.

Dependencies:
    openpmd-api  — to read AMReX plotfiles
    pyhepmc      — to write HepMC3

Both are pip-installable. If unavailable we fall back to a hand-rolled
HepMC3 ASCII writer (slower but dependency-free).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

_M_E_GEV = 0.000510998950
_C_M_PER_S = 299792458.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def convert(
    *,
    warpx_dump: Path,
    out_path: Path,
    beam_energy_GeV: float,
    process_label: str,
    particles_per_event: int = 5_000,
) -> None:
    """Convert a WarpX plotfile to HepMC3.

    Parameters
    ----------
    warpx_dump
        Path to a single AMReX plotfile directory (e.g. diags/diag1/000400).
    out_path
        Destination .hepmc file.
    beam_energy_GeV
        Used only for HepMC event-info / cross-section bookkeeping.
    process_label
        Free-form label written into HepMC weight names.
    particles_per_event
        Number of secondary particles to bundle into each HepMC event,
        which corresponds roughly to one bunch crossing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Reading WarpX dump %s", warpx_dump)
    particles = list(_read_warpx_particles(warpx_dump))
    log.info("Read %d secondary particles", len(particles))

    log.info("Writing HepMC3 -> %s", out_path)
    _write_hepmc3(
        particles=particles,
        out_path=out_path,
        beam_energy_GeV=beam_energy_GeV,
        process_label=process_label,
        particles_per_event=particles_per_event,
    )
    log.info("MDI HepMC ready: %s", out_path)


# ---------------------------------------------------------------------------
# WarpX reader
# ---------------------------------------------------------------------------

def _read_warpx_particles(plotfile: Path) -> Iterable[dict]:
    """Yield {pdg_id, px, py, pz, E, x, y, z, t} dicts from a plotfile.

    Tries openpmd-api first, falls back to yt if available. Either gives us
    SoA arrays of particle positions and momenta per species.
    """
    try:
        import openpmd_api as io  # type: ignore
    except ImportError:
        log.warning("openpmd-api not installed; trying yt fallback.")
        yield from _read_with_yt(plotfile)
        return

    # Detect openPMD file extension WarpX wrote. Newer ADIOS2 builds default
    # to .bp5; older to .bp; if HDF5 backend, .h5. Try each pattern in turn.
    pattern_path = None
    for ext in ("bp5", "bp", "h5"):
        candidates = list(plotfile.glob(f"openpmd_*.{ext}"))
        if candidates:
            pattern_path = str(plotfile / f"openpmd_%T.{ext}")
            log.info(f"openPMD series pattern: {pattern_path}")
            break
    if pattern_path is None:
        raise RuntimeError(
            f"No openPMD files (.bp5/.bp/.h5) found under {plotfile}. "
            f"Contents: {sorted(p.name for p in plotfile.iterdir())}"
        )
    series = io.Series(pattern_path, io.Access.read_only)
    iterations = list(series.iterations)
    if not iterations:
        raise RuntimeError(f"No iterations in {plotfile}")
    it = series.iterations[iterations[-1]]

    pdg_for_species = {
        "photons": 22,
        "pairs_e": 11,
        "pairs_p": -11,
    }

    for species_name, pdg in pdg_for_species.items():
        if species_name not in it.particles:
            continue
        sp = it.particles[species_name]

        x = sp["position"]["x"].load_chunk()
        y = sp["position"]["y"].load_chunk()
        z = sp["position"]["z"].load_chunk()
        ux = sp["momentum"]["x"].load_chunk()  # in kg*m/s
        uy = sp["momentum"]["y"].load_chunk()
        uz = sp["momentum"]["z"].load_chunk()
        series.flush()

        # Convert momentum from SI (kg*m/s) to GeV/c
        # p[GeV/c] = p[kg*m/s] * c / e * 1e-9
        scale = _C_M_PER_S / 1.602176634e-19 * 1e-9
        for i in range(len(x)):
            px = float(ux[i]) * scale
            py = float(uy[i]) * scale
            pz = float(uz[i]) * scale
            mass = 0.0 if pdg == 22 else _M_E_GEV
            E = math.sqrt(px * px + py * py + pz * pz + mass * mass)
            yield {
                "pdg": pdg,
                "px": px, "py": py, "pz": pz, "E": E, "m": mass,
                "x_mm": float(x[i]) * 1e3,
                "y_mm": float(y[i]) * 1e3,
                "z_mm": float(z[i]) * 1e3,
                "t_mm": 0.0,  # ddsim overlay treats t=0 as bunch crossing
            }


def _read_with_yt(plotfile: Path) -> Iterable[dict]:
    """Fallback reader using the yt library for AMReX plotfiles."""
    import yt  # type: ignore

    ds = yt.load(str(plotfile))
    ad = ds.all_data()

    pdg_for_species = {
        "photons": 22,
        "pairs_e": 11,
        "pairs_p": -11,
    }
    scale = _C_M_PER_S / 1.602176634e-19 * 1e-9

    for species_name, pdg in pdg_for_species.items():
        try:
            x = ad[(species_name, "particle_position_x")].to("m").value
            y = ad[(species_name, "particle_position_y")].to("m").value
            z = ad[(species_name, "particle_position_z")].to("m").value
            ux = ad[(species_name, "particle_momentum_x")].value
            uy = ad[(species_name, "particle_momentum_y")].value
            uz = ad[(species_name, "particle_momentum_z")].value
        except Exception as exc:
            log.debug("Skipping species %s: %s", species_name, exc)
            continue

        for i in range(len(x)):
            px = float(ux[i]) * scale
            py = float(uy[i]) * scale
            pz = float(uz[i]) * scale
            mass = 0.0 if pdg == 22 else _M_E_GEV
            E = math.sqrt(px * px + py * py + pz * pz + mass * mass)
            yield {
                "pdg": pdg,
                "px": px, "py": py, "pz": pz, "E": E, "m": mass,
                "x_mm": float(x[i]) * 1e3,
                "y_mm": float(y[i]) * 1e3,
                "z_mm": float(z[i]) * 1e3,
                "t_mm": 0.0,
            }


# ---------------------------------------------------------------------------
# HepMC3 writer
# ---------------------------------------------------------------------------

def _write_hepmc3(
    *,
    particles: list[dict],
    out_path: Path,
    beam_energy_GeV: float,
    process_label: str,
    particles_per_event: int,
) -> None:
    """Write particles as HepMC3 ASCII, chunked into events.

    Tries pyhepmc; falls back to a manual ASCII writer matching the HepMC3
    v3 format spec. The manual writer is sufficient for ddsim's reader and
    avoids forcing pyhepmc into the install graph.
    """
    try:
        import pyhepmc  # type: ignore
        _write_with_pyhepmc(particles, out_path, beam_energy_GeV,
                            process_label, particles_per_event)
        return
    except ImportError:
        pass

    _write_hepmc3_manual(particles, out_path, beam_energy_GeV,
                         process_label, particles_per_event)


def _write_with_pyhepmc(particles, out_path, beam_energy_GeV,
                        process_label, particles_per_event):
    import pyhepmc

    with pyhepmc.open(out_path, "w") as f:
        for ev_idx, chunk in enumerate(_chunks(particles, particles_per_event)):
            ev = pyhepmc.GenEvent(
                momentum_unit=pyhepmc.Units.GEV,
                length_unit=pyhepmc.Units.MM,
            )
            ev.event_number = ev_idx
            v = pyhepmc.GenVertex((0.0, 0.0, 0.0, 0.0))
            ev.add_vertex(v)
            for p in chunk:
                gp = pyhepmc.GenParticle(
                    (p["px"], p["py"], p["pz"], p["E"]),
                    p["pdg"],
                    1,  # status = 1 (final state)
                )
                gp.generated_mass = p["m"]
                v.add_particle_out(gp)
            ev.weights = [1.0]
            f.write(ev)


def _write_hepmc3_manual(particles, out_path, beam_energy_GeV,
                         process_label, particles_per_event):
    """Hand-rolled HepMC3 v3 ASCII writer.

    Format reference: https://gitlab.cern.ch/hepmc/HepMC3 -> doc/AsciiV3
    Each event:
        E <evt_num> <n_vertices> <n_particles>
        U GEV MM
        W <weight>
        V -1 0 [x y z t] @ 0
        P <id> <vtx_id> <pdg> <px> <py> <pz> <E> <m> <status>
    """
    with out_path.open("w") as f:
        f.write("HepMC::Version 3.02.06\n")
        f.write("HepMC::Asciiv3-START_EVENT_LISTING\n")

        for ev_idx, chunk in enumerate(_chunks(particles, particles_per_event)):
            n_part = len(chunk)
            f.write(f"E {ev_idx} 1 {n_part}\n")
            f.write("U GEV MM\n")
            f.write("W 1.0\n")
            f.write(f"A 0 signal_process_id {hash(process_label) & 0xFFFF}\n")
            # One vertex at the IP. Negative ID = production vertex.
            f.write("V -1 0 [] @ 0 0 0 0\n")
            for pid, p in enumerate(chunk, start=1):
                f.write(
                    f"P {pid} -1 {p['pdg']} "
                    f"{p['px']:.6e} {p['py']:.6e} {p['pz']:.6e} "
                    f"{p['E']:.6e} {p['m']:.6e} 1\n"
                )

        f.write("HepMC::Asciiv3-END_EVENT_LISTING\n")


def _chunks(seq, n):
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
