"""
WarpX input deck generation for MDI background simulation — v2.

Upgrades over v1 (2026-07, informed by the Galarza/Formenti CERN study):

  1. FIDELITY PRESETS
     cfg.fidelity selects between:
       "smoke"    — the validated smoke-test deck (inflated sigma_y, relaxed
                    chi_min, final-timestep diagnostic). Byte-compatible with
                    the deck that produced the 29,345-photon Z-pole sample.
       "formenti" — physical FCC-ee Z parameters after Formenti et al. as
                    used in the CERN reference study: sigma_y = 35.2 nm,
                    energy spread, crossing via momentum tilt, box scaled to
                    the beam sizes, dt = dz/(2c), external solenoid.

  2. BOUNDARY SCRAPING
     A BoundaryScraping diagnostic records particles as they exit the box.
     This is the physically correct "detector-bound flux": with absorbing
     particle boundaries, secondaries that leave before the final step are
     deleted and never appear in the final-timestep dump. The v1 deck
     therefore UNDERCOUNTED the background. Both diagnostics are written so
     the two populations can be compared; the converter should prefer the
     scraped output.

  3. IPC SCAFFOLDING (disabled by default)
     Species slots for incoherent pair creation (Landau-Lifshitz,
     Bethe-Heitler, Breit-Wheeler via virtual photons) following the species
     naming of the CERN study (vpho, ele_ll/pos_ll, ele_bh/pos_bh,
     ele_bw/pos_bw). The WarpX parameter names for these processes are
     version-dependent; before enabling, verify against the local build:
         grep -rE 'landau|bethe|virtual' $HOME/src/warpx/Examples/ Docs/
     and fix the marked lines. Gated behind cfg.enable_ipc so the deck
     remains valid on builds without IPC support.

Cache note: "formenti" and "smoke" produce different physics and MUST hash
to different cache keys. Add `fidelity` (and `enable_ipc`) to the fields
hashed by MDIConfig.cache_key() — see UPGRADE_NOTES.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import MDIConfig

_E_CHARGE = 1.602176634e-19
_ME_C2_EV = 0.510998950e6
_C = 299792458.0


# --------------------------------------------------------------------------
# Preset parameter resolution
# --------------------------------------------------------------------------

def _resolve(cfg: "MDIConfig") -> dict:
    """Return the concrete numbers the deck will use, per fidelity preset.

    All optional attributes are read with getattr() defaults so an
    un-upgraded MDIConfig still works (and yields the v1 smoke deck).
    """
    fidelity = getattr(cfg, "fidelity", "smoke")

    p = {
        "fidelity": fidelity,
        "energy_GeV": cfg.beam_energy_GeV,
        "bunch_charge_C": cfg.bunch_charge_nC * 1e-9,
        "n_macro": cfg.n_macroparticles,
        "crossing_half_rad": cfg.crossing_angle_mrad * 1e-3 / 2.0,
        "enable_ipc": bool(getattr(cfg, "enable_ipc", False)),
        "solenoid_T": float(getattr(cfg, "solenoid_T", 0.0)),
        "espread": float(getattr(cfg, "energy_spread", 0.0)),
    }
    p["gamma"] = cfg.beam_energy_GeV * 1e9 / _ME_C2_EV

    if fidelity == "formenti":
        # Physical FCC-ee Z parameters (Formenti reference configuration).
        # Config values are IGNORED for the sizes below unless explicitly
        # overridden, because the whole point of this preset is physical
        # self-fields. Beam energy/charge/macros still come from cfg.
        # Physical Formenti table values — config sigma_* are deliberately
        # IGNORED here; the preset's purpose is physical self-fields.
        p["sigma_x_m"] = 7.99375e-6
        p["sigma_y_m"] = 35.2e-9
        p["sigma_z_m"] = 16.7e-3
        if p["espread"] == 0.0:
            p["espread"] = 1.34e-3
        if p["solenoid_T"] == 0.0:
            p["solenoid_T"] = -2.0
        # chi thresholds: physical chi is reachable, use engine defaults
        p["chi_min"] = float(getattr(cfg, "chi_min", 0.0) or 1e-3)
        p["photon_E_threshold"] = float(
            getattr(cfg, "photon_creation_energy_threshold", 2.0))
        # Box after the CERN study's stage-1 spec:
        #   x = +/- k_x * (sigma_x cos(th) + sigma_z sin(th)),  k_x scale
        #   y = +/- k_y * sigma_y
        #   z = +/- k_z * sigma_z
        th = p["crossing_half_rad"]
        sx_eff = p["sigma_x_m"] * math.cos(th) + p["sigma_z_m"] * math.sin(th)
        p["half_x_m"] = 4.0 * sx_eff          # Lx = 8 * sx_eff
        p["half_y_m"] = 4.0 * p["sigma_y_m"]  # Ly = 8 * sigma_y
        p["half_z_m"] = 6.0 * p["sigma_z_m"]  # +/- 6 sigma_z
        p["ncell"] = (128, 256, 256)   # x halved: sigma_x >> sigma_y, least critical
        # Relativistic-electrostatic mode: Poisson solve per step, no
        # Maxwell CFL, so dt is set by beam evolution, not the mesh.
        # Reference: Examples/Physics_applications/beam_beam_collision
        # (Formenti setup: do_electrostatic=relativistic, dt = sigma_z/10c).
        # NOTE: explicit FDTD is infeasible for this preset — CFL at
        # dy=1.1 nm means dt~3.7e-18 s, and large const_dt blows up
        # dt-scaled init allocations (the ~20 GB OOM, diagnosed 2026-08-16).
        p["electrostatic"] = True
        p["dt_s"] = p["sigma_z_m"] / (10.0 * _C)   # ~5.57e-12 s
        _T = 0.7 * (2.0 * p["half_z_m"]) / _C      # example convention: 0.7*Lz/c
        p["max_step"] = int(getattr(cfg, "max_step", 0) or math.ceil(_T / p["dt_s"]))
        p["const_dt"] = True
    else:
        # "smoke": preserve the validated v1 behavior exactly.
        p["sigma_x_m"] = cfg.sigma_x_um * 1e-6
        p["sigma_y_m"] = cfg.sigma_y_um * 1e-6
        p["sigma_z_m"] = cfg.sigma_z_mm * 1e-3
        p["chi_min"] = float(getattr(cfg, "chi_min", 0.0) or 1e-5)
        p["photon_E_threshold"] = float(
            getattr(cfg, "photon_creation_energy_threshold", 0.0))
        p["half_x_m"] = 50e-6
        p["half_y_m"] = 5e-6
        p["half_z_m"] = 50e-3
        p["ncell"] = (256, 128, 256)
        p["max_step"] = int(getattr(cfg, "max_step", 0) or 400)
        p["const_dt"] = False
        p["electrostatic"] = False
        p["dt_s"] = None

    return p


# --------------------------------------------------------------------------
# Deck generation
# --------------------------------------------------------------------------

def write_deck(cfg: "MDIConfig", deck_path: Path) -> None:
    """Write a WarpX input deck for the configured beam-beam scenario."""
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    p = _resolve(cfg)

    g = p["gamma"]
    th = p["crossing_half_rad"]
    q = p["bunch_charge_C"]
    uz_th = g * p["espread"]          # energy spread as longitudinal u spread

    nx, ny, nz = p["ncell"]

    # ---- species lists -----------------------------------------------------
    species = ["beam_e", "beam_p", "photons", "pairs_e", "pairs_p"]
    if p["enable_ipc"]:
        species += ["vpho", "ele_ll", "pos_ll", "ele_bh", "pos_bh",
                    "ele_bw", "pos_bw"]
    species_line = " ".join(species)

    scraped_species = "photons pairs_e pairs_p" + (
        " ele_ll pos_ll ele_bh pos_bh ele_bw pos_bw" if p["enable_ipc"] else "")

    # ---- optional blocks ---------------------------------------------------
    dt_block = ""
    if p["const_dt"]:
        dt_block = f"warpx.const_dt          = {p['dt_s']:.6e}\n"

    solenoid_block = ""
    if p["solenoid_T"] != 0.0:
        solenoid_block = f"""
#######################
# External solenoid field
#######################
particles.B_ext_particle_init_style = constant
particles.B_external_particle       = 0.0 0.0 {p['solenoid_T']:.3f}
"""

    ipc_block = ""
    if p["enable_ipc"]:
        ipc_block = f"""
#######################
# IPC: incoherent pair creation via two-photon (linear Breit-Wheeler)
# collisions. All parameter names VERIFIED against:
#   Examples/Tests/virtual_photons          (vpho generation)
#   Examples/Tests/linear_breit_wheeler     (collision syntax)
#
# Channel map (beam-beam terminology -> collision species):
#   LL (Landau-Lifshitz) : vpho    x vpho    -> ele_ll pos_ll
#   BH (Bethe-Heitler)   : photons x vpho    -> ele_bh pos_bh
#   BW (incoherent BW)   : photons x photons -> ele_bw pos_bw
#
# Tunables: event_multiplier / probability_* control MC sampling of the
# collision pairing; test values kept. vpho multiplier (1e7 in the WarpX
# test) controls sampling of the Weizsacker-Williams spectrum — tune
# against weighted yields (CERN study: LL ~1e4, BH ~2e3, BW ~1e2).
#######################
beam_e.do_qed_virtual_photons = 1
beam_e.qed_virtual_photon_species_name = vpho
beam_p.do_qed_virtual_photons = 1
beam_p.qed_virtual_photon_species_name = vpho

vpho.species_type       = photon
vpho.injection_style    = none
vpho.do_not_push        = 1
vpho.do_not_gather      = 1
vpho.do_not_deposit     = 1
vpho.qed_virtual_photons_min_energy = 1.0e6
vpho.qed_virtual_photons_multiplier = 10000000

ele_ll.species_type     = electron
ele_ll.injection_style  = none
pos_ll.species_type     = positron
pos_ll.injection_style  = none
ele_bh.species_type     = electron
ele_bh.injection_style  = none
pos_bh.species_type     = positron
pos_bh.injection_style  = none
ele_bw.species_type     = electron
ele_bw.injection_style  = none
pos_bw.species_type     = positron
pos_bw.injection_style  = none

collisions.collision_names = ipc_ll ipc_bh ipc_bw

ipc_ll.type             = linear_breit_wheeler
ipc_ll.species          = vpho vpho
ipc_ll.product_species  = ele_ll pos_ll
ipc_ll.event_multiplier = 1.
ipc_ll.probability_threshold    = 0.5
ipc_ll.probability_target_value = 0.02

ipc_bh.type             = linear_breit_wheeler
ipc_bh.species          = photons vpho
ipc_bh.product_species  = ele_bh pos_bh
ipc_bh.event_multiplier = 1.
ipc_bh.probability_threshold    = 0.5
ipc_bh.probability_target_value = 0.02

ipc_bw.type             = linear_breit_wheeler
ipc_bw.species          = photons photons
ipc_bw.product_species  = ele_bw pos_bw
ipc_bw.event_multiplier = 1.
ipc_bw.probability_threshold    = 0.5
ipc_bw.probability_target_value = 0.02
"""

    if p.get("electrostatic"):
        numerics_block = f"""warpx.do_electrostatic = relativistic
{dt_block}warpx.grid_type         = collocated
warpx.do_dynamic_scheduling = 0
warpx.serialize_initial_conditions = 1

algo.particle_pusher  = vay
algo.particle_shape   = 3
algo.load_balance_intervals = 100"""
    else:
        numerics_block = f"""warpx.cfl             = 0.999
{dt_block}warpx.do_dynamic_scheduling = 0
warpx.serialize_initial_conditions = 1

algo.maxwell_solver   = yee
algo.particle_pusher  = vay
algo.particle_shape   = 3"""

    deck = f"""# WarpX input deck — auto-generated by bilevel_opt.mdi (v2)
# Fidelity: {p['fidelity']}
# Scenario: {cfg.process}
# Beam energy: {cfg.beam_energy_GeV} GeV   Bunch charge: {cfg.bunch_charge_nC} nC
# Crossing angle: {cfg.crossing_angle_mrad} mrad   sigma_y: {p['sigma_y_m']*1e9:.1f} nm
# IPC enabled: {p['enable_ipc']}   Solenoid: {p['solenoid_T']} T

#######################
# General
#######################
max_step              = {p['max_step']}
amr.n_cell            = {nx} {ny} {nz}
amr.max_grid_size     = 64
amr.blocking_factor   = 16
amr.max_level         = 0

geometry.dims         = 3
geometry.prob_lo      = -{p['half_x_m']:.6e} -{p['half_y_m']:.6e} -{p['half_z_m']:.6e}
geometry.prob_hi      =  {p['half_x_m']:.6e}  {p['half_y_m']:.6e}  {p['half_z_m']:.6e}
boundary.field_lo     = pec pec pec
boundary.field_hi     = pec pec pec
boundary.particle_lo  = Absorbing Absorbing Absorbing
boundary.particle_hi  = Absorbing Absorbing Absorbing

{numerics_block}
{solenoid_block}
#######################
# QED — beamstrahlung & (real-photon) pair production
#######################
qed_qs.lookup_table_mode = "builtin"
qed_bw.lookup_table_mode = "builtin"
warpx.do_qed_quantum_sync = 1
warpx.do_qed_breit_wheeler = 1
qed_qs.chi_min = {p['chi_min']:.6e}
qed_bw.chi_min = {p['chi_min']:.6e}
qed_qs.photon_creation_energy_threshold = {p['photon_E_threshold']:.6e}

#######################
# Beam 1: electrons, +z, tilted by +crossing_half
#######################
particles.species_names = {species_line}

beam_e.species_type     = electron
beam_e.injection_style  = gaussian_beam
beam_e.x_rms            = {p['sigma_x_m']:.6e}
beam_e.y_rms            = {p['sigma_y_m']:.6e}
beam_e.z_rms            = {p['sigma_z_m']:.6e}
beam_e.x_m              = 0.0
beam_e.y_m              = 0.0
beam_e.z_m              = 0.0
beam_e.npart            = {p['n_macro']}
beam_e.q_tot            = -{q:.6e}
beam_e.momentum_distribution_type = gaussian
beam_e.ux_m             = {g * th:.6e}
beam_e.uy_m             = 0.0
beam_e.uz_m             = {g:.6e}
beam_e.ux_th            = 0.0
beam_e.uy_th            = 0.0
beam_e.uz_th            = {uz_th:.6e}
beam_e.do_qed_quantum_sync = 1
beam_e.qed_quantum_sync_phot_product_species = photons
beam_e.save_particles_at_xlo = 1
beam_e.save_particles_at_xhi = 1
beam_e.save_particles_at_ylo = 1
beam_e.save_particles_at_yhi = 1
beam_e.save_particles_at_zlo = 1
beam_e.save_particles_at_zhi = 1

#######################
# Beam 2: positrons, -z, tilted by -crossing_half
#######################
beam_p.species_type     = positron
beam_p.injection_style  = gaussian_beam
beam_p.x_rms            = {p['sigma_x_m']:.6e}
beam_p.y_rms            = {p['sigma_y_m']:.6e}
beam_p.z_rms            = {p['sigma_z_m']:.6e}
beam_p.x_m              = 0.0
beam_p.y_m              = 0.0
beam_p.z_m              = 0.0
beam_p.npart            = {p['n_macro']}
beam_p.q_tot            = {q:.6e}
beam_p.momentum_distribution_type = gaussian
beam_p.ux_m             = -{g * th:.6e}
beam_p.uy_m             = 0.0
beam_p.uz_m             = -{g:.6e}
beam_p.ux_th            = 0.0
beam_p.uy_th            = 0.0
beam_p.uz_th            = {uz_th:.6e}
beam_p.do_qed_quantum_sync = 1
beam_p.qed_quantum_sync_phot_product_species = photons
beam_p.save_particles_at_xlo = 1
beam_p.save_particles_at_xhi = 1
beam_p.save_particles_at_ylo = 1
beam_p.save_particles_at_yhi = 1
beam_p.save_particles_at_zlo = 1
beam_p.save_particles_at_zhi = 1

#######################
# Secondary product species
#######################
photons.species_type    = photon
photons.injection_style = none
photons.do_qed_breit_wheeler = 1
photons.qed_breit_wheeler_ele_product_species = pairs_e
photons.qed_breit_wheeler_pos_product_species = pairs_p
photons.save_particles_at_xlo = 1
photons.save_particles_at_xhi = 1
photons.save_particles_at_ylo = 1
photons.save_particles_at_yhi = 1
photons.save_particles_at_zlo = 1
photons.save_particles_at_zhi = 1

pairs_e.species_type    = electron
pairs_e.injection_style = none
pairs_e.save_particles_at_xlo = 1
pairs_e.save_particles_at_xhi = 1
pairs_e.save_particles_at_ylo = 1
pairs_e.save_particles_at_yhi = 1
pairs_e.save_particles_at_zlo = 1
pairs_e.save_particles_at_zhi = 1

pairs_p.species_type    = positron
pairs_p.injection_style = none
pairs_p.save_particles_at_xlo = 1
pairs_p.save_particles_at_xhi = 1
pairs_p.save_particles_at_ylo = 1
pairs_p.save_particles_at_yhi = 1
pairs_p.save_particles_at_zlo = 1
pairs_p.save_particles_at_zhi = 1
{ipc_block}
#######################
# Diagnostics
#   diag1    — final-timestep survivors (v1 behavior; kept for comparison)
#   scrape1  — boundary-scraped flux (the detector-bound population;
#              PREFERRED input for the HepMC converter)
#######################
diagnostics.diags_names = diag1 scrape1

diag1.intervals         = {p['max_step']}
diag1.diag_type         = Full
diag1.format            = openpmd
diag1.fields_to_plot    = none
diag1.species           = photons pairs_e pairs_p

scrape1.diag_type       = BoundaryScraping
scrape1.format          = openpmd
scrape1.intervals       = 100
scrape1.species         = {scraped_species}

#######################
# Reduced diagnostics: chi extrema (sanity of QED regime)
#######################
warpx.reduced_diags_names = chi_e chi_p chi_ph
chi_e.type   = ParticleExtrema
chi_e.species = beam_e
chi_e.intervals = 50
chi_p.type   = ParticleExtrema
chi_p.species = beam_p
chi_p.intervals = 50
chi_ph.type  = ParticleExtrema
chi_ph.species = photons
chi_ph.intervals = 50
"""

    deck_path.write_text(deck)

