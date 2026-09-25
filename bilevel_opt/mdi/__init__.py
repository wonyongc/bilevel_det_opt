"""
Machine-Detector Interface (MDI) background generation via WarpX.

This subpackage runs WarpX to produce beam-induced backgrounds at the IP
(beamstrahlung pairs, synchrotron radiation, etc.), converts the resulting
particle dump into HepMC3, and exposes the path(s) so the outer loop can
pass them to ddsim as additional --inputFiles overlays alongside the signal.

When `mdi.scenarios` is set in the config, the outer loop sweeps over MDI
scenarios as an additional axis of the Cartesian product.

Public API:
    build_mdi_backgrounds(cfg, run_cfg) -> MDIArtifacts  (single scenario)
    build_mdi_scenarios(cfg, run_cfg) -> list[(MDIScenario, MDIArtifacts)]
    expand_scenarios(mdi_block) -> list[MDIScenario]
"""

from .runner import build_mdi_backgrounds, build_mdi_scenarios
from .runner import MDIArtifacts, MDIConfig
from .scenarios import MDIScenario, expand_scenarios

__all__ = [
    "build_mdi_backgrounds",
    "build_mdi_scenarios",
    "MDIArtifacts",
    "MDIConfig",
    "MDIScenario",
    "expand_scenarios",
]
