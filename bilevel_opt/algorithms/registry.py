"""Algorithm and score registries - looked up by name from config."""

from bilevel_opt.algorithms.seeded_radius_cog import (
    SeededRadiusCoG, SNREnergyScore, SNREnergyAccScore, EResolutionAccScore, EResolutionTrueScore)

ALGORITHMS = {
    "seeded_radius_cog": SeededRadiusCoG,
}

SCORES = {
    "snr_energy": SNREnergyScore,
    "snr_energy_acc": SNREnergyAccScore,
    "eres_acc": EResolutionAccScore,
    "eres_true": EResolutionTrueScore,
}
