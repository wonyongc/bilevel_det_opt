"""
Expand an `mdi.scenarios` config block into concrete MDIConfig instances.

The outer loop iterates over the result, treating each scenario as another
axis of the Cartesian product alongside geometry, seeds, and particles.

Currently supports sweeping `beam_energy_GeV` (the option requested in the
config schema). The architecture below generalizes cleanly to other axes
(bunch charge, crossing angle, etc.) by adding entries to `_SWEEPABLE`.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

from .runner import MDIConfig


# Whitelist of MDIConfig fields that can appear under mdi.scenarios.
# Restricted on purpose: only fields here are valid sweep axes, and only
# they get included in the scenario label / cache key derivation. Adding a
# new axis is a one-line change here plus a unit conversion if needed.
_SWEEPABLE = {
    "beam_energy_GeV",
    # Future: "bunch_charge_nC", "crossing_angle_mrad"
}


@dataclasses.dataclass(frozen=True)
class MDIScenario:
    """One concrete operating point for MDI generation.

    Attributes
    ----------
    cfg:
        Full MDIConfig with sweep values resolved.
    label:
        Short identifier, e.g. "E45.6". Used in run directory names,
        manifest entries, and heatmap axis labels.
    params:
        Dict of just the swept parameters and their values, e.g.
        {"beam_energy_GeV": 45.6}. Surfaces in the manifest so downstream
        analysis can slice by scenario without re-parsing the label.
    """
    cfg: MDIConfig
    label: str
    params: dict


def expand_scenarios(mdi_block: dict | None) -> list[MDIScenario]:
    """Expand mdi.scenarios into a list of (cfg, label, params).

    If `mdi` is missing or disabled, returns a single placeholder scenario
    with `cfg.enabled=False` so the outer loop's product iteration stays
    uniform (always at least one MDI axis value, even if it's a no-op).

    If `mdi` is enabled but has no `scenarios` block, returns a single
    scenario built from the top-level `mdi` fields — i.e. backward-compatible
    with the original single-scenario design.
    """
    if not mdi_block or not mdi_block.get("enabled", False):
        return [MDIScenario(
            cfg=MDIConfig.from_dict(None),
            label="no_mdi",
            params={},
        )]

    base = MDIConfig.from_dict(mdi_block)
    scenarios_cfg = mdi_block.get("scenarios", {})

    if not scenarios_cfg:
        # Single-scenario back-compat path
        return [MDIScenario(cfg=base, label="default", params={})]

    # Validate axes
    unknown = set(scenarios_cfg) - _SWEEPABLE
    if unknown:
        raise ValueError(
            f"Unknown MDI sweep axes: {unknown}. "
            f"Sweepable: {sorted(_SWEEPABLE)}"
        )

    # Build per-axis value lists. Each entry is (axis_name, [values...]).
    axes: list[tuple[str, list]] = []
    for axis_name, spec in scenarios_cfg.items():
        values = _expand_axis(spec, axis_name)
        if not values:
            raise ValueError(f"MDI sweep axis {axis_name!r} expanded to empty list")
        axes.append((axis_name, values))

    # Cartesian product over MDI axes
    scenarios: list[MDIScenario] = []
    for combo in _cartesian(axes):
        params = dict(combo)
        cfg = dataclasses.replace(base, **params)
        scenarios.append(MDIScenario(
            cfg=cfg,
            label=_make_label(params),
            params=params,
        ))
    return scenarios


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_axis(spec, axis_name: str) -> list:
    """Accept either a {start, stop, step} dict or an explicit list.

    Matches the convention used by the geometry sweep in outer_loop.geometry,
    so users only have to learn one syntax.
    """
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        if "values" in spec:
            return list(spec["values"])
        if {"start", "stop", "step"} <= set(spec):
            start, stop, step = spec["start"], spec["stop"], spec["step"]
            out, v = [], start
            # Inclusive of stop with float tolerance
            while v <= stop + step * 1e-9:
                out.append(round(v, 9))
                v += step
            return out
        raise ValueError(
            f"MDI sweep axis {axis_name!r} must be a list, "
            f"{{values: [...]}}, or {{start, stop, step}}"
        )
    raise ValueError(f"MDI sweep axis {axis_name!r}: unsupported spec type {type(spec)}")


def _cartesian(axes: list[tuple[str, list]]) -> Iterator[tuple]:
    """Yield tuples of (axis_name, value) pairs covering the product."""
    if not axes:
        yield ()
        return
    head_name, head_vals = axes[0]
    for v in head_vals:
        for rest in _cartesian(axes[1:]):
            yield ((head_name, v),) + rest


def _make_label(params: dict) -> str:
    """Compact filesystem-safe label, e.g. {beam_energy_GeV: 45.6} -> 'E45.6'."""
    short = {
        "beam_energy_GeV": "E",
        "bunch_charge_nC": "Q",
        "crossing_angle_mrad": "X",
    }
    parts = []
    for k, v in sorted(params.items()):
        prefix = short.get(k, k)
        # Format: int if whole, else strip trailing zeros
        if isinstance(v, float) and v.is_integer():
            parts.append(f"{prefix}{int(v)}")
        else:
            parts.append(f"{prefix}{v}")
    return "_".join(parts) if parts else "default"
