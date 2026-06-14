"""Diff experiment configs and metrics for explainable benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autotune.database import Experiment


@dataclass(frozen=True)
class FieldDiff:
    key: str
    left: Any
    right: Any


@dataclass(frozen=True)
class ExperimentDiff:
    config: tuple[FieldDiff, ...]
    metrics: tuple[FieldDiff, ...]


def diff_experiments(left: Experiment, right: Experiment) -> ExperimentDiff:
    return ExperimentDiff(
        config=_diff_mapping(_flatten(left.config), _flatten(right.config)),
        metrics=_diff_mapping(left.metrics, right.metrics),
    )


def _diff_mapping(left: dict[str, Any], right: dict[str, Any]) -> tuple[FieldDiff, ...]:
    diffs = []
    for key in sorted(set(left) | set(right)):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value != right_value:
            diffs.append(FieldDiff(key=key, left=left_value, right=right_value))
    return tuple(diffs)


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat
