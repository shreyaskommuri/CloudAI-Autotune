"""Compare completed experiments for dashboard and report summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from autotune.database import Experiment


@dataclass(frozen=True)
class RunComparison:
    best: Optional[Experiment]
    latest: Optional[Experiment]
    throughput_delta_pct: Optional[float]
    latency_delta_ms: Optional[float]


def compare_best_and_latest(
    experiments: list[Experiment],
    latency_budget_ms: Optional[float] = None,
) -> RunComparison:
    """Compare the latest completed run with the best completed throughput run."""
    completed = [exp for exp in experiments if exp.status == "completed"]
    latest = max(completed, key=lambda exp: exp.id or 0, default=None)
    best = _best_throughput_run(completed, latency_budget_ms=latency_budget_ms)

    return RunComparison(
        best=best,
        latest=latest,
        throughput_delta_pct=_throughput_delta_pct(latest, best),
        latency_delta_ms=_metric_delta(latest, best, "latency_ms"),
    )


def _best_throughput_run(
    experiments: list[Experiment],
    latency_budget_ms: Optional[float],
) -> Optional[Experiment]:
    candidates = []
    for exp in experiments:
        throughput = _metric(exp, "throughput_tokens_per_sec")
        if throughput is None:
            continue
        latency = _metric(exp, "latency_ms")
        if latency_budget_ms is not None and (latency is None or latency > latency_budget_ms):
            continue
        candidates.append(exp)
    return max(
        candidates,
        key=lambda exp: (_metric(exp, "throughput_tokens_per_sec") or float("-inf"), exp.id or 0),
        default=None,
    )


def _throughput_delta_pct(
    latest: Optional[Experiment],
    best: Optional[Experiment],
) -> Optional[float]:
    latest_value = _metric(latest, "throughput_tokens_per_sec")
    best_value = _metric(best, "throughput_tokens_per_sec")
    if latest_value is None or best_value in (None, 0):
        return None
    return (latest_value - best_value) / best_value * 100


def _metric_delta(
    latest: Optional[Experiment],
    best: Optional[Experiment],
    metric: str,
) -> Optional[float]:
    latest_value = _metric(latest, metric)
    best_value = _metric(best, metric)
    if latest_value is None or best_value is None:
        return None
    return latest_value - best_value


def _metric(exp: Optional[Experiment], metric: str) -> Optional[float]:
    if exp is None:
        return None
    try:
        return float(exp.metrics[metric])
    except (KeyError, TypeError, ValueError):
        return None
