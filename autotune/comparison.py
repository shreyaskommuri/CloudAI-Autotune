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


@dataclass(frozen=True)
class RegressionCheck:
    latest: Optional[Experiment]
    previous: Optional[Experiment]
    throughput_delta_pct: Optional[float]
    latency_delta_ms: Optional[float]

    @property
    def regressed(self) -> bool:
        """True if the latest run is worse than the run immediately before it.

        Distinct from RunComparison: a run can be worse than its immediate
        predecessor without being the worst run ever recorded, and that's the
        signal this catches.
        """
        throughput_dropped = self.throughput_delta_pct is not None and self.throughput_delta_pct < 0
        latency_grew = self.latency_delta_ms is not None and self.latency_delta_ms > 0
        return throughput_dropped or latency_grew


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


def compare_latest_to_previous(experiments: list[Experiment]) -> RegressionCheck:
    """Compare the latest completed run with the completed run immediately before it."""
    completed = sorted((exp for exp in experiments if exp.status == "completed"), key=lambda exp: exp.id or 0)
    latest = completed[-1] if completed else None
    previous = completed[-2] if len(completed) >= 2 else None

    return RegressionCheck(
        latest=latest,
        previous=previous,
        throughput_delta_pct=_throughput_delta_pct(latest, previous),
        latency_delta_ms=_metric_delta(latest, previous, "latency_ms"),
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
    baseline: Optional[Experiment],
) -> Optional[float]:
    latest_value = _metric(latest, "throughput_tokens_per_sec")
    baseline_value = _metric(baseline, "throughput_tokens_per_sec")
    if latest_value is None or baseline_value in (None, 0):
        return None
    return (latest_value - baseline_value) / baseline_value * 100


def _metric_delta(
    latest: Optional[Experiment],
    baseline: Optional[Experiment],
    metric: str,
) -> Optional[float]:
    latest_value = _metric(latest, metric)
    baseline_value = _metric(baseline, metric)
    if latest_value is None or baseline_value is None:
        return None
    return latest_value - baseline_value


def _metric(exp: Optional[Experiment], metric: str) -> Optional[float]:
    if exp is None:
        return None
    try:
        return float(exp.metrics[metric])
    except (KeyError, TypeError, ValueError):
        return None
