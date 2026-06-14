"""Evaluate experiment metrics against pass/fail budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from autotune.database import Experiment


@dataclass(frozen=True)
class Budgets:
    latency_ms: Optional[float] = None
    min_throughput_tokens_per_sec: Optional[float] = None
    runtime_sec: Optional[float] = None
    failure_rate: Optional[float] = None

    def has_checks(self) -> bool:
        return any(
            value is not None
            for value in (
                self.latency_ms,
                self.min_throughput_tokens_per_sec,
                self.runtime_sec,
                self.failure_rate,
            )
        )


@dataclass(frozen=True)
class BudgetCheck:
    experiment_id: Optional[int]
    status: str
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def evaluate_experiment(exp: Experiment, budgets: Budgets) -> BudgetCheck:
    """Return pass/fail/unknown for one experiment against the provided budgets."""
    if exp.status != "completed":
        return BudgetCheck(exp.id, "unknown", (f"experiment status is {exp.status}",))
    if not budgets.has_checks():
        return BudgetCheck(exp.id, "unknown", ("no budgets provided",))

    failures: list[str] = []
    unknowns: list[str] = []

    _check_max(
        exp,
        metric="latency_ms",
        limit=budgets.latency_ms,
        label="latency",
        unit="ms",
        failures=failures,
        unknowns=unknowns,
    )
    _check_min(
        exp,
        metric="throughput_tokens_per_sec",
        limit=budgets.min_throughput_tokens_per_sec,
        label="throughput",
        unit="tok/s",
        failures=failures,
        unknowns=unknowns,
    )
    _check_max(
        exp,
        metric="runtime_sec",
        limit=budgets.runtime_sec,
        label="runtime",
        unit="s",
        failures=failures,
        unknowns=unknowns,
    )
    _check_max(
        exp,
        metric="failure_rate",
        limit=budgets.failure_rate,
        label="failure_rate",
        unit="",
        failures=failures,
        unknowns=unknowns,
    )

    if failures:
        return BudgetCheck(exp.id, "fail", tuple(failures))
    if unknowns:
        return BudgetCheck(exp.id, "unknown", tuple(unknowns))
    return BudgetCheck(exp.id, "pass", ("all provided budgets satisfied",))


def _metric_value(exp: Experiment, metric: str) -> Optional[float]:
    try:
        return float(exp.metrics[metric])
    except (KeyError, TypeError, ValueError):
        return None


def _check_max(
    exp: Experiment,
    metric: str,
    limit: Optional[float],
    label: str,
    unit: str,
    failures: list[str],
    unknowns: list[str],
) -> None:
    if limit is None:
        return
    value = _metric_value(exp, metric)
    if value is None:
        unknowns.append(f"{label} metric missing")
        return
    if value > limit:
        failures.append(f"{label} {value:g}{unit} exceeds budget {limit:g}{unit}")


def _check_min(
    exp: Experiment,
    metric: str,
    limit: Optional[float],
    label: str,
    unit: str,
    failures: list[str],
    unknowns: list[str],
) -> None:
    if limit is None:
        return
    value = _metric_value(exp, metric)
    if value is None:
        unknowns.append(f"{label} metric missing")
        return
    if value < limit:
        failures.append(f"{label} {value:g}{unit} is below budget {limit:g}{unit}")
