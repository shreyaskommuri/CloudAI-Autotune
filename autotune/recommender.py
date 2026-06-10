"""Suggest the next benchmark config to try based on prior experiment history.

Heuristic (v1, MVP): track a single tunable knob (default `serving.batch_size`)
across completed runs. If throughput is still climbing faster than latency is
degrading (i.e. the tokens/sec-per-ms-of-latency ratio is improving or stable),
recommend doubling the knob; otherwise recommend backing off to the best
observed tradeoff point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from autotune.database import Experiment

DEFAULT_KNOB = "serving.batch_size"


@dataclass
class Recommendation:
    knob: str
    current_value: Optional[float]
    suggested_value: Optional[float]
    reason: str


def _knob_value(exp: Experiment, knob: str) -> Optional[float]:
    node = exp.config
    for part in knob.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def _efficiency(exp: Experiment) -> Optional[float]:
    """tokens/sec per ms of latency — higher is a better throughput/latency tradeoff."""
    throughput = exp.metrics.get("throughput_tokens_per_sec")
    latency = exp.metrics.get("latency_ms")
    if throughput is None or latency in (None, 0):
        return None
    return throughput / latency


def recommend_next(
    experiments: list[Experiment],
    knob: str = DEFAULT_KNOB,
    latency_budget_ms: Optional[float] = None,
) -> Recommendation:
    """Recommend the next value to try for `knob` based on completed runs.

    `latency_budget_ms`, if given, caps how far latency is allowed to grow —
    runs over budget are treated as regressions regardless of throughput gains.
    """
    completed = [
        e
        for e in experiments
        if e.status == "completed" and _knob_value(e, knob) is not None and e.metrics
    ]
    if not completed:
        return Recommendation(
            knob=knob,
            current_value=None,
            suggested_value=None,
            reason="No completed experiments with metrics yet — run a baseline first.",
        )

    completed.sort(key=lambda e: _knob_value(e, knob))
    tried_values = {_knob_value(e, knob) for e in completed}
    latest = completed[-1]
    latest_value = _knob_value(latest, knob)
    latest_throughput = latest.metrics.get("throughput_tokens_per_sec")
    latest_latency = latest.metrics.get("latency_ms")

    if latency_budget_ms is not None and (latest_latency or 0) > latency_budget_ms:
        # Latest run blew the latency budget — recommend the best prior tradeoff.
        candidates = [e for e in completed if (e.metrics.get("latency_ms") or 0) <= latency_budget_ms]
        if candidates:
            best = max(candidates, key=lambda e: _efficiency(e) or 0)
            best_value = _knob_value(best, knob)
            suggested = _untried_between(
                best_value,
                latest_value,
                tried_values,
            ) or _next_lower_untried(best_value, tried_values)
            return Recommendation(
                knob=knob,
                current_value=latest_value,
                suggested_value=suggested,
                reason=(
                    f"{knob}={latest_value} pushed latency to {latest_latency:.0f}ms, "
                    f"over the {latency_budget_ms:.0f}ms budget. Best observed tradeoff "
                    f"within budget was {knob}={best_value} "
                    f"({best.metrics.get('throughput_tokens_per_sec'):.0f} tok/s @ "
                    f"{best.metrics.get('latency_ms'):.0f}ms). Try untested {knob}="
                    f"{suggested} near that safer region."
                ),
            )
        smallest_value = _knob_value(completed[0], knob)
        suggested = _next_lower_untried(smallest_value, tried_values)
        return Recommendation(
            knob=knob,
            current_value=latest_value,
            suggested_value=suggested,
            reason=(
                f"All runs exceed the {latency_budget_ms:.0f}ms latency budget. "
                f"Try untested {knob}={suggested}, smaller than the lowest value "
                f"already tried ({smallest_value})."
            ),
        )

    if len(completed) == 1:
        suggested = _next_higher_untried(latest_value, tried_values)
        return Recommendation(
            knob=knob,
            current_value=latest_value,
            suggested_value=suggested,
            reason=(
                f"Only one data point so far ({knob}={latest_value} -> "
                f"{latest_throughput:.0f} tok/s @ {latest_latency:.0f}ms). "
                f"Try doubling to {knob}={suggested} to see how throughput scales."
            ),
        )

    prev = completed[-2]
    prev_throughput = prev.metrics.get("throughput_tokens_per_sec")
    prev_latency = prev.metrics.get("latency_ms")

    throughput_gain = _pct_change(prev_throughput, latest_throughput)
    latency_gain = _pct_change(prev_latency, latest_latency)

    if throughput_gain is not None and latency_gain is not None and throughput_gain > latency_gain:
        suggested = _next_higher_untried(latest_value, tried_values)
        return Recommendation(
            knob=knob,
            current_value=latest_value,
            suggested_value=suggested,
            reason=(
                f"{knob} {prev_throughput:.0f}->{latest_throughput:.0f} tok/s "
                f"({throughput_gain:+.0%}) outpaced latency growth "
                f"{prev_latency:.0f}->{latest_latency:.0f}ms ({latency_gain:+.0%}). "
                f"Try {knob}={suggested} — throughput is still scaling well."
            ),
        )

    best = max(completed, key=lambda e: _efficiency(e) or 0)
    best_value = _knob_value(best, knob)
    suggested = _untried_between(best_value, latest_value, tried_values) or _nearest_untried(
        best_value,
        tried_values,
    )
    return Recommendation(
        knob=knob,
        current_value=latest_value,
        suggested_value=suggested,
        reason=(
            f"Latency is now growing faster than throughput "
            f"({latency_gain:+.0%} vs {throughput_gain:+.0%}). "
            f"Best tradeoff so far is {knob}={best_value} "
            f"({best.metrics.get('throughput_tokens_per_sec'):.0f} tok/s @ "
            f"{best.metrics.get('latency_ms'):.0f}ms) — try untested {knob}="
            f"{suggested} nearby."
        ),
    )


def _pct_change(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old in (None, 0) or new is None:
        return None
    return (new - old) / old


def _next_higher_untried(value: Optional[float], tried_values: set[Optional[float]]) -> Optional[float]:
    if value is None:
        return None
    suggested = value * 2
    while suggested in tried_values:
        suggested *= 2
    return suggested


def _next_lower_untried(value: Optional[float], tried_values: set[Optional[float]]) -> Optional[float]:
    if value is None:
        return None
    suggested = value / 2
    while suggested in tried_values and suggested > 0:
        suggested /= 2
    return suggested if suggested > 0 else None


def _untried_between(
    low: Optional[float],
    high: Optional[float],
    tried_values: set[Optional[float]],
) -> Optional[float]:
    if low is None or high is None or low == high:
        return None
    suggested = (low + high) / 2
    return suggested if suggested not in tried_values else None


def _nearest_untried(value: Optional[float], tried_values: set[Optional[float]]) -> Optional[float]:
    if value is None:
        return None
    for candidate in (value / 2, value * 1.5, value * 2):
        if candidate > 0 and candidate not in tried_values:
            return candidate
    return _next_higher_untried(value, tried_values)
