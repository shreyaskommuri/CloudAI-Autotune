"""Fit a response surface to observed combos instead of the heuristic trend proxy.

`recommend --joint --search` (`recommender.suggest_joint_step`) scores untried combos by how
many knobs land on their own *independently computed* trend direction — a heuristic proxy, not
a measurement, and it never learns from a suggestion turning out wrong beyond however that new
data point shifts the next independent trend calculation.

`suggest_joint_optimize` replaces that scoring with a quadratic response surface
(`efficiency ~ b0 + sum(bi*xi + ci*xi^2)`) fit via least squares over every completed, in-budget
combo tried so far — including combos that started life as `--optimize` suggestions themselves,
so a bad suggestion pulls the next fit away from predicting well nearby without any separate
demotion logic. With too little data to fit reliably, it falls back to the existing heuristic
(`suggest_joint_step`) rather than guessing — an implicit explore-first-then-exploit switch as
real data accumulates.

Suggestions are tracked in `ExperimentDB.search_suggestions` so an untried combo already
recommended in a prior call isn't recommended again while it's still pending.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from autotune.budgets import Budgets
from autotune.database import Experiment, SuggestionRecord
from autotune.recommender import (
    DEFAULT_SEARCH_NEIGHBORHOOD_CAP,
    ComboResult,
    _bounded_neighborhood,
    _combos_from_completed,
    _completed_for_knobs,
    _efficiency,
    _knob_value,
    format_combo,
    recommend_joint,
    suggest_joint_step,
)


@dataclass
class OptimizeRecommendation:
    knobs: list[str]
    best: Optional[ComboResult]
    suggested: Optional[dict[str, float]]
    predicted_efficiency: Optional[float]
    reason: str


def _fit_efficiency_surface(
    combos: list[ComboResult], knobs: list[str]
) -> Optional[Callable[[dict[str, float]], float]]:
    """Fit efficiency ~ b0 + sum(bi*xi + ci*xi^2) over `combos` via least squares.

    Returns a predictor, or None if there isn't enough data to fit reliably: fewer distinct
    combos than free parameters (a perfect but meaningless fit) or a rank-deficient design
    matrix (e.g. only two distinct values tried for some knob, making its linear and squared
    terms collinear).
    """
    num_params = 1 + 2 * len(knobs)
    if len(combos) < num_params + 1:
        return None

    def _features(values: dict[str, float]) -> list[float]:
        row = [1.0]
        for knob in knobs:
            x = values[knob]
            row.append(x)
            row.append(x * x)
        return row

    design = np.array([_features(combo.values) for combo in combos])
    targets = np.array([combo.efficiency for combo in combos])
    try:
        coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < num_params:
        return None

    def predict(values: dict[str, float]) -> float:
        return float(np.dot(_features(values), coefficients))

    return predict


def suggest_joint_optimize(
    experiments: list[Experiment],
    knobs: list[str],
    pending_suggestions: list[SuggestionRecord] = (),
    latency_budget_ms: Optional[float] = None,
    ttft_budget_ms: Optional[float] = None,
    min_throughput_tokens_per_sec: Optional[float] = None,
    runtime_budget_sec: Optional[float] = None,
    max_failure_rate: Optional[float] = None,
    max_candidates: int = DEFAULT_SEARCH_NEIGHBORHOOD_CAP,
) -> OptimizeRecommendation:
    """Suggest one untried combination, scored by a fitted response surface where possible."""
    budgets = Budgets(
        latency_ms=latency_budget_ms,
        ttft_ms=ttft_budget_ms,
        min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
        runtime_sec=runtime_budget_sec,
        failure_rate=max_failure_rate,
    )

    joint = recommend_joint(
        experiments,
        knobs=knobs,
        latency_budget_ms=latency_budget_ms,
        ttft_budget_ms=ttft_budget_ms,
        min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
        runtime_budget_sec=runtime_budget_sec,
        max_failure_rate=max_failure_rate,
    )
    if joint.best is None:
        return OptimizeRecommendation(
            knobs=list(knobs), best=None, suggested=None, predicted_efficiency=None, reason=joint.reason
        )

    completed = _completed_for_knobs(experiments, knobs, budgets)
    tried_values = {tuple(_knob_value(exp, knob) for knob in knobs) for exp in completed}
    combos = _combos_from_completed(completed, knobs)

    already_pending = {
        tuple(suggestion.suggested_values[k] for k in knobs)
        for suggestion in pending_suggestions
        if suggestion.resolved_experiment_id is None and set(suggestion.suggested_values) == set(knobs)
    }

    neighborhood = [
        candidate
        for candidate in _bounded_neighborhood(experiments, knobs, joint.best.values, tried_values)
        if tuple(candidate[k] for k in knobs) not in already_pending
    ]

    combo_desc = format_combo(joint.best.values)
    best_desc = (
        f"Best combo tried so far is {combo_desc} ({joint.best.throughput:.0f} tok/s @ {joint.best.latency:.0f}ms)."
    )

    if not neighborhood:
        return OptimizeRecommendation(
            knobs=list(knobs),
            best=joint.best,
            suggested=None,
            predicted_efficiency=None,
            reason=(
                f"{best_desc} No untried combination available in the local neighborhood "
                "(already tried, or already suggested and still pending)."
            ),
        )

    predict = _fit_efficiency_surface(combos, knobs)
    if predict is None:
        fallback = suggest_joint_step(
            experiments,
            knobs=knobs,
            latency_budget_ms=latency_budget_ms,
            ttft_budget_ms=ttft_budget_ms,
            min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
            runtime_budget_sec=runtime_budget_sec,
            max_failure_rate=max_failure_rate,
            max_candidates=max_candidates,
        )
        return OptimizeRecommendation(
            knobs=list(knobs),
            best=fallback.best,
            suggested=fallback.suggested,
            predicted_efficiency=None,
            reason=(
                f"Only {len(combos)} distinct combo(s) tried, not enough to fit a reliable response "
                f"surface (need at least {1 + 2 * len(knobs) + 1}). Falling back to trend-based search: "
                f"{fallback.reason}"
            ),
        )

    scored = sorted(neighborhood, key=predict, reverse=True)[:max_candidates]
    best_candidate = scored[0]
    predicted = predict(best_candidate)
    changed_knobs = [k for k in knobs if best_candidate[k] != joint.best.values[k]]
    changes_desc = ", ".join(f"{k} {joint.best.values[k]:g}->{best_candidate[k]:g}" for k in changed_knobs)

    return OptimizeRecommendation(
        knobs=list(knobs),
        best=joint.best,
        suggested=best_candidate,
        predicted_efficiency=predicted,
        reason=(
            f"{best_desc} Fitted a response surface to {len(combos)} tried combo(s); predicted "
            f"efficiency for the best untried candidate ({changes_desc}) is {predicted:.4f} tok/s per ms, "
            f"vs. {joint.best.efficiency:.4f} tok/s per ms observed at the current best."
        ),
    )


def resolve_pending_suggestions(
    experiments: list[Experiment],
    knobs: list[str],
    pending: list[SuggestionRecord],
) -> list[tuple[int, int, float]]:
    """Match unresolved suggestions against experiments since run with those exact knob values.

    Returns `(suggestion_id, experiment_id, actual_efficiency)` for each match; callers persist
    these via `ExperimentDB.resolve_suggestion`. This is what makes the response surface learn
    from a suggestion's real outcome on the next `--optimize` call: once resolved, the matching
    experiment is just another completed run and is included in the next fit.
    """
    resolutions: list[tuple[int, int, float]] = []
    for suggestion in pending:
        if suggestion.resolved_experiment_id is not None or suggestion.id is None:
            continue
        if set(suggestion.suggested_values) != set(knobs):
            continue
        match = next(
            (
                exp
                for exp in experiments
                if exp.status == "completed"
                and all(_knob_value(exp, k) == suggestion.suggested_values[k] for k in knobs)
                and _efficiency(exp) is not None
            ),
            None,
        )
        if match is not None and match.id is not None:
            resolutions.append((suggestion.id, match.id, _efficiency(match)))
    return resolutions
