"""Suggest the next benchmark config to try based on prior experiment history.

Three heuristics, in increasing order of scope:

- `recommend_next`: track a single tunable knob (default `serving.batch_size`)
  across completed runs. If throughput is still climbing faster than latency is
  degrading (i.e. the tokens/sec-per-ms-of-latency ratio is improving or stable),
  recommend doubling the knob; otherwise recommend backing off to the best
  observed tradeoff point.
- `recommend_joint`: the same idea across two or more knobs looked at *together*
  instead of independently. Reports the best combination tried so far plus the
  Pareto frontier, it does not suggest untried combinations.
- `suggest_untried_combo`: extends `recommend_joint`'s best combo by one knob at
  a time, reusing `recommend_next`'s trend logic per knob rather than a new
  search algorithm. Stateless, one suggestion per call.

Budget policy reuses `autotune.budgets`: any run that fails a configured
latency, TTFT, throughput, runtime, or failure-rate budget is excluded from
"best observed tradeoff" candidates, the same policy `autotune check` applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from autotune.budgets import Budgets, evaluate_experiment
from autotune.database import Experiment

DEFAULT_KNOB = "serving.batch_size"


@dataclass
class Recommendation:
    knob: str
    current_value: Optional[float]
    suggested_value: Optional[float]
    reason: str


@dataclass
class ComboResult:
    values: dict[str, float]
    experiment_id: Optional[int]
    throughput: float
    latency: float
    efficiency: float


@dataclass
class JointRecommendation:
    knobs: list[str]
    best: Optional[ComboResult]
    frontier: list[ComboResult]
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


def discover_knobs(experiments: list[Experiment]) -> list[str]:
    """Return every dotted config key with a numeric value across completed runs.

    Lets callers report a recommendation for every knob that's actually been
    tuned without already knowing its dotted name up front.
    """
    knobs: set[str] = set()
    for exp in experiments:
        if exp.status != "completed":
            continue
        for key, value in _flatten_config(exp.config).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                knobs.add(key)
    return sorted(knobs)


def _flatten_config(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _efficiency(exp: Experiment) -> Optional[float]:
    """tokens/sec per ms of latency — higher is a better throughput/latency tradeoff."""
    throughput = _metric_value(exp, "throughput_tokens_per_sec")
    latency = _metric_value(exp, "latency_ms")
    if throughput is None or latency in (None, 0):
        return None
    return throughput / latency


def _metric_value(exp: Experiment, key: str) -> Optional[float]:
    try:
        return float(exp.metrics[key])
    except (KeyError, TypeError, ValueError):
        return None


def recommend_next(
    experiments: list[Experiment],
    knob: str = DEFAULT_KNOB,
    latency_budget_ms: Optional[float] = None,
    ttft_budget_ms: Optional[float] = None,
    min_throughput_tokens_per_sec: Optional[float] = None,
    runtime_budget_sec: Optional[float] = None,
    max_failure_rate: Optional[float] = None,
) -> Recommendation:
    """Recommend the next value to try for `knob` based on completed runs.

    Any budget argument, if given, caps how far that metric is allowed to
    regress — a run that fails a budget is treated as over-budget regardless
    of throughput gains, the same policy `autotune check` applies.
    """
    budgets = Budgets(
        latency_ms=latency_budget_ms,
        ttft_ms=ttft_budget_ms,
        min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
        runtime_sec=runtime_budget_sec,
        failure_rate=max_failure_rate,
    )
    completed = [
        e
        for e in experiments
        if e.status == "completed"
        and _knob_value(e, knob) is not None
        and _metric_value(e, "throughput_tokens_per_sec") is not None
        and _metric_value(e, "latency_ms") is not None
    ]
    if not completed:
        return Recommendation(
            knob=knob,
            current_value=None,
            suggested_value=None,
            reason="No completed experiments with usable throughput and latency metrics yet — run a baseline first.",
        )

    completed.sort(key=lambda e: _knob_value(e, knob))
    tried_values = {_knob_value(e, knob) for e in completed}
    latest = completed[-1]
    latest_value = _knob_value(latest, knob)
    latest_throughput = _metric_value(latest, "throughput_tokens_per_sec")
    latest_latency = _metric_value(latest, "latency_ms")

    latest_breach = evaluate_experiment(latest, budgets) if budgets.has_checks() else None
    if latest_breach is not None and latest_breach.status == "fail":
        # Latest run failed a budget — recommend the best prior in-budget tradeoff.
        breach_text = "; ".join(latest_breach.reasons)
        candidates = [e for e in completed if evaluate_experiment(e, budgets).status != "fail"]
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
                    f"{knob}={latest_value} breached budget ({breach_text}). "
                    f"Best observed tradeoff within budget was {knob}={best_value} "
                    f"({_metric_value(best, 'throughput_tokens_per_sec'):.0f} tok/s @ "
                    f"{_metric_value(best, 'latency_ms'):.0f}ms). Try untested {knob}="
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
                f"All completed runs breach the configured budget ({breach_text}). "
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
    prev_throughput = _metric_value(prev, "throughput_tokens_per_sec")
    prev_latency = _metric_value(prev, "latency_ms")

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
    if throughput_gain is None or latency_gain is None:
        return Recommendation(
            knob=knob,
            current_value=latest_value,
            suggested_value=suggested,
            reason=(
                "Recent percentage change is undefined because the previous "
                "throughput or latency baseline was zero. "
                f"Best observed tradeoff so far is {knob}={best_value} "
                f"({_metric_value(best, 'throughput_tokens_per_sec'):.0f} tok/s @ "
                f"{_metric_value(best, 'latency_ms'):.0f}ms) — try untested {knob}="
                f"{suggested} nearby."
            ),
        )
    return Recommendation(
        knob=knob,
        current_value=latest_value,
        suggested_value=suggested,
        reason=(
            f"Latency is now growing faster than throughput "
            f"({latency_gain:+.0%} vs {throughput_gain:+.0%}). "
            f"Best tradeoff so far is {knob}={best_value} "
            f"({_metric_value(best, 'throughput_tokens_per_sec'):.0f} tok/s @ "
            f"{_metric_value(best, 'latency_ms'):.0f}ms) — try untested {knob}="
            f"{suggested} nearby."
        ),
    )


def recommend_joint(
    experiments: list[Experiment],
    knobs: list[str],
    latency_budget_ms: Optional[float] = None,
    ttft_budget_ms: Optional[float] = None,
    min_throughput_tokens_per_sec: Optional[float] = None,
    runtime_budget_sec: Optional[float] = None,
    max_failure_rate: Optional[float] = None,
) -> JointRecommendation:
    """Report the best combination of `knobs` tried so far, plus the Pareto frontier.

    Unlike `recommend_next`, this looks at knobs jointly rather than one at a time. It only
    reports on combinations already run, it does not suggest untried combinations, that needs
    a real search strategy and is deliberately out of scope here.
    """
    if len(knobs) < 2:
        raise ValueError("recommend_joint needs at least two knobs; use recommend_next for a single knob.")

    budgets = Budgets(
        latency_ms=latency_budget_ms,
        ttft_ms=ttft_budget_ms,
        min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
        runtime_sec=runtime_budget_sec,
        failure_rate=max_failure_rate,
    )

    completed = _completed_for_knobs(experiments, knobs, budgets)

    if not completed:
        return JointRecommendation(
            knobs=list(knobs),
            best=None,
            frontier=[],
            reason=(
                "No completed, in-budget experiments have values for all of "
                f"{', '.join(knobs)} plus usable throughput and latency metrics yet."
            ),
        )

    combos = _combos_from_completed(completed, knobs)
    best = max(combos, key=lambda c: c.efficiency)
    frontier = _pareto_frontier(combos)
    combo_desc = format_combo(best.values)

    return JointRecommendation(
        knobs=list(knobs),
        best=best,
        frontier=frontier,
        reason=(
            f"Best combo tried so far is {combo_desc} "
            f"({best.throughput:.0f} tok/s @ {best.latency:.0f}ms). "
            f"{len(combos)} distinct combo(s) tried, {len(frontier)} on the Pareto frontier "
            "(not beaten on both throughput and latency by another combo)."
        ),
    )


@dataclass
class ExploreRecommendation:
    knobs: list[str]
    best: Optional[ComboResult]
    suggested: Optional[dict[str, float]]
    reason: str


def suggest_untried_combo(
    experiments: list[Experiment],
    knobs: list[str],
    latency_budget_ms: Optional[float] = None,
    ttft_budget_ms: Optional[float] = None,
    min_throughput_tokens_per_sec: Optional[float] = None,
    runtime_budget_sec: Optional[float] = None,
    max_failure_rate: Optional[float] = None,
) -> ExploreRecommendation:
    """Suggest one untried combination of `knobs`, extending the best combo tried so far.

    Picks one knob to perturb, holding the rest at the best combo's values, reusing the same
    doubling/halving trend logic `recommend_next` already uses for that one knob. Prefers a
    knob whose independent trend still looks like it's scaling well; falls back to the first
    knob with any suggestion otherwise. Stateless by design, call again after ingesting the
    result to get the next suggestion, there is no separate search-budget concept to configure.
    """
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
        return ExploreRecommendation(knobs=list(knobs), best=None, suggested=None, reason=joint.reason)

    completed = _completed_for_knobs(experiments, knobs, budgets)
    tried_values = {tuple(_knob_value(exp, knob) for knob in knobs) for exp in completed}

    # Isolate each knob's marginal trend by only looking at experiments that match the best
    # combo on every *other* knob — otherwise recommend_next's "latest vs. previous" trend
    # can compare two runs that actually differ on a different knob, not this one.
    per_knob: dict[str, Recommendation] = {}
    for knob in knobs:
        other_knobs = [k for k in knobs if k != knob]
        isolated = [
            e for e in experiments if all(_knob_value(e, other) == joint.best.values[other] for other in other_knobs)
        ]
        per_knob[knob] = recommend_next(
            isolated,
            knob=knob,
            latency_budget_ms=latency_budget_ms,
            ttft_budget_ms=ttft_budget_ms,
            min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
            runtime_budget_sec=runtime_budget_sec,
            max_failure_rate=max_failure_rate,
        )

    # Prefer a knob whose independent trend still looks like it's scaling well ("outpaced" is
    # recommend_next's marker for that case); otherwise fall back to any knob with a suggestion.
    preferred = [k for k in knobs if "outpaced" in per_knob[k].reason]
    candidate_order = list(dict.fromkeys(preferred + knobs))
    combo_desc = format_combo(joint.best.values)
    best_desc = (
        f"Best combo tried so far is {combo_desc} ({joint.best.throughput:.0f} tok/s @ {joint.best.latency:.0f}ms)."
    )

    for knob in candidate_order:
        suggested_value = per_knob[knob].suggested_value
        if suggested_value is None:
            continue
        candidate = dict(joint.best.values)
        candidate[knob] = suggested_value
        if tuple(candidate[k] for k in knobs) in tried_values:
            continue
        return ExploreRecommendation(
            knobs=list(knobs),
            best=joint.best,
            suggested=candidate,
            reason=(
                f"{best_desc} Exploring by adjusting {knob} to {suggested_value:g} "
                f"({per_knob[knob].reason}), holding the other knob(s) at the best combo's values."
            ),
        )

    return ExploreRecommendation(
        knobs=list(knobs),
        best=joint.best,
        suggested=None,
        reason=f"{best_desc} No untried direction could be suggested for any knob from the current data.",
    )


def _completed_for_knobs(experiments: list[Experiment], knobs: list[str], budgets: Budgets) -> list[Experiment]:
    return [
        e
        for e in experiments
        if e.status == "completed"
        and all(_knob_value(e, knob) is not None for knob in knobs)
        and _metric_value(e, "throughput_tokens_per_sec") is not None
        and _metric_value(e, "latency_ms") is not None
        and (not budgets.has_checks() or evaluate_experiment(e, budgets).status != "fail")
    ]


def _combos_from_completed(completed: list[Experiment], knobs: list[str]) -> list[ComboResult]:
    # One combo per unique tuple of knob values; keep the latest experiment (highest id)
    # when a combo was run more than once.
    combos_by_values: dict[tuple[float, ...], Experiment] = {}
    for exp in completed:
        key = tuple(_knob_value(exp, knob) for knob in knobs)
        current = combos_by_values.get(key)
        if current is None or (exp.id or 0) > (current.id or 0):
            combos_by_values[key] = exp

    return [
        ComboResult(
            values=dict(zip(knobs, key, strict=True)),
            experiment_id=exp.id,
            throughput=_metric_value(exp, "throughput_tokens_per_sec") or 0.0,
            latency=_metric_value(exp, "latency_ms") or 0.0,
            efficiency=_efficiency(exp) or 0.0,
        )
        for key, exp in combos_by_values.items()
    ]


def format_combo(values: dict[str, float]) -> str:
    """Render a knob-value combo as `k1=v1, k2=v2`, used in both reasons and CLI output."""
    return ", ".join(f"{k}={v:g}" for k, v in values.items())


def _pareto_frontier(combos: list[ComboResult]) -> list[ComboResult]:
    """Combos not strictly dominated by another: higher throughput, lower-or-equal latency
    (or equal throughput with strictly lower latency), beats a combo out of the frontier."""

    def dominates(a: ComboResult, b: ComboResult) -> bool:
        at_least_as_good = a.throughput >= b.throughput and a.latency <= b.latency
        strictly_better = a.throughput > b.throughput or a.latency < b.latency
        return at_least_as_good and strictly_better

    return [combo for combo in combos if not any(dominates(other, combo) for other in combos if other is not combo)]


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
