import pytest

from autotune.database import Experiment, SuggestionRecord
from autotune.optimizer import (
    _fit_efficiency_surface,
    resolve_pending_suggestions,
    suggest_joint_optimize,
)
from autotune.recommender import ComboResult

KNOBS = ["serving.batch_size", "serving.num_requests"]

# A generous sweep across three distinct values per knob, well over the 6-combo minimum needed
# to fit a two-knob quadratic surface (5 params).
RICH_COMBOS = [
    (1, 1, 100, 120, 90),
    (2, 2, 100, 200, 110),
    (3, 4, 100, 330, 160),
    (4, 1, 200, 180, 140),
    (5, 2, 200, 300, 165),
    (6, 4, 200, 500, 170),
    (7, 8, 200, 520, 175),
    (8, 2, 400, 340, 200),
    (9, 4, 400, 560, 210),
]


def _combo_exp(id_, batch_size, num_requests, throughput, latency, status="completed"):
    return Experiment(
        id=id_,
        created_at="2026-01-01",
        scenario="vllm_baseline",
        backend="vllm",
        config_path=f"cfg_{id_}.toml",
        config={"serving": {"batch_size": batch_size, "num_requests": num_requests}},
        status=status,
        metrics={"throughput_tokens_per_sec": throughput, "latency_ms": latency},
    )


def _rich_experiments() -> list[Experiment]:
    return [_combo_exp(*row) for row in RICH_COMBOS]


def _suggestion(**overrides) -> SuggestionRecord:
    defaults = dict(
        id=1,
        created_at="2026-01-01",
        knobs=KNOBS,
        suggested_values={"serving.batch_size": 8, "serving.num_requests": 400},
        predicted_efficiency=2.8,
        resolved_experiment_id=None,
        actual_efficiency=None,
    )
    defaults.update(overrides)
    return SuggestionRecord(**defaults)


def test_fit_efficiency_surface_returns_none_with_too_few_combos():
    combos = [
        ComboResult(
            values={"serving.batch_size": 1, "serving.num_requests": 100},
            experiment_id=1,
            throughput=100,
            latency=90,
            efficiency=1.1,
        ),
        ComboResult(
            values={"serving.batch_size": 2, "serving.num_requests": 100},
            experiment_id=2,
            throughput=200,
            latency=110,
            efficiency=1.8,
        ),
    ]

    assert _fit_efficiency_surface(combos, KNOBS) is None


def test_fit_efficiency_surface_returns_none_when_rank_deficient():
    # Only two distinct values of num_requests are tried: its linear and squared terms are
    # collinear, so the design matrix can't be fit reliably even though there are enough rows.
    points = [(1, 100), (2, 100), (4, 100), (1, 200), (2, 200), (4, 200), (8, 200)]
    combos = [
        ComboResult(
            values={"serving.batch_size": bs, "serving.num_requests": nr},
            experiment_id=i,
            throughput=0.0,
            latency=1.0,
            efficiency=float(bs + nr),
        )
        for i, (bs, nr) in enumerate(points)
    ]

    assert _fit_efficiency_surface(combos, KNOBS) is None


def test_fit_efficiency_surface_recovers_a_known_quadratic():
    def true_efficiency(bs: float, nr: float) -> float:
        return 10 + 0.5 * bs - 0.01 * bs**2 + 0.3 * nr - 0.002 * nr**2

    points = [(bs, nr) for bs in (1, 2, 4, 8, 16) for nr in (100, 200, 400)]
    combos = [
        ComboResult(
            values={"serving.batch_size": bs, "serving.num_requests": nr},
            experiment_id=i,
            throughput=0.0,
            latency=1.0,
            efficiency=true_efficiency(bs, nr),
        )
        for i, (bs, nr) in enumerate(points)
    ]

    predict = _fit_efficiency_surface(combos, KNOBS)

    assert predict is not None
    held_out = {"serving.batch_size": 6, "serving.num_requests": 300}
    assert predict(held_out) == pytest.approx(true_efficiency(6, 300), rel=1e-6)


def test_suggest_joint_optimize_with_no_experiments():
    rec = suggest_joint_optimize([], knobs=KNOBS)

    assert rec.best is None
    assert rec.suggested is None
    assert rec.predicted_efficiency is None


def test_suggest_joint_optimize_falls_back_to_heuristic_with_insufficient_data():
    experiments = [
        _combo_exp(1, 1, 100, throughput=120, latency=90),
        _combo_exp(2, 4, 100, throughput=330, latency=160),
        _combo_exp(3, 4, 200, throughput=500, latency=170),
    ]

    rec = suggest_joint_optimize(experiments, knobs=KNOBS)

    assert rec.predicted_efficiency is None
    assert rec.suggested is not None
    assert "Falling back to trend-based search" in rec.reason


def test_suggest_joint_optimize_uses_fitted_surface_with_enough_data():
    rec = suggest_joint_optimize(_rich_experiments(), knobs=KNOBS)

    assert rec.suggested is not None
    assert rec.predicted_efficiency is not None
    assert "Fitted a response surface" in rec.reason


def test_suggest_joint_optimize_never_repeats_an_already_tried_combo():
    tried = {(bs, nr) for _, bs, nr, _, _ in RICH_COMBOS}

    rec = suggest_joint_optimize(_rich_experiments(), knobs=KNOBS)

    assert rec.suggested is not None
    assert (rec.suggested["serving.batch_size"], rec.suggested["serving.num_requests"]) not in tried


def test_suggest_joint_optimize_skips_a_pending_suggestion():
    experiments = _rich_experiments()
    first = suggest_joint_optimize(experiments, knobs=KNOBS)
    assert first.suggested is not None

    pending = [_suggestion(suggested_values=first.suggested, predicted_efficiency=first.predicted_efficiency)]
    second = suggest_joint_optimize(experiments, knobs=KNOBS, pending_suggestions=pending)

    assert second.suggested is not None
    assert second.suggested != first.suggested


def test_suggest_joint_optimize_ignores_a_resolved_suggestion():
    experiments = _rich_experiments()
    first = suggest_joint_optimize(experiments, knobs=KNOBS)
    assert first.suggested is not None

    # A resolved suggestion no longer blocks re-suggesting the same combo (it's not "pending"
    # anymore) - resolution happens because the combo was actually run, which would already put
    # it in `experiments`/tried_values in practice.
    pending = [
        _suggestion(
            suggested_values=first.suggested,
            resolved_experiment_id=42,
            actual_efficiency=1.5,
        )
    ]
    second = suggest_joint_optimize(experiments, knobs=KNOBS, pending_suggestions=pending)

    assert second.suggested == first.suggested


def test_suggest_joint_optimize_respects_budgets():
    experiments = [
        _combo_exp(1, 4, 200, throughput=300, latency=150),
        _combo_exp(2, 8, 400, throughput=430, latency=260),  # breaches budget, excluded
    ]

    rec = suggest_joint_optimize(experiments, knobs=KNOBS, latency_budget_ms=200)

    assert rec.best is not None
    assert rec.best.values == {"serving.batch_size": 4, "serving.num_requests": 200}


def test_resolve_pending_suggestions_matches_completed_experiment():
    experiments = [_combo_exp(1, 8, 400, throughput=600, latency=230)]
    pending = [_suggestion()]

    resolutions = resolve_pending_suggestions(experiments, KNOBS, pending)

    assert resolutions == [(1, 1, pytest.approx(600 / 230))]


def test_resolve_pending_suggestions_no_match_yet():
    experiments = [_combo_exp(1, 1, 100, throughput=120, latency=90)]
    pending = [_suggestion()]

    assert resolve_pending_suggestions(experiments, KNOBS, pending) == []


def test_resolve_pending_suggestions_ignores_already_resolved():
    experiments = [_combo_exp(1, 8, 400, throughput=600, latency=230)]
    pending = [_suggestion(resolved_experiment_id=99, actual_efficiency=2.6)]

    assert resolve_pending_suggestions(experiments, KNOBS, pending) == []


def test_resolve_pending_suggestions_ignores_mismatched_knob_set():
    experiments = [_combo_exp(1, 8, 400, throughput=600, latency=230)]
    pending = [_suggestion(knobs=["serving.batch_size"], suggested_values={"serving.batch_size": 8})]

    assert resolve_pending_suggestions(experiments, KNOBS, pending) == []
