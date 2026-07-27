import pytest

from autotune.comparison import compare_best_and_latest, compare_latest_to_previous
from autotune.database import Experiment


def _exp(experiment_id, metrics, status="completed"):
    return Experiment(
        id=experiment_id,
        created_at=f"2026-01-0{experiment_id}",
        scenario="vllm_baseline",
        backend="vllm",
        config_path=f"batch{experiment_id}.toml",
        config={},
        status=status,
        metrics=metrics,
    )


def test_compare_best_and_latest_reports_deltas_from_best_run():
    summary = compare_best_and_latest(
        [
            _exp(1, {"throughput_tokens_per_sec": 300, "latency_ms": 140}),
            _exp(2, {"throughput_tokens_per_sec": 400, "latency_ms": 180}),
            _exp(3, {"throughput_tokens_per_sec": 360, "latency_ms": 170}),
        ]
    )

    assert summary.best.id == 2
    assert summary.latest.id == 3
    assert summary.throughput_delta_pct == pytest.approx(-10.0)
    assert summary.latency_delta_ms == pytest.approx(-10.0)


def test_compare_best_and_latest_respects_latency_budget():
    summary = compare_best_and_latest(
        [
            _exp(1, {"throughput_tokens_per_sec": 300, "latency_ms": 140}),
            _exp(2, {"throughput_tokens_per_sec": 500, "latency_ms": 260}),
            _exp(3, {"throughput_tokens_per_sec": 360, "latency_ms": 170}),
        ],
        latency_budget_ms=200,
    )

    assert summary.best.id == 3
    assert summary.latest.id == 3
    assert summary.throughput_delta_pct == pytest.approx(0.0)
    assert summary.latency_delta_ms == pytest.approx(0.0)


def test_compare_best_and_latest_ignores_incomplete_and_metricless_runs():
    summary = compare_best_and_latest(
        [
            _exp(1, {"throughput_tokens_per_sec": 300, "latency_ms": 140}),
            _exp(2, {}, status="failed"),
            _exp(3, {}),
        ]
    )

    assert summary.best.id == 1
    assert summary.latest.id == 3
    assert summary.throughput_delta_pct is None
    assert summary.latency_delta_ms is None


def test_compare_latest_to_previous_flags_a_regression():
    check = compare_latest_to_previous(
        [
            _exp(1, {"throughput_tokens_per_sec": 400, "latency_ms": 150}),
            _exp(2, {"throughput_tokens_per_sec": 350, "latency_ms": 180}),
        ]
    )

    assert check.latest.id == 2
    assert check.previous.id == 1
    assert check.throughput_delta_pct == pytest.approx(-12.5)
    assert check.latency_delta_ms == pytest.approx(30.0)
    assert check.regressed is True


def test_compare_latest_to_previous_does_not_flag_an_improvement():
    check = compare_latest_to_previous(
        [
            _exp(1, {"throughput_tokens_per_sec": 300, "latency_ms": 180}),
            _exp(2, {"throughput_tokens_per_sec": 400, "latency_ms": 150}),
        ]
    )

    assert check.regressed is False


def test_compare_latest_to_previous_is_not_fooled_by_the_global_best():
    """A run can regress vs. its immediate predecessor without being the worst run ever."""
    check = compare_latest_to_previous(
        [
            _exp(1, {"throughput_tokens_per_sec": 200, "latency_ms": 200}),
            _exp(2, {"throughput_tokens_per_sec": 400, "latency_ms": 150}),
            _exp(3, {"throughput_tokens_per_sec": 350, "latency_ms": 160}),
        ]
    )

    assert check.latest.id == 3
    assert check.previous.id == 2
    assert check.regressed is True


def test_compare_latest_to_previous_with_fewer_than_two_completed_runs():
    single = compare_latest_to_previous([_exp(1, {"throughput_tokens_per_sec": 300, "latency_ms": 140})])
    assert single.latest.id == 1
    assert single.previous is None
    assert single.regressed is False

    empty = compare_latest_to_previous([])
    assert empty.latest is None
    assert empty.previous is None
    assert empty.regressed is False


def test_compare_latest_to_previous_ignores_incomplete_runs():
    check = compare_latest_to_previous(
        [
            _exp(1, {"throughput_tokens_per_sec": 300, "latency_ms": 140}),
            _exp(2, {}, status="failed"),
            _exp(3, {"throughput_tokens_per_sec": 250, "latency_ms": 200}),
        ]
    )

    assert check.latest.id == 3
    assert check.previous.id == 1
    assert check.regressed is True
