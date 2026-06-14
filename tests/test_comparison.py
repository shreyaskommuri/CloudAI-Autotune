import pytest

from autotune.comparison import compare_best_and_latest
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
