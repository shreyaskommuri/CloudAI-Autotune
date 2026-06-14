from autotune.budgets import Budgets, evaluate_experiment
from autotune.database import Experiment


def _exp(metrics, status="completed"):
    return Experiment(
        id=1,
        created_at="2026-01-01",
        scenario="vllm_baseline",
        backend="vllm",
        config_path="cfg.toml",
        config={},
        status=status,
        metrics=metrics,
    )


def test_budget_check_passes_when_all_provided_limits_are_satisfied():
    check = evaluate_experiment(
        _exp(
            {
                "latency_ms": 180,
                "ttft_ms": 40,
                "throughput_tokens_per_sec": 350,
                "runtime_sec": 30,
                "failure_rate": 0.01,
            }
        ),
        Budgets(
            latency_ms=200,
            ttft_ms=50,
            min_throughput_tokens_per_sec=300,
            runtime_sec=60,
            failure_rate=0.05,
        ),
    )

    assert check.status == "pass"
    assert check.passed


def test_budget_check_fails_when_any_limit_is_violated():
    check = evaluate_experiment(
        _exp({"latency_ms": 260, "ttft_ms": 75, "throughput_tokens_per_sec": 250}),
        Budgets(latency_ms=200, ttft_ms=50, min_throughput_tokens_per_sec=300),
    )

    assert check.status == "fail"
    assert "latency 260ms exceeds budget 200ms" in check.reasons
    assert "ttft 75ms exceeds budget 50ms" in check.reasons
    assert "throughput 250tok/s is below budget 300tok/s" in check.reasons


def test_budget_check_is_unknown_when_required_metric_is_missing():
    check = evaluate_experiment(
        _exp({"latency_ms": 180}),
        Budgets(latency_ms=200, min_throughput_tokens_per_sec=300),
    )

    assert check.status == "unknown"
    assert check.reasons == ("throughput metric missing",)


def test_budget_check_reports_missing_metrics_alongside_failures():
    check = evaluate_experiment(
        _exp({"latency_ms": 260}),
        Budgets(latency_ms=200, ttft_ms=50),
    )

    assert check.status == "fail"
    assert check.reasons == (
        "latency 260ms exceeds budget 200ms",
        "ttft metric missing",
    )


def test_budget_check_is_unknown_for_non_completed_experiment():
    check = evaluate_experiment(
        _exp({"latency_ms": 180}, status="failed"),
        Budgets(latency_ms=200),
    )

    assert check.status == "unknown"
    assert check.reasons == ("experiment status is failed",)
