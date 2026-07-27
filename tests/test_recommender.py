from autotune.database import Experiment
from autotune.recommender import discover_knobs, recommend_next


def _exp(id_, batch_size, throughput, latency, status="completed", **extra_metrics):
    metrics = {"throughput_tokens_per_sec": throughput, "latency_ms": latency}
    metrics.update(extra_metrics)
    return Experiment(
        id=id_,
        created_at="2026-01-01",
        scenario="vllm_baseline",
        backend="vllm",
        config_path=f"cfg_{id_}.toml",
        config={"serving": {"batch_size": batch_size}},
        status=status,
        metrics=metrics,
    )


def test_recommend_with_no_experiments():
    rec = recommend_next([])

    assert rec.suggested_value is None
    assert "No completed experiments" in rec.reason


def test_recommend_ignores_completed_runs_without_usable_metrics():
    experiments = [
        _exp(1, batch_size=1, throughput=None, latency=None),
        _exp(2, batch_size=2, throughput=230, latency=120),
    ]

    rec = recommend_next(experiments)

    assert rec.current_value == 2
    assert rec.suggested_value == 4


def test_recommend_handles_only_unusable_metrics_without_crashing():
    rec = recommend_next([_exp(1, batch_size=1, throughput=None, latency=None)])

    assert rec.suggested_value is None
    assert "usable throughput and latency metrics" in rec.reason


def test_recommend_doubles_after_single_run():
    rec = recommend_next([_exp(1, batch_size=1, throughput=120, latency=90)])

    assert rec.current_value == 1
    assert rec.suggested_value == 2


def test_recommend_continues_doubling_when_throughput_outpaces_latency():
    experiments = [
        _exp(1, batch_size=1, throughput=120, latency=90),
        _exp(2, batch_size=4, throughput=330, latency=160),
    ]

    rec = recommend_next(experiments)

    # throughput grew +175%, latency grew +78% -> still scaling well, suggest doubling
    assert rec.current_value == 4
    assert rec.suggested_value == 8
    assert "outpaced" in rec.reason


def test_recommend_backs_off_when_latency_grows_faster():
    experiments = [
        _exp(1, batch_size=4, throughput=330, latency=160),
        _exp(2, batch_size=8, throughput=350, latency=400),
    ]

    rec = recommend_next(experiments)

    # throughput grew ~+6%, latency grew +150% -> try an untested value
    # near the best observed tradeoff instead of repeating batch_size=4.
    assert rec.suggested_value == 6
    assert "growing faster" in rec.reason
    assert "untested" in rec.reason


def test_recommend_handles_zero_baseline_percentage_change():
    experiments = [
        _exp(1, batch_size=1, throughput=0, latency=0),
        _exp(2, batch_size=2, throughput=120, latency=90),
    ]

    rec = recommend_next(experiments)

    assert rec.current_value == 2
    assert rec.suggested_value == 3
    assert "undefined" in rec.reason


def test_recommend_accepts_numeric_metric_strings():
    experiments = [
        _exp(1, batch_size=1, throughput="120", latency="90"),
        _exp(2, batch_size=2, throughput="260", latency="120"),
    ]

    rec = recommend_next(experiments)

    assert rec.current_value == 2
    assert rec.suggested_value == 4


def test_recommend_respects_latency_budget():
    experiments = [
        _exp(1, batch_size=4, throughput=330, latency=160),
        _exp(2, batch_size=8, throughput=400, latency=500),
    ]

    rec = recommend_next(experiments, latency_budget_ms=200)

    assert rec.suggested_value == 6
    assert "budget" in rec.reason
    assert "untested" in rec.reason


def test_recommend_respects_ttft_budget_even_when_latency_is_fine():
    experiments = [
        _exp(1, batch_size=4, throughput=330, latency=160, ttft_ms=30),
        _exp(2, batch_size=8, throughput=400, latency=180, ttft_ms=90),
    ]

    rec = recommend_next(experiments, ttft_budget_ms=50)

    # latency stays under any reasonable budget, but ttft blew past 50ms —
    # the recommender must still treat run #2 as a regression.
    assert rec.suggested_value == 6
    assert "ttft" in rec.reason
    assert "budget" in rec.reason


def test_recommend_respects_failure_rate_budget():
    experiments = [
        _exp(1, batch_size=4, throughput=330, latency=160, failure_rate=0.0),
        _exp(2, batch_size=8, throughput=400, latency=180, failure_rate=0.2),
    ]

    rec = recommend_next(experiments, max_failure_rate=0.05)

    assert rec.suggested_value == 6
    assert "failure_rate" in rec.reason


def test_recommend_respects_min_throughput_budget():
    experiments = [
        _exp(1, batch_size=4, throughput=330, latency=160),
        _exp(2, batch_size=8, throughput=100, latency=90),
    ]

    rec = recommend_next(experiments, min_throughput_tokens_per_sec=200)

    assert rec.suggested_value == 6
    assert "throughput" in rec.reason


def test_recommend_ignores_missing_ttft_metric_when_no_budget_breach():
    # Older runs without a tracked ttft_ms metric shouldn't be excluded from
    # consideration just because a ttft budget is configured — only an actual
    # ttft breach should count as a regression.
    experiments = [
        _exp(1, batch_size=1, throughput=120, latency=90),
        _exp(2, batch_size=2, throughput=230, latency=120),
    ]

    rec = recommend_next(experiments, ttft_budget_ms=50)

    assert rec.current_value == 2
    assert rec.suggested_value == 4


def test_recommend_skips_already_tried_growth_candidate():
    experiments = [
        _exp(1, batch_size=1, throughput=120, latency=90),
        _exp(2, batch_size=2, throughput=230, latency=120),
        _exp(3, batch_size=4, throughput=430, latency=160),
    ]

    rec = recommend_next(experiments)

    assert rec.suggested_value == 8
    assert rec.suggested_value not in {1, 2, 4}


def _make_experiment(id_, config, status="completed"):
    return Experiment(
        id=id_,
        created_at="2026-01-01",
        scenario="vllm_baseline",
        backend="vllm",
        config_path=f"cfg_{id_}.toml",
        config=config,
        status=status,
        metrics={"throughput_tokens_per_sec": 300, "latency_ms": 150},
    )


def test_discover_knobs_finds_every_numeric_dotted_key():
    experiments = [
        _make_experiment(1, {"serving": {"batch_size": 4, "tp_size": 2}}),
        _make_experiment(2, {"serving": {"batch_size": 8}, "model": {"max_seq_len": 4096}}),
    ]

    assert discover_knobs(experiments) == ["model.max_seq_len", "serving.batch_size", "serving.tp_size"]


def test_discover_knobs_ignores_non_numeric_and_bool_values():
    experiments = [
        _make_experiment(1, {"serving": {"batch_size": 4, "backend": "vllm", "warmup": True}}),
    ]

    assert discover_knobs(experiments) == ["serving.batch_size"]


def test_discover_knobs_ignores_incomplete_runs():
    experiments = [
        _make_experiment(1, {"serving": {"batch_size": 4}}, status="completed"),
        _make_experiment(2, {"serving": {"tp_size": 2}}, status="failed"),
    ]

    assert discover_knobs(experiments) == ["serving.batch_size"]


def test_discover_knobs_with_no_experiments():
    assert discover_knobs([]) == []
