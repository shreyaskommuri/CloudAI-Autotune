from autotune.database import Experiment
from autotune.recommender import recommend_next


def _exp(id_, batch_size, throughput, latency, status="completed"):
    return Experiment(
        id=id_,
        created_at="2026-01-01",
        scenario="vllm_baseline",
        backend="vllm",
        config_path=f"cfg_{id_}.toml",
        config={"serving": {"batch_size": batch_size}},
        status=status,
        metrics={"throughput_tokens_per_sec": throughput, "latency_ms": latency},
    )


def test_recommend_with_no_experiments():
    rec = recommend_next([])

    assert rec.suggested_value is None
    assert "No completed experiments" in rec.reason


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

    # throughput grew ~+6%, latency grew +150% -> recommend best tradeoff (batch_size=4)
    assert rec.suggested_value == 4
    assert "growing faster" in rec.reason


def test_recommend_respects_latency_budget():
    experiments = [
        _exp(1, batch_size=4, throughput=330, latency=160),
        _exp(2, batch_size=8, throughput=400, latency=500),
    ]

    rec = recommend_next(experiments, latency_budget_ms=200)

    assert rec.suggested_value == 4
    assert "budget" in rec.reason
