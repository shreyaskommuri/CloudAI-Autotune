import pytest

from autotune.database import Experiment
from autotune.recommender import (
    discover_knobs,
    recommend_joint,
    recommend_next,
    suggest_joint_step,
    suggest_untried_combo,
)


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


def test_recommend_joint_requires_at_least_two_knobs():
    with pytest.raises(ValueError, match="at least two knobs"):
        recommend_joint([], knobs=["serving.batch_size"])


def test_recommend_joint_with_no_experiments():
    rec = recommend_joint([], knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is None
    assert rec.frontier == []
    assert "No completed" in rec.reason


def test_recommend_joint_picks_best_combo_by_efficiency():
    experiments = [
        _combo_exp(1, 1, 100, throughput=120, latency=90),
        _combo_exp(2, 4, 200, throughput=330, latency=160),
        _combo_exp(3, 8, 400, throughput=430, latency=260),
    ]

    rec = recommend_joint(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is not None
    assert rec.best.values == {"serving.batch_size": 4, "serving.num_requests": 200}


def test_recommend_joint_deduplicates_repeated_combos_keeping_latest():
    experiments = [
        _combo_exp(1, 4, 200, throughput=300, latency=150),
        _combo_exp(2, 4, 200, throughput=330, latency=160),
    ]

    rec = recommend_joint(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is not None
    assert rec.best.experiment_id == 2
    assert rec.best.throughput == 330


def test_recommend_joint_frontier_excludes_dominated_combos():
    experiments = [
        _combo_exp(1, 1, 100, throughput=100, latency=200),  # beaten on both axes by combo 2
        _combo_exp(2, 4, 200, throughput=300, latency=150),  # lower latency, real tradeoff vs combo 3
        _combo_exp(3, 8, 400, throughput=400, latency=180),  # higher throughput, real tradeoff vs combo 2
    ]

    rec = recommend_joint(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    frontier_values = [combo.values for combo in rec.frontier]
    assert {"serving.batch_size": 1, "serving.num_requests": 100} not in frontier_values
    assert {"serving.batch_size": 4, "serving.num_requests": 200} in frontier_values
    assert {"serving.batch_size": 8, "serving.num_requests": 400} in frontier_values


def test_recommend_joint_frontier_keeps_real_tradeoffs():
    experiments = [
        _combo_exp(1, 4, 200, throughput=300, latency=120),  # lower latency, lower throughput
        _combo_exp(2, 8, 400, throughput=430, latency=260),  # higher throughput, higher latency
    ]

    rec = recommend_joint(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    assert len(rec.frontier) == 2


def test_recommend_joint_respects_latency_budget():
    experiments = [
        _combo_exp(1, 4, 200, throughput=300, latency=150),
        _combo_exp(2, 8, 400, throughput=430, latency=260),  # breaches budget
    ]

    rec = recommend_joint(
        experiments,
        knobs=["serving.batch_size", "serving.num_requests"],
        latency_budget_ms=200,
    )

    assert rec.best is not None
    assert rec.best.values == {"serving.batch_size": 4, "serving.num_requests": 200}


def test_recommend_joint_ignores_experiments_missing_a_knob():
    experiments = [
        _combo_exp(1, 4, 200, throughput=300, latency=150),
        Experiment(
            id=2,
            created_at="2026-01-01",
            scenario="vllm_baseline",
            backend="vllm",
            config_path="cfg_2.toml",
            config={"serving": {"batch_size": 8}},
            status="completed",
            metrics={"throughput_tokens_per_sec": 400, "latency_ms": 200},
        ),
    ]

    rec = recommend_joint(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is not None
    assert rec.best.experiment_id == 1


def test_suggest_untried_combo_with_no_experiments():
    rec = suggest_untried_combo([], knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is None
    assert rec.suggested is None
    assert "No completed" in rec.reason


def test_suggest_untried_combo_extends_best_combo_on_the_scaling_knob():
    experiments = [
        # batch_size alone: 1 -> 4 throughput/latency both grow, throughput outpaces latency
        _combo_exp(1, 1, 200, throughput=120, latency=90),
        _combo_exp(2, 4, 200, throughput=330, latency=160),
        # num_requests alone (batch_size held at 4): 200 -> 400 latency grows faster
        _combo_exp(3, 4, 400, throughput=350, latency=400),
    ]

    rec = suggest_untried_combo(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is not None
    assert rec.best.values == {"serving.batch_size": 4, "serving.num_requests": 200}
    assert rec.suggested is not None
    # batch_size's independent trend is still scaling well (outpaced); num_requests is not.
    assert rec.suggested["serving.batch_size"] == 8
    assert rec.suggested["serving.num_requests"] == 200
    assert "serving.batch_size" in rec.reason
    assert "outpaced" in rec.reason


def test_suggest_untried_combo_never_repeats_an_already_tried_combo():
    experiments = [
        _combo_exp(1, 1, 100, throughput=120, latency=90),
        _combo_exp(2, 4, 100, throughput=330, latency=160),
        # The natural "double batch_size, hold num_requests" candidate was already tried.
        _combo_exp(3, 8, 100, throughput=350, latency=200),
    ]

    rec = suggest_untried_combo(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    tried = {(1, 100), (4, 100), (8, 100)}
    assert rec.suggested is not None
    assert (rec.suggested["serving.batch_size"], rec.suggested["serving.num_requests"]) not in tried


def test_suggest_untried_combo_respects_budgets():
    experiments = [
        _combo_exp(1, 4, 200, throughput=300, latency=150),
        _combo_exp(2, 8, 400, throughput=430, latency=260),  # breaches budget, excluded
    ]

    rec = suggest_untried_combo(
        experiments,
        knobs=["serving.batch_size", "serving.num_requests"],
        latency_budget_ms=200,
    )

    assert rec.best is not None
    assert rec.best.values == {"serving.batch_size": 4, "serving.num_requests": 200}


def test_suggest_joint_step_with_no_experiments():
    rec = suggest_joint_step([], knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is None
    assert rec.suggested is None
    assert "No completed" in rec.reason


def test_suggest_joint_step_can_move_two_knobs_at_once():
    """Unlike suggest_untried_combo, suggest_joint_step is allowed to change more than one
    knob per suggestion when both independently look like they're still scaling well."""
    experiments = [
        _combo_exp(1, 1, 100, throughput=120, latency=90),
        # batch_size alone (num_requests=100): 1 -> 4 outpaces.
        _combo_exp(2, 4, 100, throughput=330, latency=160),
        # num_requests alone (batch_size=4): 100 -> 200 outpaces even harder.
        _combo_exp(3, 4, 200, throughput=500, latency=170),
    ]

    rec = suggest_joint_step(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    assert rec.best is not None
    assert rec.best.values == {"serving.batch_size": 4, "serving.num_requests": 200}
    assert rec.suggested is not None
    # Both knobs moved in one suggestion, not just one.
    assert rec.suggested == {"serving.batch_size": 8, "serving.num_requests": 400}
    assert "serving.batch_size" in rec.reason
    assert "serving.num_requests" in rec.reason


def test_suggest_joint_step_never_repeats_an_already_tried_combo():
    experiments = [
        _combo_exp(1, 1, 100, throughput=120, latency=90),
        _combo_exp(2, 4, 100, throughput=330, latency=160),
        _combo_exp(3, 4, 200, throughput=500, latency=170),
        # The natural top candidate (batch_size=8, num_requests=400) was already tried.
        _combo_exp(4, 8, 400, throughput=520, latency=175),
    ]

    rec = suggest_joint_step(experiments, knobs=["serving.batch_size", "serving.num_requests"])

    tried = {(1, 100), (4, 100), (4, 200), (8, 400)}
    assert rec.suggested is not None
    assert (rec.suggested["serving.batch_size"], rec.suggested["serving.num_requests"]) not in tried


def test_suggest_joint_step_respects_budgets():
    experiments = [
        _combo_exp(1, 4, 200, throughput=300, latency=150),
        _combo_exp(2, 8, 400, throughput=430, latency=260),  # breaches budget, excluded
    ]

    rec = suggest_joint_step(
        experiments,
        knobs=["serving.batch_size", "serving.num_requests"],
        latency_budget_ms=200,
    )

    assert rec.best is not None
    assert rec.best.values == {"serving.batch_size": 4, "serving.num_requests": 200}


def test_suggest_joint_step_respects_max_candidates_cap():
    experiments = [
        _combo_exp(1, 1, 100, throughput=120, latency=90),
        _combo_exp(2, 4, 100, throughput=330, latency=160),
        _combo_exp(3, 4, 200, throughput=500, latency=170),
    ]

    rec = suggest_joint_step(
        experiments,
        knobs=["serving.batch_size", "serving.num_requests"],
        max_candidates=1,
    )

    assert rec.suggested is not None
    assert "kept top 1" in rec.reason
