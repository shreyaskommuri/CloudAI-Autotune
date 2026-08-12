from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from autotune.database import ExperimentDB
from autotune.dse import DSETrial

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture()
def demo_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "dashboard-test.db"
    with ExperimentDB(db_path) as db:
        for batch_size, throughput, latency in ((1, 120, 90), (4, 330, 160), (8, 430, 260)):
            experiment_id = db.add_experiment(
                scenario="vllm_baseline",
                backend="vllm",
                config_path=f"batch{batch_size}.toml",
                config={"serving": {"batch_size": batch_size}},
            )
            db.update_result(
                experiment_id,
                status="completed",
                metrics={"throughput_tokens_per_sec": throughput, "latency_ms": latency},
            )
    return db_path


def test_dashboard_loads_without_a_database(tmp_path: Path):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert any("No database found" in warning.value for warning in at.warning)


def test_dashboard_renders_recommendation_for_existing_runs(demo_db: Path):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value(str(demo_db))
    at.run(timeout=30)

    assert not at.exception
    metric_labels = [metric.label for metric in at.main.metric]
    assert "Suggested next value for `serving.batch_size`" in metric_labels


def test_dashboard_runtime_budget_input_reaches_recommendation(demo_db: Path):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value(str(demo_db))
    at.run(timeout=30)

    labels = [number_input.label for number_input in at.sidebar.number_input]
    runtime_budget_index = labels.index("Runtime budget (sec, optional)")
    at.sidebar.number_input[runtime_budget_index].set_value(1.0)
    at.run(timeout=30)

    assert not at.exception


def test_dashboard_flags_a_regression_against_the_immediately_preceding_run(demo_db: Path):
    """The batch8 run in demo_db has worse latency than batch4, even though it's
    still the best throughput run overall — this should surface as a regression
    against the immediately preceding run, distinct from the best-vs-latest check.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value(str(demo_db))
    at.run(timeout=30)

    assert not at.exception
    assert any("Regression vs. previous run" in error.value for error in at.error)


@pytest.fixture()
def dse_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "dse-test.db"
    with ExperimentDB(db_path) as db:
        exp_id = db.add_experiment(
            scenario="nemo_run_sweep",
            backend="cloudai-dse",
            config_path="results/nemo_run_sweep/0/trajectory.csv",
            config={},
        )
        db.add_dse_trials(
            exp_id,
            [
                DSETrial(step=1, action={"batch_size": 1}, reward=0.5, observation=[100.0]),
                DSETrial(step=2, action={"batch_size": 4}, reward=1.2, observation=[330.0]),
            ],
        )
        db.update_result(exp_id, status="completed", metrics={"best_reward": 1.2, "best_step": 2, "num_trials": 2})
    return db_path


def test_dashboard_shows_dse_sweep_best_reward_and_action(dse_db: Path):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value(str(dse_db))
    at.run(timeout=30)

    assert not at.exception
    metric_labels = [metric.label for metric in at.main.metric]
    assert "Best reward" in metric_labels


def test_dashboard_hides_dse_section_when_no_sweeps_present(demo_db: Path):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value(str(demo_db))
    at.run(timeout=30)

    assert not at.exception
    headers = [subheader.value for subheader in at.main.subheader]
    assert "DSE sweeps" not in headers


@pytest.fixture()
def multi_knob_db(tmp_path: Path) -> Path:
    """9 combos, 3 distinct values per knob - enough for --optimize's fitted surface, not
    just the heuristic fallback (mirrors test_optimizer.py's RICH_COMBOS)."""
    db_path = tmp_path / "multi-knob-test.db"
    combos = [
        (1, 100, 120, 90),
        (2, 100, 200, 110),
        (4, 100, 330, 160),
        (1, 200, 180, 140),
        (2, 200, 300, 165),
        (4, 200, 500, 170),
        (8, 200, 520, 175),
        (2, 400, 340, 200),
        (4, 400, 560, 210),
    ]
    with ExperimentDB(db_path) as db:
        for batch_size, num_requests, throughput, latency in combos:
            experiment_id = db.add_experiment(
                scenario="vllm_baseline",
                backend="vllm",
                config_path=f"c{batch_size}_{num_requests}.toml",
                config={"serving": {"batch_size": batch_size, "num_requests": num_requests}},
            )
            db.update_result(
                experiment_id,
                status="completed",
                metrics={"throughput_tokens_per_sec": throughput, "latency_ms": latency},
            )
    return db_path


def _load_with_knobs_selected(db_path: Path):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value(str(db_path))
    at.run(timeout=30)
    at.main.multiselect[0].set_value(["serving.batch_size", "serving.num_requests"])
    at.run(timeout=30)
    return at


def test_multi_knob_section_prompts_when_fewer_than_two_knobs_selected(multi_knob_db: Path):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value(str(multi_knob_db))
    at.run(timeout=30)

    assert not at.exception
    captions = [caption.value for caption in at.main.caption]
    assert any("Select at least two knobs" in caption for caption in captions)


def test_multi_knob_section_shows_best_combo_and_frontier(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)

    assert not at.exception
    metric_labels = [metric.label for metric in at.main.metric]
    assert any(label.startswith("Best combo:") for label in metric_labels)
    dataframe_columns = [set(df.value.columns) for df in at.main.dataframe]
    assert any({"serving.batch_size", "serving.num_requests", "efficiency"} <= columns for columns in dataframe_columns)


def test_explore_mode_suggests_an_untried_combo(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)
    at.main.radio[0].set_value("Explore")
    at.run(timeout=30)

    assert not at.exception
    assert any("Suggested untried combo:" in text.value for text in at.main.markdown)


def test_search_mode_suggests_an_untried_combo(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)
    at.main.radio[0].set_value("Search")
    at.run(timeout=30)

    assert not at.exception
    assert any("Suggested untried combo:" in text.value for text in at.main.markdown)


def test_optimize_mode_shows_a_predicted_efficiency(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)
    at.main.radio[0].set_value("Optimize")
    at.run(timeout=30)

    assert not at.exception
    assert any("Predicted efficiency:" in text.value for text in at.main.markdown)


def test_optimize_mode_does_not_record_a_suggestion_without_a_button_click(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)
    at.main.radio[0].set_value("Optimize")
    at.run(timeout=30)

    with ExperimentDB(multi_knob_db) as db:
        assert db.list_suggestions(["serving.batch_size", "serving.num_requests"]) == []


def test_optimize_mode_records_a_suggestion_on_button_click(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)
    at.main.radio[0].set_value("Optimize")
    at.run(timeout=30)
    at.main.button[0].click()
    at.run(timeout=30)

    assert not at.exception
    with ExperimentDB(multi_knob_db) as db:
        suggestions = db.list_suggestions(["serving.batch_size", "serving.num_requests"])
    assert len(suggestions) == 1
    assert suggestions[0].predicted_efficiency is not None


def _history_dataframe(at: AppTest):
    for dataframe in at.main.dataframe:
        if "predicted_efficiency" in dataframe.value.columns:
            return dataframe.value
    return None


def test_suggestion_history_is_empty_before_any_suggestion_is_recorded(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)

    assert not at.exception
    captions = [caption.value for caption in at.main.caption]
    assert any("No suggestions recorded yet" in caption for caption in captions)
    assert _history_dataframe(at) is None


def test_suggestion_history_shows_a_pending_row_after_recording(multi_knob_db: Path):
    at = _load_with_knobs_selected(multi_knob_db)
    at.main.radio[0].set_value("Optimize")
    at.run(timeout=30)
    at.main.button[0].click()
    at.run(timeout=30)

    history = _history_dataframe(at)
    assert history is not None
    assert len(history) == 1
    assert history.iloc[0]["status"] == "Pending"
    assert history.iloc[0]["actual_efficiency"] is None


def test_suggestion_history_resolves_regardless_of_active_mode(multi_knob_db: Path):
    """Resolution isn't gated to Optimize mode - a suggestion recorded earlier should resolve
    even while the user is looking at Explore or Search, since it's shared setup for the whole
    multi-knob section, not per-mode."""
    at = _load_with_knobs_selected(multi_knob_db)
    at.main.radio[0].set_value("Optimize")
    at.run(timeout=30)
    at.main.button[0].click()
    at.run(timeout=30)

    with ExperimentDB(multi_knob_db) as db:
        suggested_values = db.list_suggestions(["serving.batch_size", "serving.num_requests"])[0].suggested_values
        experiment_id = db.add_experiment(
            scenario="vllm_baseline",
            backend="vllm",
            config_path="c_suggested.toml",
            config={
                "serving": {
                    "batch_size": suggested_values["serving.batch_size"],
                    "num_requests": suggested_values["serving.num_requests"],
                }
            },
        )
        db.update_result(
            experiment_id, status="completed", metrics={"throughput_tokens_per_sec": 600, "latency_ms": 230}
        )

    at.main.radio[0].set_value("Explore")
    at.run(timeout=30)

    assert not at.exception
    history = _history_dataframe(at)
    assert history.iloc[0]["status"] == "Resolved"
    assert history.iloc[0]["actual_efficiency"] is not None
