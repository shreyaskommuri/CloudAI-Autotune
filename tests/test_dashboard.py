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
