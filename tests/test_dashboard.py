from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from autotune.database import ExperimentDB

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
