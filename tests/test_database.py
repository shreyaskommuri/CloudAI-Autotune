import sqlite3

from autotune.database import ExperimentDB


def test_add_and_get_experiment(tmp_path):
    db = ExperimentDB(tmp_path / "test.db")
    try:
        exp_id = db.add_experiment(
            scenario="vllm_baseline",
            backend="vllm",
            config_path="configs/examples/vllm_baseline.toml",
            config={"serving": {"batch_size": 1}},
        )

        exp = db.get(exp_id)

        assert exp is not None
        assert exp.scenario == "vllm_baseline"
        assert exp.config["serving"]["batch_size"] == 1
        assert exp.status == "pending"
        assert exp.metrics == {}
        assert exp.metadata == {}
    finally:
        db.close()


def test_add_and_get_experiment_metadata(tmp_path):
    db = ExperimentDB(tmp_path / "test.db")
    try:
        exp_id = db.add_experiment(
            scenario="vllm_baseline",
            backend="vllm",
            config_path="cfg.toml",
            config={},
            metadata={"hardware.gpu": "A100", "run.nodes": 2},
        )

        exp = db.get(exp_id)

        assert exp is not None
        assert exp.metadata == {"hardware.gpu": "A100", "run.nodes": 2}
    finally:
        db.close()


def test_existing_database_is_migrated_for_metadata(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                scenario TEXT NOT NULL,
                backend TEXT NOT NULL,
                config_path TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                report_path TEXT,
                metrics_json TEXT,
                notes TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    db = ExperimentDB(db_path)
    try:
        exp_id = db.add_experiment(
            scenario="vllm_baseline",
            backend="vllm",
            config_path="cfg.toml",
            config={},
            metadata={"hardware.gpu": "H100"},
        )

        assert db.get(exp_id).metadata == {"hardware.gpu": "H100"}
    finally:
        db.close()


def test_update_result_and_list(tmp_path):
    db = ExperimentDB(tmp_path / "test.db")
    try:
        exp_id = db.add_experiment(
            scenario="vllm_baseline",
            backend="vllm",
            config_path="cfg.toml",
            config={"serving": {"batch_size": 4}},
        )

        db.update_result(
            exp_id,
            status="completed",
            report_path="runs/0001/report.json",
            metrics={"throughput_tokens_per_sec": 330.0, "latency_ms": 160.0},
        )

        exp = db.get(exp_id)
        assert exp.status == "completed"
        assert exp.metrics["throughput_tokens_per_sec"] == 330.0

        all_experiments = db.list_experiments()
        assert len(all_experiments) == 1

        scoped = db.list_experiments(scenario="vllm_baseline")
        assert len(scoped) == 1

        empty = db.list_experiments(scenario="does_not_exist")
        assert empty == []
    finally:
        db.close()
