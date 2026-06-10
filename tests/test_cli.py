from click.testing import CliRunner

from autotune.cli import cli


def test_ingest_records_existing_report_and_recommendation(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"

    ingests = [
        ("configs/examples/vllm_baseline.toml", "reports/examples/vllm_batch1.json"),
        ("configs/examples/vllm_batch4.toml", "reports/examples/vllm_batch4.json"),
        ("configs/examples/vllm_batch8.toml", "reports/examples/vllm_batch8.json"),
    ]
    for config_path, report_path in ingests:
        result = runner.invoke(
            cli,
            ["ingest", report_path, "--config", config_path, "--db", str(db_path)],
        )

        assert result.exit_code == 0
        assert "ingested" in result.output

    listed = runner.invoke(cli, ["list", "--db", str(db_path)])
    assert listed.exit_code == 0
    assert listed.output.count("status=completed") == 3

    recommended = runner.invoke(
        cli,
        [
            "recommend",
            "--db",
            str(db_path),
            "--knob",
            "serving.batch_size",
            "--latency-budget-ms",
            "200",
        ],
    )
    assert recommended.exit_code == 0
    assert "Suggested: 6.0" in recommended.output
    assert "untested" in recommended.output


def test_ingest_accepts_cloudai_sglang_jsonl_report(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "sglang.db"

    result = runner.invoke(
        cli,
        [
            "ingest",
            "reports/examples/sglang_bench.jsonl",
            "--config",
            "configs/examples/sglang_baseline.toml",
            "--db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert "throughput_tokens_per_sec': 42.5" in result.output
    assert "failure_rate': 0.030000" in result.output


def test_demo_loads_sample_reports_and_prints_recommendation(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "autotune-demo.db"

    result = runner.invoke(cli, ["demo", "--db", str(db_path)])

    assert result.exit_code == 0
    assert result.output.count("demo ingested") == 4
    assert f"Demo database: {db_path}" in result.output
    assert "Scenario: vllm_baseline" in result.output
    assert "Knob: serving.batch_size" in result.output
    assert "Suggested: 6.0" in result.output
    assert db_path.exists()
