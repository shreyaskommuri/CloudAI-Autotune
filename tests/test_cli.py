from click.testing import CliRunner

from autotune import config_mutator
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


def test_recommend_can_write_suggested_config(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"
    out_config = tmp_path / "batch6.toml"

    ingests = [
        ("configs/examples/vllm_baseline.toml", "reports/examples/vllm_batch1.json"),
        ("configs/examples/vllm_batch4.toml", "reports/examples/vllm_batch4.json"),
        ("configs/examples/vllm_batch8.toml", "reports/examples/vllm_batch8.json"),
    ]
    for config_path, report_path in ingests:
        runner.invoke(
            cli,
            ["ingest", report_path, "--config", config_path, "--db", str(db_path)],
        )

    result = runner.invoke(
        cli,
        [
            "recommend",
            "--db",
            str(db_path),
            "--knob",
            "serving.batch_size",
            "--latency-budget-ms",
            "200",
            "--derive-from",
            "configs/examples/vllm_baseline.toml",
            "--out-config",
            str(out_config),
        ],
    )

    assert result.exit_code == 0
    assert "Suggested: 6.0" in result.output
    assert f"Wrote suggested config to {out_config}" in result.output
    assert config_mutator.load_config(out_config)["serving"]["batch_size"] == 6.0


def test_recommend_requires_config_write_options_together(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "recommend",
            "--db",
            str(tmp_path / "empty.db"),
            "--out-config",
            str(tmp_path / "next.toml"),
        ],
    )

    assert result.exit_code != 0
    assert "--derive-from and --out-config must be provided together" in result.output


def test_export_writes_csv_to_stdout(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"
    ingest = runner.invoke(
        cli,
        [
            "ingest",
            "reports/examples/vllm_batch4.json",
            "--config",
            "configs/examples/vllm_batch4.toml",
            "--db",
            str(db_path),
        ],
    )

    result = runner.invoke(cli, ["export", "--db", str(db_path)])

    assert ingest.exit_code == 0
    assert result.exit_code == 0
    assert "scenario,backend,status" in result.output
    assert "vllm_baseline" in result.output
    assert "metric.throughput_tokens_per_sec" in result.output
    assert "330.0" in result.output


def test_export_writes_json_file_with_scenario_filter(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"
    out_path = tmp_path / "exports" / "vllm.json"
    for config_path, report_path in [
        ("configs/examples/vllm_batch4.toml", "reports/examples/vllm_batch4.json"),
        ("configs/examples/sglang_baseline.toml", "reports/examples/sglang_bench.jsonl"),
    ]:
        runner.invoke(
            cli,
            ["ingest", report_path, "--config", config_path, "--db", str(db_path)],
        )

    result = runner.invoke(
        cli,
        [
            "export",
            "--db",
            str(db_path),
            "--scenario",
            "vllm_baseline",
            "--format",
            "json",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert f"Exported 1 experiments to {out_path}" in result.output
    assert '"scenario": "vllm_baseline"' in out_path.read_text()
    assert "sglang_baseline" not in out_path.read_text()


def test_check_reports_budget_pass_and_failure(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"
    for config_path, report_path in [
        ("configs/examples/vllm_batch4.toml", "reports/examples/vllm_batch4.json"),
        ("configs/examples/vllm_batch8.toml", "reports/examples/vllm_batch8.json"),
    ]:
        ingest = runner.invoke(
            cli,
            ["ingest", report_path, "--config", config_path, "--db", str(db_path)],
        )
        assert ingest.exit_code == 0

    result = runner.invoke(
        cli,
        [
            "check",
            "--db",
            str(db_path),
            "--latency-budget-ms",
            "200",
            "--min-throughput-tokens-per-sec",
            "300",
        ],
    )

    assert result.exit_code == 0
    assert "[1] vllm_baseline status=pass" in result.output
    assert "[2] vllm_baseline status=fail" in result.output
    assert "latency 260ms exceeds budget 200ms" in result.output


def test_check_strict_exits_nonzero_on_failure(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"
    runner.invoke(
        cli,
        [
            "ingest",
            "reports/examples/vllm_batch8.json",
            "--config",
            "configs/examples/vllm_batch8.toml",
            "--db",
            str(db_path),
        ],
    )

    result = runner.invoke(
        cli,
        [
            "check",
            "--db",
            str(db_path),
            "--latency-budget-ms",
            "200",
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "status=fail" in result.output


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


def test_recommend_handles_ingested_report_without_usable_metrics(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "empty-metrics.db"
    report_path = tmp_path / "stdout.log"
    report_path.write_text("benchmark finished without recognized metrics")

    ingested = runner.invoke(
        cli,
        [
            "ingest",
            str(report_path),
            "--config",
            "configs/examples/vllm_baseline.toml",
            "--db",
            str(db_path),
        ],
    )
    recommended = runner.invoke(cli, ["recommend", "--db", str(db_path)])

    assert ingested.exit_code == 0
    assert recommended.exit_code == 0
    assert "usable throughput and latency metrics" in recommended.output


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
