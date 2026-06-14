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


def test_ingest_can_record_report_without_config(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"

    result = runner.invoke(
        cli,
        [
            "ingest",
            "reports/examples/vllm_batch4.json",
            "--db",
            str(db_path),
            "--scenario",
            "manual_vllm",
            "--backend",
            "vllm",
            "--set",
            "serving.batch_size=4",
        ],
    )
    listed = runner.invoke(cli, ["list", "--db", str(db_path)])
    exported = runner.invoke(cli, ["export", "--db", str(db_path), "--format", "json"])
    recommended = runner.invoke(
        cli,
        [
            "recommend",
            "--db",
            str(db_path),
            "--scenario",
            "manual_vllm",
            "--knob",
            "serving.batch_size",
        ],
    )

    assert result.exit_code == 0
    assert listed.exit_code == 0
    assert exported.exit_code == 0
    assert recommended.exit_code == 0
    assert "manual_vllm (vllm) status=completed" in listed.output
    assert '"config_path": "reports/examples/vllm_batch4.json"' in exported.output
    assert "Current: 4.0  ->  Suggested: 8.0" in recommended.output


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


def test_ingest_records_notes_for_experiment_context(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"

    result = runner.invoke(
        cli,
        [
            "ingest",
            "reports/examples/vllm_batch4.json",
            "--config",
            "configs/examples/vllm_batch4.toml",
            "--db",
            str(db_path),
            "--notes",
            "baseline before tensor-parallel change",
        ],
    )
    listed = runner.invoke(cli, ["list", "--db", str(db_path)])
    exported = runner.invoke(cli, ["export", "--db", str(db_path), "--format", "json"])

    assert result.exit_code == 0
    assert listed.exit_code == 0
    assert exported.exit_code == 0
    assert "baseline before tensor-parallel change" in listed.output
    assert '"notes": "baseline before tensor-parallel change"' in exported.output


def test_ingest_records_metadata_for_environment_context(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"

    result = runner.invoke(
        cli,
        [
            "ingest",
            "reports/examples/vllm_batch4.json",
            "--config",
            "configs/examples/vllm_batch4.toml",
            "--db",
            str(db_path),
            "--metadata",
            "hardware.gpu=A100",
            "--metadata",
            "run.nodes=2",
        ],
    )
    listed = runner.invoke(cli, ["list", "--db", str(db_path)])
    exported = runner.invoke(cli, ["export", "--db", str(db_path), "--format", "json"])

    assert result.exit_code == 0
    assert listed.exit_code == 0
    assert exported.exit_code == 0
    assert "metadata={'hardware.gpu': 'A100', 'run.nodes': 2}" in listed.output
    assert '"metadata.hardware.gpu": "A100"' in exported.output
    assert '"metadata.run.nodes": 2' in exported.output


def test_ingest_rejects_invalid_metadata(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "ingest",
            "reports/examples/vllm_batch4.json",
            "--config",
            "configs/examples/vllm_batch4.toml",
            "--db",
            str(tmp_path / "demo.db"),
            "--metadata",
            "hardware.gpu",
        ],
    )

    assert result.exit_code != 0
    assert "Expected key=value" in result.output


def test_diff_reports_config_and_metric_changes(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"
    for config_path, report_path in [
        ("configs/examples/vllm_baseline.toml", "reports/examples/vllm_batch1.json"),
        ("configs/examples/vllm_batch4.toml", "reports/examples/vllm_batch4.json"),
    ]:
        result = runner.invoke(
            cli,
            ["ingest", report_path, "--config", config_path, "--db", str(db_path)],
        )
        assert result.exit_code == 0

    diff = runner.invoke(cli, ["diff", "1", "2", "--db", str(db_path)])

    assert diff.exit_code == 0
    assert "Comparing [1] vllm_baseline -> [2] vllm_baseline" in diff.output
    assert "serving.batch_size: 1 -> 4" in diff.output
    assert "throughput_tokens_per_sec: 120.0 -> 330.0" in diff.output


def test_diff_reports_missing_experiment(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["diff", "1", "2", "--db", str(tmp_path / "empty.db")])

    assert result.exit_code != 0
    assert "Experiment 1 not found" in result.output


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


def test_export_writes_markdown_to_stdout(tmp_path):
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

    result = runner.invoke(cli, ["export", "--db", str(db_path), "--format", "markdown"])

    assert ingest.exit_code == 0
    assert result.exit_code == 0
    assert "| id | created_at | scenario | backend | status |" in result.output
    assert "| --- | --- | --- | --- | --- |" in result.output
    assert "vllm_baseline" in result.output
    assert "metric.throughput_tokens_per_sec" in result.output
    assert "330.0" in result.output


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
    assert "Summary: 1 pass, 1 fail, 0 unknown (2 total)" in result.output


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


def test_check_supports_ttft_budget(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "demo.db"
    report_path = tmp_path / "report.json"
    report_path.write_text(
        '{"throughput_tokens_per_sec": 330, "latency_ms": 160, "ttft_ms": 75}'
    )
    ingest = runner.invoke(
        cli,
        [
            "ingest",
            str(report_path),
            "--config",
            "configs/examples/vllm_batch4.toml",
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
            "--ttft-budget-ms",
            "50",
        ],
    )

    assert ingest.exit_code == 0
    assert result.exit_code == 0
    assert "[1] vllm_baseline status=fail" in result.output
    assert "ttft 75ms exceeds budget 50ms" in result.output
    assert "Summary: 0 pass, 1 fail, 0 unknown (1 total)" in result.output


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
