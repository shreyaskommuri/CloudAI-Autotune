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
    assert "Suggested: 4.0" in recommended.output
