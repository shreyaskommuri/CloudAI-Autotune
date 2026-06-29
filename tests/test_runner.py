from autotune.runner import CloudAIRunner


def test_run_returns_failed_result_when_binary_is_missing(tmp_path):
    runner = CloudAIRunner(cloudai_bin="this-binary-does-not-exist", runs_dir=tmp_path)

    result = runner.run(tmp_path / "scenario.toml", run_id="0001_test")

    assert not result.succeeded
    assert result.returncode != 0
    assert result.stdout_path.exists()
    log = result.stdout_path.read_text()
    assert "autotune: command: this-binary-does-not-exist run" in log
    assert "this-binary-does-not-exist" in log
    assert "autotune: CloudAI launch failed:" in log
    assert result.failure_reason is not None
    assert "launch failed" in result.failure_reason
    assert result.report_path is None


def test_run_returns_failed_result_when_command_exits_nonzero(tmp_path):
    cloudai = tmp_path / "failing-cloudai"
    cloudai.write_text("#!/bin/sh\necho 'backend exploded' >&2\nexit 7\n")
    cloudai.chmod(0o755)
    runner = CloudAIRunner(cloudai_bin=str(cloudai), runs_dir=tmp_path)

    result = runner.run(tmp_path / "scenario.toml", run_id="0002_test")

    assert not result.succeeded
    assert result.returncode == 7
    assert result.stdout_path.exists()
    log = result.stdout_path.read_text()
    assert log.splitlines()[0].startswith("autotune: command:")
    assert "backend exploded" in log
    assert "autotune: CloudAI exited with code 7" in log
    assert result.failure_reason == "CloudAI exited with code 7"
    assert result.report_path is None


def test_run_returns_failed_result_when_command_times_out(tmp_path):
    cloudai = tmp_path / "slow-cloudai"
    cloudai.write_text("#!/bin/sh\nsleep 1\n")
    cloudai.chmod(0o755)
    runner = CloudAIRunner(cloudai_bin=str(cloudai), runs_dir=tmp_path, timeout_sec=0.01)

    result = runner.run(tmp_path / "scenario.toml", run_id="0003_test")

    assert not result.succeeded
    assert result.returncode != 0
    assert result.stdout_path.exists()
    log = result.stdout_path.read_text()
    assert "autotune: CloudAI timed out after 0.01 seconds" in log
    assert result.failure_reason == "CloudAI timed out after 0.01 seconds"
    assert result.report_path is None


def test_run_succeeds_with_a_real_command(tmp_path):
    runner = CloudAIRunner(cloudai_bin="true", runs_dir=tmp_path)

    result = runner.run(tmp_path / "scenario.toml", run_id="0004_test")

    assert result.succeeded
    assert result.returncode == 0
    assert result.stdout_path.exists()
    assert "autotune: command: true run" in result.stdout_path.read_text()


def test_run_detects_common_summary_report_names(tmp_path):
    cloudai = tmp_path / "fake-cloudai"
    cloudai.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--output' ] || [ \"$1\" = '--output-dir' ]; then\n"
        "    shift\n"
        "    out=\"$1\"\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "dir=$(dirname \"$out\")\n"
        "mkdir -p \"$dir\"\n"
        "printf '{\"throughput_tokens_per_sec\": 123}' > \"$dir/summary.json\"\n"
    )
    cloudai.chmod(0o755)
    runner = CloudAIRunner(cloudai_bin=str(cloudai), runs_dir=tmp_path)

    result = runner.run(tmp_path / "scenario.toml", run_id="0005_test")

    assert result.succeeded
    assert result.report_path == tmp_path / "0005_test" / "summary.json"


def test_run_prefers_cloudai_summary_report(tmp_path):
    script = tmp_path / "fake_cloudai.sh"
    script.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--output' ] || [ \"$1\" = '--output-dir' ]; then\n"
        "    shift\n"
        "    out=\"$1\"\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "dir=$(dirname \"$out\")\n"
        "mkdir -p \"$dir\"\n"
        "printf '{\"metrics\":{\"throughput_tokens_per_sec\": 123}}' > \"$dir/cloudai-summary.json\"\n"
        "printf '{\"throughput_tokens_per_sec\": 1}' > \"$dir/summary.json\"\n"
    )
    script.chmod(0o755)
    runner = CloudAIRunner(cloudai_bin=str(script), runs_dir=tmp_path)

    result = runner.run(tmp_path / "config.toml", "0006_test")

    assert result.succeeded
    assert result.report_path == tmp_path / "0006_test" / "cloudai-summary.json"


def test_run_detects_cloudai_summary_in_scenario_output_directory(tmp_path):
    script = tmp_path / "fake_cloudai.sh"
    script.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '--output-dir' ]; then\n"
        "    shift\n"
        "    out=\"$1\"\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "mkdir -p \"$out/scenario_timestamp\"\n"
        "printf '{\"metrics\":{\"throughput_tokens_per_sec\": 123}}' "
        "> \"$out/scenario_timestamp/cloudai-summary.json\"\n"
    )
    script.chmod(0o755)
    runner = CloudAIRunner(
        cloudai_bin=str(script),
        runs_dir=tmp_path,
        dry_run=True,
        system_config=tmp_path / "system.toml",
    )

    result = runner.run(tmp_path / "config.toml", "0007_test")

    assert result.succeeded
    assert result.report_path == tmp_path / "0007_test" / "scenario_timestamp" / "cloudai-summary.json"


def test_command_string_uses_current_cloudai_cli_when_system_config_is_set(tmp_path):
    runner = CloudAIRunner(
        cloudai_bin="cloudai",
        runs_dir=tmp_path,
        dry_run=True,
        system_config=tmp_path / "system.toml",
        tests_dir=tmp_path / "tests",
        hook_dir=tmp_path / "hooks",
    )

    command = runner.command_string(tmp_path / "scenario.toml", tmp_path / "run" / "report.json")

    assert command == (
        "cloudai dry-run "
        "--test-scenario "
        f"{tmp_path / 'scenario.toml'} "
        "--system-config "
        f"{tmp_path / 'system.toml'} "
        "--output-dir "
        f"{tmp_path / 'run'} "
        "--tests-dir "
        f"{tmp_path / 'tests'} "
        "--hook-dir "
        f"{tmp_path / 'hooks'}"
    )
