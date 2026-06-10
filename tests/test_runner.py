from autotune.runner import CloudAIRunner


def test_run_returns_failed_result_when_binary_is_missing(tmp_path):
    runner = CloudAIRunner(cloudai_bin="this-binary-does-not-exist", runs_dir=tmp_path)

    result = runner.run(tmp_path / "scenario.toml", run_id="0001_test")

    assert not result.succeeded
    assert result.returncode != 0
    assert result.stdout_path.exists()
    assert "this-binary-does-not-exist" in result.stdout_path.read_text()
    assert result.report_path is None


def test_run_returns_failed_result_when_command_exits_nonzero(tmp_path):
    runner = CloudAIRunner(cloudai_bin="false", runs_dir=tmp_path)

    result = runner.run(tmp_path / "scenario.toml", run_id="0002_test")

    assert not result.succeeded
    assert result.returncode != 0
    assert result.stdout_path.exists()
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
    assert "timed out" in result.stdout_path.read_text()
    assert result.report_path is None


def test_run_succeeds_with_a_real_command(tmp_path):
    runner = CloudAIRunner(cloudai_bin="true", runs_dir=tmp_path)

    result = runner.run(tmp_path / "scenario.toml", run_id="0004_test")

    assert result.succeeded
    assert result.returncode == 0
    assert result.stdout_path.exists()


def test_command_string_uses_current_cloudai_cli_when_system_config_is_set(tmp_path):
    runner = CloudAIRunner(
        cloudai_bin="cloudai",
        runs_dir=tmp_path,
        dry_run=True,
        system_config=tmp_path / "system.toml",
        tests_dir=tmp_path / "tests",
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
        f"{tmp_path / 'tests'}"
    )
