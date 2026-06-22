"""Wrapper that invokes CloudAI to execute (or dry-run) a scenario config.

This module shells out to the `cloudai` CLI so Autotune stays a thin control
layer rather than reimplementing benchmark execution. Each run's stdout/stderr
and any generated report are captured under `runs/<run_id>/`.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_RUNS_DIR = Path("runs")
REPORT_CANDIDATES = (
    "cloudai-summary.json",
    "summary.json",
    "results.json",
    "metrics.json",
    "report.jsonl",
    "summary.jsonl",
)


@dataclass
class RunResult:
    run_id: str
    config_path: Path
    run_dir: Path
    returncode: int
    stdout_path: Path
    report_path: Optional[Path]
    failure_reason: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def log_diagnostic(self, message: str) -> None:
        """Append an Autotune diagnostic without overwriting CloudAI output."""
        with open(self.stdout_path, "a") as stdout_file:
            stdout_file.write(f"\nautotune: {message}\n")


class CloudAIRunner:
    """Invokes the CloudAI CLI against a scenario config and captures output."""

    def __init__(
        self,
        cloudai_bin: str = "cloudai",
        runs_dir: Path | str = DEFAULT_RUNS_DIR,
        dry_run: bool = False,
        system_config: Path | str | None = None,
        tests_dir: Path | str | None = None,
        timeout_sec: Optional[float] = None,
    ):
        self.cloudai_bin = cloudai_bin
        self.runs_dir = Path(runs_dir)
        self.dry_run = dry_run
        self.system_config = Path(system_config) if system_config is not None else None
        self.tests_dir = Path(tests_dir) if tests_dir is not None else None
        self.timeout_sec = timeout_sec

    def run(self, config_path: Path | str, run_id: str) -> RunResult:
        """Run (or dry-run) CloudAI against `config_path`.

        Returns a RunResult pointing at captured stdout and, if produced, a
        report file that `parser.parse_report` can consume.
        """
        config_path = Path(config_path)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        report_path = run_dir / "report.json"
        cmd = self._build_command(config_path, report_path)

        stdout_path = run_dir / "stdout.log"
        failure_reason: Optional[str] = None
        with open(stdout_path, "w") as stdout_file:
            stdout_file.write(f"autotune: command: {' '.join(shlex.quote(part) for part in cmd)}\n")
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_sec,
                )
                returncode = proc.returncode
                if returncode != 0:
                    failure_reason = f"CloudAI exited with code {returncode}"
                    stdout_file.write(f"\nautotune: {failure_reason}\n")
            except subprocess.TimeoutExpired:
                failure_reason = f"CloudAI timed out after {self.timeout_sec} seconds"
                stdout_file.write(f"\nautotune: {failure_reason}\n")
                returncode = -1
            except OSError as exc:
                failure_reason = f"CloudAI launch failed: {exc}"
                stdout_file.write(f"\nautotune: {failure_reason}\n")
                returncode = -1

        return RunResult(
            run_id=run_id,
            config_path=config_path,
            run_dir=run_dir,
            returncode=returncode,
            stdout_path=stdout_path,
            report_path=_find_report(report_path),
            failure_reason=failure_reason,
        )

    def _build_command(self, config_path: Path, report_path: Path) -> list[str]:
        if self.system_config is not None:
            cmd = [
                self.cloudai_bin,
                "dry-run" if self.dry_run else "run",
                "--test-scenario",
                str(config_path),
                "--system-config",
                str(self.system_config),
                "--output-dir",
                str(report_path.parent),
            ]
            if self.tests_dir is not None:
                cmd.extend(["--tests-dir", str(self.tests_dir)])
            return cmd

        cmd = [self.cloudai_bin, "run", "--config", str(config_path), "--output", str(report_path)]
        if self.dry_run:
            cmd.append("--dry-run")
        return cmd

    def command_string(self, config_path: Path | str, report_path: Path | str) -> str:
        """Return the shell-quoted command Autotune would run, for logging/debugging."""
        return " ".join(shlex.quote(part) for part in self._build_command(Path(config_path), Path(report_path)))


def _find_report(expected_path: Path) -> Optional[Path]:
    if expected_path.exists():
        return expected_path
    for name in REPORT_CANDIDATES:
        candidate = expected_path.parent / name
        if candidate.exists():
            return candidate
    return None
