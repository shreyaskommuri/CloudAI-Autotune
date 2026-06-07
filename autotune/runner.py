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


@dataclass
class RunResult:
    run_id: str
    config_path: Path
    run_dir: Path
    returncode: int
    stdout_path: Path
    report_path: Optional[Path]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class CloudAIRunner:
    """Invokes the CloudAI CLI against a scenario config and captures output."""

    def __init__(
        self,
        cloudai_bin: str = "cloudai",
        runs_dir: Path | str = DEFAULT_RUNS_DIR,
        dry_run: bool = False,
        timeout_sec: Optional[int] = None,
    ):
        self.cloudai_bin = cloudai_bin
        self.runs_dir = Path(runs_dir)
        self.dry_run = dry_run
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
        with open(stdout_path, "w") as stdout_file:
            proc = subprocess.run(
                cmd,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_sec,
            )

        return RunResult(
            run_id=run_id,
            config_path=config_path,
            run_dir=run_dir,
            returncode=proc.returncode,
            stdout_path=stdout_path,
            report_path=report_path if report_path.exists() else None,
        )

    def _build_command(self, config_path: Path, report_path: Path) -> list[str]:
        cmd = [self.cloudai_bin, "run", "--config", str(config_path), "--output", str(report_path)]
        if self.dry_run:
            cmd.append("--dry-run")
        return cmd

    def command_string(self, config_path: Path | str, report_path: Path | str) -> str:
        """Return the shell-quoted command Autotune would run, for logging/debugging."""
        return " ".join(shlex.quote(part) for part in self._build_command(Path(config_path), Path(report_path)))
