"""Command-line entry point tying the loader, runner, parser, DB, and recommender together."""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import Optional

import click

from autotune import config_mutator
from autotune.budgets import Budgets, evaluate_experiment
from autotune.database import Experiment, ExperimentDB
from autotune.parser import parse_report
from autotune.recommender import DEFAULT_KNOB, recommend_next
from autotune.runner import CloudAIRunner

DEMO_REPORTS = (
    ("configs/examples/vllm_baseline.toml", "reports/examples/vllm_batch1.json"),
    ("configs/examples/vllm_batch4.toml", "reports/examples/vllm_batch4.json"),
    ("configs/examples/vllm_batch8.toml", "reports/examples/vllm_batch8.json"),
    ("configs/examples/sglang_baseline.toml", "reports/examples/sglang_bench.jsonl"),
)


@click.group()
def cli() -> None:
    """CloudAI Autotune — closed-loop benchmark experiment manager."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", "db_path", default="autotune.db", help="Path to the experiment database.")
@click.option("--dry-run", is_flag=True, help="Pass --dry-run through to CloudAI without executing.")
@click.option("--cloudai-bin", default="cloudai", help="Name/path of the CloudAI CLI binary.")
@click.option("--system-config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--tests-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
def run(
    config_path: Path,
    db_path: str,
    dry_run: bool,
    cloudai_bin: str,
    system_config: Optional[Path],
    tests_dir: Optional[Path],
) -> None:
    """Run a CloudAI scenario, parse its report, and record the experiment."""
    config = config_mutator.load_config(config_path)
    scenario = config.get("scenario", {})
    scenario_name = scenario.get("name") if isinstance(scenario, dict) else None
    backend = scenario.get("backend") if isinstance(scenario, dict) else None

    with ExperimentDB(db_path) as db:
        experiment_id = db.add_experiment(
            scenario=scenario_name or config.get("name", config_path.stem),
            backend=backend or "unknown",
            config_path=str(config_path),
            config=config,
            status="running",
        )

        run_id = f"{experiment_id:04d}_{config_path.stem}_{int(time.time())}"
        runner = CloudAIRunner(
            cloudai_bin=cloudai_bin,
            dry_run=dry_run,
            system_config=system_config,
            tests_dir=tests_dir,
        )
        result = runner.run(config_path, run_id)

        if not result.succeeded:
            db.update_result(experiment_id, status="failed", report_path=str(result.stdout_path))
            click.echo(f"[{experiment_id}] CloudAI exited with code {result.returncode}. See {result.stdout_path}")
            return

        report_source = result.report_path or result.stdout_path
        metrics = parse_report(report_source)
        db.update_result(
            experiment_id,
            status="completed",
            report_path=str(report_source),
            metrics=metrics,
        )

        click.echo(f"[{experiment_id}] completed — {metrics}")


@cli.command(name="list")
@click.option("--db", "db_path", default="autotune.db")
@click.option("--scenario", default=None, help="Filter by scenario name.")
def list_experiments(db_path: str, scenario: Optional[str]) -> None:
    """List recorded experiments and their metrics."""
    with ExperimentDB(db_path) as db:
        experiments = db.list_experiments(scenario=scenario)
        if not experiments:
            click.echo("No experiments recorded yet.")
            return
        for exp in experiments:
            click.echo(
                f"[{exp.id}] {exp.scenario} ({exp.backend}) status={exp.status} "
                f"metrics={exp.metrics}"
            )


@cli.command()
@click.option("--db", "db_path", default="autotune.db")
@click.option("--scenario", default=None, help="Restrict export to one scenario.")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(("csv", "json")),
    default="csv",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write export to a file.",
)
def export(
    db_path: str,
    scenario: Optional[str],
    export_format: str,
    out_path: Optional[Path],
) -> None:
    """Export experiment history as CSV or JSON."""
    with ExperimentDB(db_path) as db:
        rows = [_experiment_row(exp) for exp in db.list_experiments(scenario=scenario)]

    if export_format == "json":
        output = json.dumps(rows, indent=2) + "\n"
    else:
        output = _rows_to_csv(rows)

    if out_path is None:
        click.echo(output, nl=False)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    click.echo(f"Exported {len(rows)} experiments to {out_path}")


@cli.command()
@click.option("--db", "db_path", default="autotune.db")
@click.option("--scenario", default=None, help="Restrict checks to one scenario.")
@click.option(
    "--latency-budget-ms",
    type=float,
    default=None,
    help="Maximum acceptable latency in ms.",
)
@click.option(
    "--min-throughput-tokens-per-sec",
    type=float,
    default=None,
    help="Minimum acceptable generated-token throughput.",
)
@click.option(
    "--runtime-budget-sec",
    type=float,
    default=None,
    help="Maximum acceptable runtime in seconds.",
)
@click.option(
    "--max-failure-rate",
    type=float,
    default=None,
    help="Maximum acceptable failed-request ratio.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero when any experiment fails or cannot be evaluated.",
)
def check(
    db_path: str,
    scenario: Optional[str],
    latency_budget_ms: Optional[float],
    min_throughput_tokens_per_sec: Optional[float],
    runtime_budget_sec: Optional[float],
    max_failure_rate: Optional[float],
    strict: bool,
) -> None:
    """Check recorded experiments against metric budgets."""
    budgets = Budgets(
        latency_ms=latency_budget_ms,
        min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
        runtime_sec=runtime_budget_sec,
        failure_rate=max_failure_rate,
    )
    with ExperimentDB(db_path) as db:
        experiments = db.list_experiments(scenario=scenario)

    if not experiments:
        click.echo("No experiments recorded yet.")
        if strict:
            raise click.exceptions.Exit(1)
        return

    checks = [evaluate_experiment(exp, budgets) for exp in experiments]
    for exp, check_result in zip(experiments, checks):
        reason = "; ".join(check_result.reasons)
        click.echo(f"[{exp.id}] {exp.scenario} status={check_result.status} - {reason}")

    if strict and any(check_result.status != "pass" for check_result in checks):
        raise click.exceptions.Exit(1)


def _experiment_row(exp: Experiment) -> dict[str, object]:
    row: dict[str, object] = {
        "id": exp.id,
        "created_at": exp.created_at,
        "scenario": exp.scenario,
        "backend": exp.backend,
        "status": exp.status,
        "config_path": exp.config_path,
        "report_path": exp.report_path,
        "notes": exp.notes,
    }
    for key, value in exp.metrics.items():
        row[f"metric.{key}"] = value
    return row


def _rows_to_csv(rows: list[dict[str, object]]) -> str:
    fieldnames = _export_fieldnames(rows)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _export_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    base = [
        "id",
        "created_at",
        "scenario",
        "backend",
        "status",
        "config_path",
        "report_path",
        "notes",
    ]
    metric_keys = sorted({key for row in rows for key in row if key.startswith("metric.")})
    return base + metric_keys


@cli.command()
@click.argument("report_path", type=click.Path(exists=True, path_type=Path))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--db", "db_path", default="autotune.db", help="Path to the experiment database.")
def ingest(report_path: Path, config_path: Path, db_path: str) -> None:
    """Record an existing CloudAI report without launching CloudAI."""
    experiment_id, metrics = _ingest_report(report_path, config_path, db_path)

    click.echo(f"[{experiment_id}] ingested {report_path} — {metrics}")


@cli.command()
@click.option("--db", "db_path", default="autotune-demo.db", help="Path to the demo database.")
@click.option("--scenario", default="vllm_baseline", help="Scenario to use for the demo recommendation.")
@click.option("--knob", default=DEFAULT_KNOB, help="Dotted config key to tune, e.g. serving.batch_size.")
@click.option("--latency-budget-ms", type=float, default=200.0, help="Maximum acceptable latency in ms.")
def demo(db_path: str, scenario: str, knob: str, latency_budget_ms: float) -> None:
    """Load bundled sample reports and print a recommendation."""
    for config_path, report_path in DEMO_REPORTS:
        experiment_id, metrics = _ingest_report(Path(report_path), Path(config_path), db_path)
        click.echo(f"[{experiment_id}] demo ingested {report_path} — {metrics}")

    with ExperimentDB(db_path) as db:
        experiments = db.list_experiments(scenario=scenario)
        rec = recommend_next(experiments, knob=knob, latency_budget_ms=latency_budget_ms)

    click.echo(f"Demo database: {db_path}")
    click.echo(f"Scenario: {scenario}")
    click.echo(f"Knob: {rec.knob}")
    click.echo(f"Current: {rec.current_value}  ->  Suggested: {rec.suggested_value}")
    click.echo(f"Reason: {rec.reason}")


def _ingest_report(report_path: Path, config_path: Path, db_path: str) -> tuple[int, dict[str, object]]:
    config = config_mutator.load_config(config_path)
    scenario = config.get("scenario", {})
    scenario_name = scenario.get("name") if isinstance(scenario, dict) else None
    backend = scenario.get("backend") if isinstance(scenario, dict) else None
    metrics = parse_report(report_path)

    with ExperimentDB(db_path) as db:
        experiment_id = db.add_experiment(
            scenario=scenario_name or config.get("name", config_path.stem),
            backend=backend or "unknown",
            config_path=str(config_path),
            config=config,
            status="completed",
        )
        db.update_result(
            experiment_id,
            status="completed",
            report_path=str(report_path),
            metrics=metrics,
        )

    return experiment_id, metrics


@cli.command()
@click.option("--db", "db_path", default="autotune.db")
@click.option("--scenario", default=None, help="Restrict recommendation to one scenario's history.")
@click.option("--knob", default=DEFAULT_KNOB, help="Dotted config key to tune, e.g. serving.batch_size.")
@click.option("--latency-budget-ms", type=float, default=None, help="Maximum acceptable latency in ms.")
@click.option(
    "--derive-from",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Base config to copy when writing the suggested config.",
)
@click.option(
    "--out-config",
    type=click.Path(path_type=Path),
    default=None,
    help="Write a derived config with the suggested knob value.",
)
def recommend(
    db_path: str,
    scenario: Optional[str],
    knob: str,
    latency_budget_ms: Optional[float],
    derive_from: Optional[Path],
    out_config: Optional[Path],
) -> None:
    """Recommend the next config value to try based on experiment history."""
    if (derive_from is None) != (out_config is None):
        raise click.UsageError("--derive-from and --out-config must be provided together.")

    with ExperimentDB(db_path) as db:
        experiments = db.list_experiments(scenario=scenario)
        rec = recommend_next(experiments, knob=knob, latency_budget_ms=latency_budget_ms)
        click.echo(f"Knob: {rec.knob}")
        click.echo(f"Current: {rec.current_value}  ->  Suggested: {rec.suggested_value}")
        click.echo(f"Reason: {rec.reason}")

    if derive_from is not None and out_config is not None:
        if rec.suggested_value is None:
            click.echo("No suggested value available; derived config not written.")
            return
        written = config_mutator.derive_config(
            derive_from,
            {knob: rec.suggested_value},
            out_config,
        )
        click.echo(f"Wrote suggested config to {written}")


@cli.command()
@click.argument("base_config", type=click.Path(exists=True, path_type=Path))
@click.argument("out_config", type=click.Path(path_type=Path))
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Override a dotted config key, e.g. --set serving.batch_size=8 (repeatable).",
)
def derive(base_config: Path, out_config: Path, overrides: tuple[str, ...]) -> None:
    """Derive a new config from a base config with one or more overrides applied."""
    parsed_overrides: dict[str, object] = {}
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter(f"Expected key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        parsed_overrides[key.strip()] = _coerce(raw_value.strip())

    written = config_mutator.derive_config(base_config, parsed_overrides, out_config)
    click.echo(f"Wrote derived config to {written}")


def _coerce(value: str) -> object:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


if __name__ == "__main__":
    cli()
