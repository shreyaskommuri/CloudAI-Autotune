"""Command-line entry point tying the loader, runner, parser, DB, and recommender together."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import click

from autotune import config_mutator
from autotune.database import ExperimentDB
from autotune.parser import parse_report
from autotune.recommender import DEFAULT_KNOB, recommend_next
from autotune.runner import CloudAIRunner


@click.group()
def cli() -> None:
    """CloudAI Autotune — closed-loop benchmark experiment manager."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", "db_path", default="autotune.db", help="Path to the experiment database.")
@click.option("--dry-run", is_flag=True, help="Pass --dry-run through to CloudAI without executing.")
@click.option("--cloudai-bin", default="cloudai", help="Name/path of the CloudAI CLI binary.")
def run(config_path: Path, db_path: str, dry_run: bool, cloudai_bin: str) -> None:
    """Run a CloudAI scenario, parse its report, and record the experiment."""
    config = config_mutator.load_config(config_path)
    scenario = config.get("scenario", {})

    with ExperimentDB(db_path) as db:
        experiment_id = db.add_experiment(
            scenario=scenario.get("name", config_path.stem),
            backend=scenario.get("backend", "unknown"),
            config_path=str(config_path),
            config=config,
            status="running",
        )

        run_id = f"{experiment_id:04d}_{config_path.stem}_{int(time.time())}"
        runner = CloudAIRunner(cloudai_bin=cloudai_bin, dry_run=dry_run)
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
@click.argument("report_path", type=click.Path(exists=True, path_type=Path))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--db", "db_path", default="autotune.db", help="Path to the experiment database.")
def ingest(report_path: Path, config_path: Path, db_path: str) -> None:
    """Record an existing CloudAI report without launching CloudAI."""
    config = config_mutator.load_config(config_path)
    scenario = config.get("scenario", {})
    metrics = parse_report(report_path)

    with ExperimentDB(db_path) as db:
        experiment_id = db.add_experiment(
            scenario=scenario.get("name", config_path.stem),
            backend=scenario.get("backend", "unknown"),
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

    click.echo(f"[{experiment_id}] ingested {report_path} — {metrics}")


@cli.command()
@click.option("--db", "db_path", default="autotune.db")
@click.option("--scenario", default=None, help="Restrict recommendation to one scenario's history.")
@click.option("--knob", default=DEFAULT_KNOB, help="Dotted config key to tune, e.g. serving.batch_size.")
@click.option("--latency-budget-ms", type=float, default=None, help="Maximum acceptable latency in ms.")
def recommend(db_path: str, scenario: Optional[str], knob: str, latency_budget_ms: Optional[float]) -> None:
    """Recommend the next config value to try based on experiment history."""
    with ExperimentDB(db_path) as db:
        experiments = db.list_experiments(scenario=scenario)
        rec = recommend_next(experiments, knob=knob, latency_budget_ms=latency_budget_ms)
        click.echo(f"Knob: {rec.knob}")
        click.echo(f"Current: {rec.current_value}  ->  Suggested: {rec.suggested_value}")
        click.echo(f"Reason: {rec.reason}")


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
