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
from autotune.budgets import BudgetCheck, Budgets, evaluate_experiment
from autotune.comparison import RegressionCheck, compare_latest_to_previous
from autotune.database import Experiment, ExperimentDB
from autotune.diffing import FieldDiff, diff_experiments
from autotune.dse import best_trial, derive_test_name, find_trajectory_files, parse_trajectory
from autotune.parser import parse_report
from autotune.recommender import (
    DEFAULT_KNOB,
    DEFAULT_SEARCH_NEIGHBORHOOD_CAP,
    discover_knobs,
    format_combo,
    recommend_joint,
    recommend_next,
    suggest_joint_step,
    suggest_untried_combo,
)
from autotune.runner import CloudAIRunner, RunResult

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
@click.option(
    "--timeout-sec",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Fail and log the run if CloudAI exceeds this many seconds.",
)
@click.option("--system-config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--tests-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
@click.option("--hook-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
@click.option("--notes", default=None, help="Intent or context to store with this experiment.")
@click.option(
    "--metadata",
    "metadata_items",
    multiple=True,
    help="Experiment metadata as key=value, e.g. --metadata hardware.gpu=A100.",
)
def run(
    config_path: Path,
    db_path: str,
    dry_run: bool,
    cloudai_bin: str,
    timeout_sec: Optional[float],
    system_config: Optional[Path],
    tests_dir: Optional[Path],
    hook_dir: Optional[Path],
    notes: Optional[str],
    metadata_items: tuple[str, ...],
) -> None:
    """Run a CloudAI scenario, parse its report, and record the experiment."""
    config = config_mutator.load_config(config_path)
    metadata = _parse_metadata(metadata_items)
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
            metadata=metadata,
        )

        run_id = f"{experiment_id:04d}_{config_path.stem}_{int(time.time())}"
        runner = CloudAIRunner(
            cloudai_bin=cloudai_bin,
            dry_run=dry_run,
            system_config=system_config,
            tests_dir=tests_dir,
            hook_dir=hook_dir,
            timeout_sec=timeout_sec,
        )
        result = runner.run(config_path, run_id)

        if not result.succeeded:
            db.update_result(experiment_id, status="failed", report_path=str(result.stdout_path), notes=notes)
            reason = result.failure_reason or f"CloudAI failed with code {result.returncode}"
            click.echo(f"[{experiment_id}] {reason}. See {result.stdout_path}")
            raise click.exceptions.Exit(1)

        report_source = result.report_path or result.stdout_path
        metrics, parse_error = _parse_cloudai_report(result, report_source)
        if parse_error is not None:
            db.update_result(
                experiment_id,
                status="failed",
                report_path=str(result.stdout_path),
                notes=notes,
            )
            click.echo(f"[{experiment_id}] {parse_error}. See {result.stdout_path}")
            raise click.exceptions.Exit(1)
        db.update_result(
            experiment_id,
            status="completed",
            report_path=str(report_source),
            metrics=metrics,
            notes=notes,
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
            line = f"[{exp.id}] {exp.scenario} ({exp.backend}) status={exp.status} metrics={exp.metrics}"
            if exp.notes:
                line += f" notes={exp.notes}"
            if exp.metadata:
                line += f" metadata={exp.metadata}"
            click.echo(line)


@cli.command(name="diff")
@click.argument("left_id", type=int)
@click.argument("right_id", type=int)
@click.option("--db", "db_path", default="autotune.db")
def diff_experiment(db_path: str, left_id: int, right_id: int) -> None:
    """Compare config and metric differences between two experiments."""
    with ExperimentDB(db_path) as db:
        left = db.get(left_id)
        right = db.get(right_id)

    if left is None:
        raise click.ClickException(f"Experiment {left_id} not found.")
    if right is None:
        raise click.ClickException(f"Experiment {right_id} not found.")

    diff = diff_experiments(left, right)
    click.echo(f"Comparing [{left.id}] {left.scenario} -> [{right.id}] {right.scenario}")
    click.echo("Config:")
    _echo_diffs(diff.config)
    click.echo("Metrics:")
    _echo_diffs(diff.metrics)


@cli.command()
@click.option("--db", "db_path", default="autotune.db")
@click.option("--scenario", default=None, help="Restrict export to one scenario.")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(("csv", "json", "markdown")),
    default="csv",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--template",
    type=click.Choice(("table", "issue", "pr")),
    default="table",
    show_default=True,
    help="Markdown template to use with --format markdown.",
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
    template: str,
    out_path: Optional[Path],
) -> None:
    """Export experiment history as CSV, JSON, or Markdown."""
    if template != "table" and export_format != "markdown":
        raise click.UsageError("--template can only be used with --format markdown.")

    with ExperimentDB(db_path) as db:
        experiments = db.list_experiments(scenario=scenario)
        rows = [_experiment_row(exp) for exp in experiments]

    if export_format == "json":
        output = json.dumps(rows, indent=2) + "\n"
    elif export_format == "markdown":
        output = _rows_to_markdown(rows, template=template, experiments=experiments)
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
    "--ttft-budget-ms",
    type=float,
    default=None,
    help="Maximum acceptable time to first token in ms.",
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
    ttft_budget_ms: Optional[float],
    runtime_budget_sec: Optional[float],
    max_failure_rate: Optional[float],
    strict: bool,
) -> None:
    """Check recorded experiments against metric budgets."""
    budgets = Budgets(
        latency_ms=latency_budget_ms,
        ttft_ms=ttft_budget_ms,
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
    for exp, check_result in zip(experiments, checks, strict=True):
        reason = "; ".join(check_result.reasons)
        click.echo(f"[{exp.id}] {exp.scenario} status={check_result.status} - {reason}")
    click.echo(_check_summary(checks))

    if strict and any(check_result.status != "pass" for check_result in checks):
        raise click.exceptions.Exit(1)


def _check_summary(checks: list[BudgetCheck]) -> str:
    counts = {"pass": 0, "fail": 0, "unknown": 0}
    for check_result in checks:
        status = check_result.status
        counts[status if status in counts else "unknown"] += 1
    total = sum(counts.values())
    return f"Summary: {counts['pass']} pass, {counts['fail']} fail, {counts['unknown']} unknown ({total} total)"


def _echo_diffs(diffs: tuple[FieldDiff, ...]) -> None:
    if not diffs:
        click.echo("  no differences")
        return
    for item in diffs:
        click.echo(f"  {item.key}: {_format_diff_value(item.left)} -> {_format_diff_value(item.right)}")


def _format_diff_value(value: object) -> str:
    return "missing" if value is None else str(value)


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
    for key, value in exp.metadata.items():
        row[f"metadata.{key}"] = value
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
    metadata_keys = sorted({key for row in rows for key in row if key.startswith("metadata.")})
    return base + metric_keys + metadata_keys


def _rows_to_markdown(
    rows: list[dict[str, object]],
    template: str = "table",
    experiments: Optional[list[Experiment]] = None,
) -> str:
    if template == "issue":
        return _rows_to_issue_markdown(rows)
    if template == "pr":
        return _rows_to_pr_markdown(rows, experiments or [])
    fieldnames = _export_fieldnames(rows)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(key)) for key in fieldnames) + " |")
    return "\n".join(lines) + "\n"


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rows_to_issue_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "## Benchmark Result Summary",
        "",
        f"Recorded experiments: {len(rows)}",
        "",
        "### Results",
        "",
        _rows_to_markdown(rows, template="table").rstrip(),
        "",
        "### Notes",
        "",
        "- Review pass/fail budgets with `autotune check`.",
        "- Compare config and metric changes with `autotune diff <left> <right>`.",
        "",
    ]
    return "\n".join(lines)


def _regression_summary(regression: RegressionCheck) -> str:
    if regression.latest is None:
        return "No completed experiments to check."
    if regression.previous is None:
        return f"No prior completed run to compare #{regression.latest.id} against yet."

    parts = [f"Run #{regression.latest.id} vs. immediately preceding run #{regression.previous.id}."]
    if regression.throughput_delta_pct is not None:
        direction = "higher" if regression.throughput_delta_pct > 0 else "lower"
        parts.append(f"Throughput is {abs(regression.throughput_delta_pct):.1f}% {direction}.")
    if regression.latency_delta_ms is not None:
        direction = "higher" if regression.latency_delta_ms > 0 else "lower"
        parts.append(f"Latency is {abs(regression.latency_delta_ms):.1f} ms {direction}.")
    summary = " ".join(parts)

    return f"REGRESSED: {summary}" if regression.regressed else f"No regression: {summary}"


def _rows_to_pr_markdown(rows: list[dict[str, object]], experiments: list[Experiment]) -> str:
    lines = [
        "## Benchmark Evidence",
        "",
        "### What Was Tested",
        "",
        _rows_to_markdown(rows, template="table").rstrip(),
        "",
        "### Regression Check",
        "",
        _regression_summary(compare_latest_to_previous(experiments)),
        "",
        "### Validation",
        "",
        "- Export generated by `autotune export --format markdown --template pr`.",
        "- Include `autotune check` output here when this PR changes performance-sensitive code.",
        "",
    ]
    return "\n".join(lines)


@cli.command()
@click.argument("report_path", type=click.Path(exists=True, path_type=Path))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--scenario", default=None, help="Scenario name to use when --config is omitted.")
@click.option("--backend", default=None, help="Backend name to use when --config is omitted.")
@click.option("--db", "db_path", default="autotune.db", help="Path to the experiment database.")
@click.option("--notes", default=None, help="Intent or context to store with this experiment.")
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Config metadata as dotted key=value, e.g. --set serving.batch_size=4.",
)
@click.option(
    "--metadata",
    "metadata_items",
    multiple=True,
    help="Experiment metadata as key=value, e.g. --metadata hardware.gpu=A100.",
)
def ingest(
    report_path: Path,
    config_path: Optional[Path],
    scenario: Optional[str],
    backend: Optional[str],
    db_path: str,
    notes: Optional[str],
    overrides: tuple[str, ...],
    metadata_items: tuple[str, ...],
) -> None:
    """Record an existing CloudAI report without launching CloudAI."""
    experiment_id, metrics = _ingest_report(
        report_path,
        config_path,
        db_path,
        notes=notes,
        scenario=scenario,
        backend=backend,
        overrides=_parse_assignments(overrides, param_hint="--set"),
        metadata=_parse_metadata(metadata_items),
    )

    click.echo(f"[{experiment_id}] ingested {report_path} — {metrics}")


@cli.command(name="ingest-dse")
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--db", "db_path", default="autotune.db", help="Path to the experiment database.")
@click.option(
    "--scenario",
    default=None,
    help="Scenario name to record every ingested sweep under; defaults to each CloudAI test name.",
)
def ingest_dse(results_dir: Path, db_path: str, scenario: Optional[str]) -> None:
    """Ingest CloudAI DSE trajectory.csv files found under a results directory.

    Each trajectory.csv (one per CloudAI test/iteration) becomes one experiment row plus its
    per-trial rows, so the sweep CloudAI already ran can be browsed and charted in Autotune
    without re-running or re-implementing the search.
    """
    trajectory_paths = find_trajectory_files(results_dir)
    if not trajectory_paths:
        click.echo(f"No trajectory.csv files found under {results_dir}.")
        return

    with ExperimentDB(db_path) as db:
        for trajectory_path in trajectory_paths:
            trials = parse_trajectory(trajectory_path)
            if not trials:
                continue

            test_name = derive_test_name(trajectory_path)
            experiment_id = db.add_experiment(
                scenario=scenario or test_name,
                backend="cloudai-dse",
                config_path=str(trajectory_path),
                config={},
                status="completed",
            )
            db.add_dse_trials(experiment_id, trials)

            best = best_trial(trials)
            metrics = {"best_reward": best.reward, "best_step": best.step, "num_trials": len(trials)} if best else {}
            db.update_result(experiment_id, status="completed", report_path=str(trajectory_path), metrics=metrics)

            best_desc = f"best reward={best.reward:g} at step {best.step}" if best else "no trials"
            click.echo(f"[{experiment_id}] ingested {len(trials)} trials from {trajectory_path} ({best_desc})")


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


def _ingest_report(
    report_path: Path,
    config_path: Optional[Path],
    db_path: str,
    notes: Optional[str] = None,
    scenario: Optional[str] = None,
    backend: Optional[str] = None,
    overrides: Optional[dict[str, object]] = None,
    metadata: Optional[dict[str, object]] = None,
) -> tuple[int, dict[str, object]]:
    config = _ingest_config(report_path, config_path, scenario=scenario, backend=backend)
    for key, value in (overrides or {}).items():
        config = config_mutator.set_value(config, key, value)
    scenario = config.get("scenario", {})
    scenario_name = scenario.get("name") if isinstance(scenario, dict) else None
    backend = scenario.get("backend") if isinstance(scenario, dict) else None
    metrics = parse_report(report_path)

    with ExperimentDB(db_path) as db:
        experiment_id = db.add_experiment(
            scenario=scenario_name or config.get("name", report_path.stem),
            backend=backend or "unknown",
            config_path=str(config_path or report_path),
            config=config,
            status="completed",
            metadata=metadata,
        )
        db.update_result(
            experiment_id,
            status="completed",
            report_path=str(report_path),
            metrics=metrics,
            notes=notes,
        )

    return experiment_id, metrics


def _ingest_config(
    report_path: Path,
    config_path: Optional[Path],
    scenario: Optional[str],
    backend: Optional[str],
) -> dict[str, object]:
    if config_path is not None:
        return config_mutator.load_config(config_path)
    return {
        "scenario": {
            "name": scenario or report_path.stem,
            "backend": backend or "unknown",
        }
    }


def _echo_joint_recommendation(
    experiments: list[Experiment],
    knobs: list[str],
    explore: bool,
    search: bool,
    max_candidates: int,
    latency_budget_ms: Optional[float],
    ttft_budget_ms: Optional[float],
    min_throughput_tokens_per_sec: Optional[float],
    runtime_budget_sec: Optional[float],
    max_failure_rate: Optional[float],
    derive_from: Optional[Path],
    out_config: Optional[Path],
) -> None:
    joint_rec = recommend_joint(
        experiments,
        knobs=knobs,
        latency_budget_ms=latency_budget_ms,
        ttft_budget_ms=ttft_budget_ms,
        min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
        runtime_budget_sec=runtime_budget_sec,
        max_failure_rate=max_failure_rate,
    )
    click.echo(f"Knobs: {', '.join(joint_rec.knobs)}")
    click.echo(f"Reason: {joint_rec.reason}")
    if joint_rec.best is not None:
        best_desc = format_combo(joint_rec.best.values)
        click.echo(f"Best combo: {best_desc} ({joint_rec.best.throughput:.0f} tok/s @ {joint_rec.best.latency:.0f}ms)")
        for combo in joint_rec.frontier:
            if combo is joint_rec.best:
                continue
            combo_desc = format_combo(combo.values)
            click.echo(f"Frontier: {combo_desc} ({combo.throughput:.0f} tok/s @ {combo.latency:.0f}ms)")

    combo_to_write = joint_rec.best.values if joint_rec.best is not None else None

    if explore:
        explore_rec = suggest_untried_combo(
            experiments,
            knobs=knobs,
            latency_budget_ms=latency_budget_ms,
            ttft_budget_ms=ttft_budget_ms,
            min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
            runtime_budget_sec=runtime_budget_sec,
            max_failure_rate=max_failure_rate,
        )
        click.echo(f"Explore reason: {explore_rec.reason}")
        if explore_rec.suggested is not None:
            click.echo(f"Suggested untried combo: {format_combo(explore_rec.suggested)}")
            combo_to_write = explore_rec.suggested

    if search:
        search_rec = suggest_joint_step(
            experiments,
            knobs=knobs,
            latency_budget_ms=latency_budget_ms,
            ttft_budget_ms=ttft_budget_ms,
            min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
            runtime_budget_sec=runtime_budget_sec,
            max_failure_rate=max_failure_rate,
            max_candidates=max_candidates,
        )
        click.echo(f"Search reason: {search_rec.reason}")
        if search_rec.suggested is not None:
            click.echo(f"Suggested untried combo: {format_combo(search_rec.suggested)}")
            combo_to_write = search_rec.suggested

    if derive_from is not None and out_config is not None:
        if combo_to_write is None:
            click.echo("No combo available; derived config not written.")
            return
        written = config_mutator.derive_config(derive_from, combo_to_write, out_config)
        click.echo(f"Wrote suggested config to {written}")


@cli.command()
@click.option("--db", "db_path", default="autotune.db")
@click.option("--scenario", default=None, help="Restrict recommendation to one scenario's history.")
@click.option(
    "--knob",
    "knobs",
    multiple=True,
    help="Dotted config key to tune; repeat for multi-knob recommendations.",
)
@click.option(
    "--all-knobs",
    is_flag=True,
    help="Recommend for every numeric config key seen across completed runs, instead of --knob.",
)
@click.option(
    "--joint",
    is_flag=True,
    help="Treat two or more --knob values as one combination instead of independent recommendations.",
)
@click.option(
    "--explore",
    is_flag=True,
    help="With --joint, also suggest one untried combination extending the best combo found.",
)
@click.option(
    "--search",
    is_flag=True,
    help=(
        "With --joint, suggest one untried combination that may change two or more knobs at "
        "once, instead of --explore's one-knob-at-a-time step."
    ),
)
@click.option(
    "--max-candidates",
    type=int,
    default=DEFAULT_SEARCH_NEIGHBORHOOD_CAP,
    show_default=True,
    help="With --search, cap on how many untried combos in the local neighborhood to consider.",
)
@click.option("--latency-budget-ms", type=float, default=None, help="Maximum acceptable latency in ms.")
@click.option(
    "--ttft-budget-ms",
    type=float,
    default=None,
    help="Maximum acceptable time to first token in ms.",
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
    knobs: tuple[str, ...],
    all_knobs: bool,
    joint: bool,
    explore: bool,
    search: bool,
    max_candidates: int,
    latency_budget_ms: Optional[float],
    ttft_budget_ms: Optional[float],
    min_throughput_tokens_per_sec: Optional[float],
    runtime_budget_sec: Optional[float],
    max_failure_rate: Optional[float],
    derive_from: Optional[Path],
    out_config: Optional[Path],
) -> None:
    """Recommend the next config value to try based on experiment history."""
    if (derive_from is None) != (out_config is None):
        raise click.UsageError("--derive-from and --out-config must be provided together.")
    if all_knobs and knobs:
        raise click.UsageError("--all-knobs and --knob cannot be used together.")
    if joint and all_knobs:
        raise click.UsageError("--joint and --all-knobs cannot be used together.")
    if joint and len(knobs) < 2:
        raise click.UsageError("--joint needs at least two --knob values.")
    if explore and not joint:
        raise click.UsageError("--explore requires --joint.")
    if search and not joint:
        raise click.UsageError("--search requires --joint.")
    if search and explore:
        raise click.UsageError("--search and --explore cannot be used together.")

    with ExperimentDB(db_path) as db:
        experiments = db.list_experiments(scenario=scenario)

    if all_knobs:
        knobs = tuple(discover_knobs(experiments))
        if not knobs:
            click.echo("No completed experiments with numeric config values found.")
            return
    else:
        knobs = knobs or (DEFAULT_KNOB,)

    if joint:
        _echo_joint_recommendation(
            experiments,
            knobs=list(knobs),
            explore=explore,
            search=search,
            max_candidates=max_candidates,
            latency_budget_ms=latency_budget_ms,
            ttft_budget_ms=ttft_budget_ms,
            min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
            runtime_budget_sec=runtime_budget_sec,
            max_failure_rate=max_failure_rate,
            derive_from=derive_from,
            out_config=out_config,
        )
        return

    recommendations = [
        recommend_next(
            experiments,
            knob=knob,
            latency_budget_ms=latency_budget_ms,
            ttft_budget_ms=ttft_budget_ms,
            min_throughput_tokens_per_sec=min_throughput_tokens_per_sec,
            runtime_budget_sec=runtime_budget_sec,
            max_failure_rate=max_failure_rate,
        )
        for knob in knobs
    ]
    for rec in recommendations:
        click.echo(f"Knob: {rec.knob}")
        click.echo(f"Current: {rec.current_value}  ->  Suggested: {rec.suggested_value}")
        click.echo(f"Reason: {rec.reason}")

    if derive_from is not None and out_config is not None:
        overrides = {rec.knob: rec.suggested_value for rec in recommendations if rec.suggested_value is not None}
        if not overrides:
            click.echo("No suggested value available; derived config not written.")
            return
        written = config_mutator.derive_config(
            derive_from,
            overrides,
            out_config,
        )
        click.echo(f"Wrote suggested config to {written}")


@cli.command(name="smoke-cloudai")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--cloudai-bin", default="cloudai", help="Name/path of the CloudAI CLI binary.")
@click.option("--system-config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--tests-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
@click.option("--hook-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
@click.option("--runs-dir", type=click.Path(path_type=Path), default=Path("runs"))
@click.option("--timeout-sec", type=int, default=60, show_default=True)
def smoke_cloudai(
    config_path: Path,
    cloudai_bin: str,
    system_config: Optional[Path],
    tests_dir: Optional[Path],
    hook_dir: Optional[Path],
    runs_dir: Path,
    timeout_sec: int,
) -> None:
    """Dry-run CloudAI through Autotune and report whether the CLI contract works."""
    runner = CloudAIRunner(
        cloudai_bin=cloudai_bin,
        dry_run=True,
        system_config=system_config,
        tests_dir=tests_dir,
        hook_dir=hook_dir,
        runs_dir=runs_dir,
        timeout_sec=timeout_sec,
    )
    run_id = f"smoke_{config_path.stem}_{int(time.time())}"
    result = runner.run(config_path, run_id)
    click.echo(f"Command log: {result.stdout_path}")
    if not result.succeeded:
        reason = result.failure_reason or f"CloudAI failed with code {result.returncode}"
        click.echo(f"CloudAI smoke failed: {reason}.")
        raise click.exceptions.Exit(1)
    if result.report_path is not None:
        click.echo(f"Detected report: {result.report_path}")
        metrics, parse_error = _parse_cloudai_report(result, result.report_path)
        if parse_error is not None:
            click.echo(f"CloudAI smoke failed: {parse_error}.")
            raise click.exceptions.Exit(1)
        click.echo(f"Parsed metrics: {metrics}")
    else:
        click.echo("CloudAI smoke passed; no machine-readable report was detected.")


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


def _parse_metadata(items: tuple[str, ...]) -> dict[str, object]:
    return _parse_assignments(items, param_hint="--metadata")


def _parse_assignments(items: tuple[str, ...], param_hint: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"Expected key=value, got: {item}", param_hint=param_hint)
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter("Key cannot be empty.", param_hint=param_hint)
        metadata[key] = _coerce(raw_value.strip())
    return metadata


def _parse_cloudai_report(
    result: RunResult,
    report_path: Path,
) -> tuple[Optional[dict[str, Optional[float]]], Optional[str]]:
    try:
        metrics = parse_report(report_path)
    except (OSError, UnicodeError) as exc:
        reason = f"CloudAI report parsing failed: {exc}"
        result.log_diagnostic(reason)
        return None, reason
    if not any(value is not None for value in metrics.values()):
        reason = "CloudAI report contained no recognized metrics"
        result.log_diagnostic(reason)
        return None, reason
    return metrics, None


if __name__ == "__main__":
    cli()
