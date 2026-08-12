"""Streamlit dashboard: browse experiments, compare metrics, view recommendations.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotune.comparison import compare_best_and_latest, compare_latest_to_previous
from autotune.database import DEFAULT_DB_PATH, ExperimentDB
from autotune.optimizer import resolve_pending_suggestions, suggest_joint_optimize
from autotune.recommender import (
    DEFAULT_KNOB,
    discover_knobs,
    format_combo,
    recommend_joint,
    recommend_next,
    suggest_joint_step,
    suggest_untried_combo,
)


def _format_number(value: object, suffix: str = "") -> str | None:
    try:
        return f"{float(value):g}{suffix}"
    except (TypeError, ValueError):
        return None


st.set_page_config(page_title="CloudAI Autotune", layout="wide")
st.title("CloudAI Autotune")
st.caption("Closed-loop benchmark experiment manager for CloudAI scenarios")

with st.sidebar:
    db_path = st.text_input("Database path", value=str(DEFAULT_DB_PATH))
    knob = st.text_input("Tunable knob (dotted config key)", value=DEFAULT_KNOB)
    latency_budget = st.number_input("Latency budget (ms, optional)", min_value=0.0, value=0.0, step=10.0)
    ttft_budget = st.number_input("TTFT budget (ms, optional)", min_value=0.0, value=0.0, step=10.0)
    min_throughput = st.number_input("Min throughput (tok/s, optional)", min_value=0.0, value=0.0, step=10.0)
    runtime_budget = st.number_input("Runtime budget (sec, optional)", min_value=0.0, value=0.0, step=10.0)
    max_failure_rate = st.number_input(
        "Max failure rate (0-1, optional)", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )

if not Path(db_path).exists():
    st.warning(f"No database found at `{db_path}` yet. Run `autotune run <config>` to create one.")
    st.stop()

with ExperimentDB(db_path) as db:
    experiments = db.list_experiments()

if not experiments:
    st.info("No experiments recorded yet.")
    st.stop()

scenarios = sorted({e.scenario for e in experiments})
selected_scenario = st.selectbox("Scenario", options=["All"] + scenarios)
filtered = experiments if selected_scenario == "All" else [e for e in experiments if e.scenario == selected_scenario]
budget = latency_budget if latency_budget > 0 else None

rows = []
for exp in filtered:
    node = exp.config
    knob_value = None
    try:
        for part in knob.split("."):
            node = node[part]
        knob_value = node
    except (KeyError, TypeError):
        pass
    rows.append(
        {
            "id": exp.id,
            "scenario": exp.scenario,
            "backend": exp.backend,
            "status": exp.status,
            knob: knob_value,
            "throughput_tokens_per_sec": exp.metrics.get("throughput_tokens_per_sec"),
            "latency_ms": exp.metrics.get("latency_ms"),
            "ttft_ms": exp.metrics.get("ttft_ms"),
            "runtime_sec": exp.metrics.get("runtime_sec"),
            "failure_rate": exp.metrics.get("failure_rate"),
            "created_at": exp.created_at,
        }
    )

df = pd.DataFrame(rows)

st.subheader("Experiments")
st.dataframe(df, use_container_width=True, hide_index=True)

comparison = compare_best_and_latest(filtered, latency_budget_ms=budget)
st.subheader("Comparison")
col1, col2, col3 = st.columns(3)
with col1:
    best_value = comparison.best.metrics.get("throughput_tokens_per_sec") if comparison.best is not None else None
    st.metric(
        label="Best throughput run",
        value=f"#{comparison.best.id}" if comparison.best is not None else "n/a",
        delta=_format_number(best_value, " tok/s"),
    )
with col2:
    latest_value = comparison.latest.metrics.get("throughput_tokens_per_sec") if comparison.latest is not None else None
    st.metric(
        label="Latest completed run",
        value=f"#{comparison.latest.id}" if comparison.latest is not None else "n/a",
        delta=_format_number(latest_value, " tok/s"),
    )
with col3:
    st.metric(
        label="Latest vs. best",
        value=(f"{comparison.throughput_delta_pct:+.1f}%" if comparison.throughput_delta_pct is not None else "n/a"),
        delta=(f"{comparison.latency_delta_ms:+.1f} ms latency" if comparison.latency_delta_ms is not None else None),
    )

if comparison.best is not None and comparison.latest is not None:
    if comparison.best.id == comparison.latest.id:
        st.success(f"Latest completed run #{comparison.latest.id} is also the best throughput run.")
    else:
        parts = [f"Latest completed run #{comparison.latest.id} differs from best run #{comparison.best.id}."]
        if comparison.throughput_delta_pct is not None:
            direction = "higher" if comparison.throughput_delta_pct > 0 else "lower"
            parts.append(f"Throughput is {abs(comparison.throughput_delta_pct):.1f}% {direction}.")
        if comparison.latency_delta_ms is not None:
            direction = "higher" if comparison.latency_delta_ms > 0 else "lower"
            parts.append(f"Latency is {abs(comparison.latency_delta_ms):.1f} ms {direction}.")
        st.warning(" ".join(parts))

regression = compare_latest_to_previous(filtered)
if regression.latest is not None and regression.previous is not None:
    parts = [f"Latest completed run #{regression.latest.id} vs. immediately preceding run #{regression.previous.id}."]
    if regression.throughput_delta_pct is not None:
        direction = "higher" if regression.throughput_delta_pct > 0 else "lower"
        parts.append(f"Throughput is {abs(regression.throughput_delta_pct):.1f}% {direction}.")
    if regression.latency_delta_ms is not None:
        direction = "higher" if regression.latency_delta_ms > 0 else "lower"
        parts.append(f"Latency is {abs(regression.latency_delta_ms):.1f} ms {direction}.")
    message = " ".join(parts)
    if regression.regressed:
        st.error(f"Regression vs. previous run: {message}")
    else:
        st.success(f"No regression vs. previous run: {message}")

completed = df[df["status"] == "completed"].dropna(subset=[knob, "throughput_tokens_per_sec", "latency_ms"])
if not completed.empty:
    completed = completed.sort_values(knob)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Throughput vs. knob value")
        st.line_chart(completed.set_index(knob)["throughput_tokens_per_sec"])
    with col2:
        st.subheader("Latency vs. knob value")
        st.line_chart(completed.set_index(knob)["latency_ms"])

st.subheader("Recommendation")
scenario_filter = None if selected_scenario == "All" else selected_scenario
with ExperimentDB(db_path) as db:
    rec = recommend_next(
        db.list_experiments(scenario=scenario_filter),
        knob=knob,
        latency_budget_ms=budget,
        ttft_budget_ms=ttft_budget if ttft_budget > 0 else None,
        min_throughput_tokens_per_sec=min_throughput if min_throughput > 0 else None,
        runtime_budget_sec=runtime_budget if runtime_budget > 0 else None,
        max_failure_rate=max_failure_rate if max_failure_rate > 0 else None,
    )

st.metric(label=f"Suggested next value for `{rec.knob}`", value=str(rec.suggested_value), delta=str(rec.current_value))
st.write(rec.reason)

st.subheader("Multi-knob recommendation")
available_knobs = discover_knobs(filtered)
selected_knobs = st.multiselect("Knobs to consider jointly", options=available_knobs)

joint_budgets = {
    "latency_budget_ms": budget,
    "ttft_budget_ms": ttft_budget if ttft_budget > 0 else None,
    "min_throughput_tokens_per_sec": min_throughput if min_throughput > 0 else None,
    "runtime_budget_sec": runtime_budget if runtime_budget > 0 else None,
    "max_failure_rate": max_failure_rate if max_failure_rate > 0 else None,
}

if len(selected_knobs) < 2:
    st.caption("Select at least two knobs above to see a joint recommendation.")
else:
    joint = recommend_joint(filtered, knobs=selected_knobs, **joint_budgets)
    st.write(joint.reason)

    if joint.best is not None:
        st.metric(
            label=f"Best combo: {format_combo(joint.best.values)}",
            value=f"{joint.best.throughput:g} tok/s",
            delta=f"{joint.best.latency:g} ms latency",
        )
        frontier_rows = [
            {
                **combo.values,
                "throughput_tokens_per_sec": combo.throughput,
                "latency_ms": combo.latency,
                "efficiency": combo.efficiency,
            }
            for combo in joint.frontier
        ]
        st.write("Pareto frontier (not beaten on both throughput and latency by another combo):")
        st.dataframe(pd.DataFrame(frontier_rows), width="stretch", hide_index=True)

    with ExperimentDB(db_path) as db:
        pending = db.list_suggestions(selected_knobs)
        resolutions = resolve_pending_suggestions(filtered, selected_knobs, pending)
        for suggestion_id, experiment_id, actual_efficiency in resolutions:
            db.resolve_suggestion(suggestion_id, experiment_id, actual_efficiency)
        if resolutions:
            pending = db.list_suggestions(selected_knobs)

    mode = st.radio("Suggest an untried combination via", ["Explore", "Search", "Optimize"], horizontal=True)

    if mode == "Explore":
        explore_rec = suggest_untried_combo(filtered, knobs=selected_knobs, **joint_budgets)
        st.write(explore_rec.reason)
        if explore_rec.suggested is not None:
            st.write(f"Suggested untried combo: {format_combo(explore_rec.suggested)}")
    elif mode == "Search":
        search_rec = suggest_joint_step(filtered, knobs=selected_knobs, **joint_budgets)
        st.write(search_rec.reason)
        if search_rec.suggested is not None:
            st.write(f"Suggested untried combo: {format_combo(search_rec.suggested)}")
    else:
        optimize_rec = suggest_joint_optimize(
            filtered, knobs=selected_knobs, pending_suggestions=pending, **joint_budgets
        )
        st.write(optimize_rec.reason)
        if optimize_rec.suggested is not None:
            st.write(f"Suggested untried combo: {format_combo(optimize_rec.suggested)}")
            if optimize_rec.predicted_efficiency is not None:
                st.write(f"Predicted efficiency: {optimize_rec.predicted_efficiency:.4f} tok/s per ms")
            if st.button("Record this suggestion"):
                with ExperimentDB(db_path) as db:
                    db.add_suggestion(selected_knobs, optimize_rec.suggested, optimize_rec.predicted_efficiency)
                st.success("Recorded. It won't be suggested again until it's actually run and ingested.")

    # Re-fetch rather than reuse `pending`: a button click just above may have recorded a new
    # suggestion this same run, and this table should reflect that immediately, not next rerun.
    st.write("Suggestion history for this knob set:")
    with ExperimentDB(db_path) as db:
        history = db.list_suggestions(selected_knobs)
    if not history:
        st.caption("No suggestions recorded yet. Use Optimize mode above and click Record to start tracking one.")
    else:
        history_rows = [
            {
                **suggestion.suggested_values,
                "predicted_efficiency": suggestion.predicted_efficiency,
                "status": "Resolved" if suggestion.resolved_experiment_id is not None else "Pending",
                "actual_efficiency": suggestion.actual_efficiency,
                "created_at": suggestion.created_at,
            }
            for suggestion in history
        ]
        st.dataframe(pd.DataFrame(history_rows), width="stretch", hide_index=True)

dse_experiments = [e for e in filtered if e.backend == "cloudai-dse"]
if dse_experiments:
    st.subheader("DSE sweeps")
    st.caption("Sweeps CloudAI already ran (grid search over swept parameters), ingested via `autotune ingest-dse`.")

    dse_options = {f"#{e.id} {e.scenario} ({e.metrics.get('num_trials', '?')} trials)": e.id for e in dse_experiments}
    selected_label = st.selectbox("Sweep", options=list(dse_options))
    selected_id = dse_options[selected_label]

    with ExperimentDB(db_path) as db:
        trials = db.get_dse_trials(selected_id)

    if trials:
        trials_df = pd.DataFrame({"step": [t.step for t in trials], "reward": [t.reward for t in trials]})
        st.line_chart(trials_df.set_index("step")["reward"])

        best = max(trials, key=lambda t: t.reward)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Best reward", value=f"{best.reward:g}", delta=f"step {best.step}")
        with col2:
            st.write("Best action")
            st.json(best.action)
    else:
        st.info("No trials stored for this sweep.")
