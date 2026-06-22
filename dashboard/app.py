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

from autotune.comparison import compare_best_and_latest
from autotune.database import DEFAULT_DB_PATH, ExperimentDB
from autotune.recommender import DEFAULT_KNOB, recommend_next


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
    best_value = (
        comparison.best.metrics.get("throughput_tokens_per_sec")
        if comparison.best is not None
        else None
    )
    st.metric(
        label="Best throughput run",
        value=f"#{comparison.best.id}" if comparison.best is not None else "n/a",
        delta=_format_number(best_value, " tok/s"),
    )
with col2:
    latest_value = (
        comparison.latest.metrics.get("throughput_tokens_per_sec")
        if comparison.latest is not None
        else None
    )
    st.metric(
        label="Latest completed run",
        value=f"#{comparison.latest.id}" if comparison.latest is not None else "n/a",
        delta=_format_number(latest_value, " tok/s"),
    )
with col3:
    st.metric(
        label="Latest vs. best",
        value=(
            f"{comparison.throughput_delta_pct:+.1f}%"
            if comparison.throughput_delta_pct is not None
            else "n/a"
        ),
        delta=(
            f"{comparison.latency_delta_ms:+.1f} ms latency"
            if comparison.latency_delta_ms is not None
            else None
        ),
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
    rec = recommend_next(db.list_experiments(scenario=scenario_filter), knob=knob, latency_budget_ms=budget)

st.metric(label=f"Suggested next value for `{rec.knob}`", value=str(rec.suggested_value), delta=str(rec.current_value))
st.write(rec.reason)
