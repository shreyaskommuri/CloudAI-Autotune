# CloudAI Autotune: Closed-Loop Benchmark Optimization for LLM Infrastructure

CloudAI Autotune is a closed-loop benchmarking agent that helps AI infra teams
find better LLM serving configurations. It runs [NVIDIA CloudAI](https://github.com/NVIDIA/cloudai)
scenarios, parses the results, tracks every experiment in SQLite, compares
configs over time, and recommends the next config to try.

It is **not** a benchmark engine — it's a control layer on top of CloudAI.

```
Run 1: batch_size = 1   ->  120 tok/s,  90 ms latency
Run 2: batch_size = 4   ->  330 tok/s, 160 ms latency

Recommendation: try batch_size = 8 — throughput is still scaling
faster than latency is degrading.
```

## How it works

1. **Config loader** (`autotune/config_mutator.py`) — loads CloudAI TOML
   scenario configs and derives new ones by overriding dotted keys
   (e.g. `serving.batch_size`).
2. **Runner** (`autotune/runner.py`) — wraps the `cloudai` CLI to run (or
   dry-run) a scenario, capturing stdout and any generated report under
   `runs/<run_id>/`.
3. **Parser** (`autotune/parser.py`) — normalizes CloudAI JSON reports or
   plain-text logs into `latency_ms`, `throughput_tokens_per_sec`,
   `runtime_sec`, and `failure_rate`.
4. **Database** (`autotune/database.py`) — records every run (config,
   status, metrics, report path) in SQLite.
5. **Recommender** (`autotune/recommender.py`) — compares the
   throughput/latency tradeoff across runs and suggests the next value to
   try for a tunable knob.
6. **Dashboard** (`dashboard/app.py`) — a Streamlit UI for browsing
   experiments, charting metrics, and viewing recommendations.

## Project layout

```
cloudai-autotune/
  README.md
  configs/examples/      # sample CloudAI scenario configs (vLLM, SGLang)
  autotune/
    runner.py            # CloudAI CLI wrapper
    parser.py            # report/log -> normalized metrics
    database.py          # SQLite experiment store
    recommender.py       # next-config suggestion heuristic
    config_mutator.py    # load / derive / write TOML configs
    cli.py               # `autotune` command-line entry point
  dashboard/app.py       # Streamlit dashboard
  runs/                  # captured run artifacts (stdout, reports)
  reports/               # exported/aggregated reports
  tests/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `autotune` CLI and the Streamlit dashboard dependencies.
CloudAI itself must be installed separately and available on `PATH` as
`cloudai` (or pass `--cloudai-bin /path/to/cloudai`).

## Usage

Run a scenario and record the result:

```bash
autotune run configs/examples/vllm_baseline.toml
```

Derive a new config by overriding a knob:

```bash
autotune derive configs/examples/vllm_baseline.toml configs/derived/batch8.toml \
  --set serving.batch_size=8
autotune run configs/derived/batch8.toml
```

List recorded experiments:

```bash
autotune list
```

Get a recommendation for the next value to try:

```bash
autotune recommend --knob serving.batch_size --latency-budget-ms 200
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Development

```bash
pip install -e .
pytest
```
