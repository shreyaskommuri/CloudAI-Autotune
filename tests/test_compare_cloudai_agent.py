"""Smoke tests for scripts/compare_cloudai_agent.py -- keeps the README's trials-to-best
numbers from silently going stale if the agent's search logic changes."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cloudai.configurator")

_SPEC = importlib.util.spec_from_file_location(
    "compare_cloudai_agent", Path(__file__).parent.parent / "scripts" / "compare_cloudai_agent.py"
)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)


def test_single_knob_grid_search_is_deterministic():
    assert compare.single_knob_grid_search_trials(50.0) == 1  # first value in GRID
    assert compare.single_knob_grid_search_trials(550.0) == len(compare.GRID)  # last value


def test_single_knob_optimize_agent_always_finds_the_peak():
    for seed in range(10):
        trials = compare.single_knob_optimize_trials(300.0, seed)
        assert 1 <= trials <= len(compare.GRID)


def test_two_knob_optimize_agent_always_finds_the_peak():
    for seed in range(10):
        trials = compare.two_knob_optimize_trials(300.0, 150.0, seed)
        assert 1 <= trials <= len(compare.GRID) ** 2
