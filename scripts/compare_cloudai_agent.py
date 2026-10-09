"""Compare AutotuneOptimizeAgent against a GridSearchAgent-equivalent on trials-to-best.

A real GPU benchmark comparison isn't possible on this machine: DynamoMocker needs
`ai-dynamo-runtime`, which ships Linux-only wheels (no macOS build at any version). This
compares the two strategies against a known, noise-free, single-peaked reward curve instead —
fed directly, bypassing CloudAI's runner entirely. Uses the real AutotuneOptimizeAgent class
(not a reimplementation), driven via a mocked CloudAIGymEnv, same pattern as
tests/test_cloudai_agent.py.

This is the easiest possible case for the smart agent: no noise, one clean peak. A real
workload's reward surface will be rougher, so treat this as an optimistic ceiling, not a
realistic estimate — see the README for the actual numbers and caveats.

Usage: `python scripts/compare_cloudai_agent.py`
"""

from __future__ import annotations

from statistics import mean, median
from unittest.mock import MagicMock

from cloudai.configurator import CloudAIGymEnv

from autotune.cloudai_agent import AutotuneOptimizeAgent, AutotuneOptimizeAgentConfig

GRID = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0]


def _mock_env(action_space: dict[str, list[float]]) -> MagicMock:
    env = MagicMock(spec=CloudAIGymEnv)
    env.define_action_space.return_value = action_space
    return env


def single_knob_reward(x: float, peak: float) -> float:
    """Reward peaks at `peak`, falls off quadratically -- models "there's a real sweet spot"."""
    return -((x - peak) ** 2) / 1000.0


def single_knob_grid_search_trials(peak: float) -> int:
    """GridSearchAgent: tries every value in listed order, stops at the first one that happens
    to be best -- a pure function of where `peak` sits in GRID, nothing to do with search skill."""
    best = max(GRID, key=lambda x: single_knob_reward(x, peak))
    return GRID.index(best) + 1


def single_knob_optimize_trials(peak: float, seed: int) -> int:
    agent = AutotuneOptimizeAgent(_mock_env({"bandwidth": GRID}), AutotuneOptimizeAgentConfig(random_seed=seed))
    best_value = max(GRID, key=lambda x: single_knob_reward(x, peak))
    for trial in range(1, agent.max_steps + 1):
        result = agent.select_action(observation=None)
        if result is None:
            return agent.max_steps
        step, action = result
        if action["bandwidth"] == best_value:
            return trial
        agent.update_policy(
            {"trial_index": step, "value": single_knob_reward(action["bandwidth"], peak), "action": action}
        )
    return agent.max_steps


def two_knob_reward(a: float, b: float, peak_a: float, peak_b: float) -> float:
    return -(((a - peak_a) ** 2) + ((b - peak_b) ** 2)) / 1000.0


def two_knob_grid_search_trials(peak_a: float, peak_b: float) -> int:
    combos = [(a, b) for a in GRID for b in GRID]
    best = max(combos, key=lambda c: two_knob_reward(c[0], c[1], peak_a, peak_b))
    return combos.index(best) + 1


def two_knob_optimize_trials(peak_a: float, peak_b: float, seed: int) -> int:
    agent = AutotuneOptimizeAgent(
        _mock_env({"knob_a": GRID, "knob_b": GRID}), AutotuneOptimizeAgentConfig(random_seed=seed)
    )
    for trial in range(1, agent.max_steps + 1):
        result = agent.select_action(observation=None)
        if result is None:
            return agent.max_steps
        step, action = result
        if action["knob_a"] == peak_a and action["knob_b"] == peak_b:
            return trial
        r = two_knob_reward(action["knob_a"], action["knob_b"], peak_a, peak_b)
        agent.update_policy({"trial_index": step, "value": r, "action": action})
    return agent.max_steps


def _report(label: str, trials: list[int]) -> None:
    print(f"{label}: mean={mean(trials):.2f} median={median(trials)} min={min(trials)} max={max(trials)}")


if __name__ == "__main__":
    print("=== Single knob, peak in the middle of the grid (300) ===")
    print(f"GridSearchAgent: trial {single_knob_grid_search_trials(300.0)}")
    _report("AutotuneOptimizeAgent (30 seeds)", [single_knob_optimize_trials(300.0, s) for s in range(30)])

    print("\n=== Single knob, peak at the start of the grid (50, best case for grid search) ===")
    print(f"GridSearchAgent: trial {single_knob_grid_search_trials(50.0)}")
    _report("AutotuneOptimizeAgent (30 seeds)", [single_knob_optimize_trials(50.0, s) for s in range(30)])

    print("\n=== Single knob, peak at the end of the grid (550, worst case for grid search) ===")
    print(f"GridSearchAgent: trial {single_knob_grid_search_trials(550.0)}")
    _report("AutotuneOptimizeAgent (30 seeds)", [single_knob_optimize_trials(550.0, s) for s in range(30)])

    print(f"\n=== Two knobs, {len(GRID)}x{len(GRID)}={len(GRID) ** 2} combinations, peak at (300, 150) ===")
    print(f"GridSearchAgent: trial {two_knob_grid_search_trials(300.0, 150.0)}")
    _report(
        "AutotuneOptimizeAgent (30 seeds)",
        [two_knob_optimize_trials(300.0, 150.0, s) for s in range(30)],
    )
