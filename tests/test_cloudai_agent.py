"""Tests for the CloudAI `cloudai.agents` entry-point adapter.

Requires the `cloudai-agent` extra (`pip install -e .[cloudai-agent]`) since it imports real
CloudAI classes for `spec=` mocking, same pattern CloudAI's own `tests/test_agents.py` uses for
`GridSearchAgent`. Skipped entirely (not failed) when `cloudai` isn't installed, so the default
test run (`pytest -q`, no extras) is unaffected — see pyproject.toml for why this extra isn't in
CI's default install.
"""

from unittest.mock import MagicMock

import pytest

cloudai_configurator = pytest.importorskip("cloudai.configurator")

from autotune.cloudai_agent import AutotuneOptimizeAgent, AutotuneOptimizeAgentConfig  # noqa: E402


@pytest.fixture
def mock_env():
    env = MagicMock(spec=cloudai_configurator.CloudAIGymEnv)
    env.define_action_space.return_value = {"batch_size": [1, 2, 4, 8, 16]}
    return env


def make_agent(mock_env, seed: int = 0) -> AutotuneOptimizeAgent:
    return AutotuneOptimizeAgent(mock_env, AutotuneOptimizeAgentConfig(random_seed=seed))


def test_max_steps_matches_grid_size(mock_env):
    agent = make_agent(mock_env)
    assert agent.max_steps == 5  # one value per point in the 5-element grid


def test_cold_start_picks_a_valid_untried_combo(mock_env):
    agent = make_agent(mock_env)
    step, action = agent.select_action(observation=None)
    assert step == 1
    assert action["batch_size"] in [1, 2, 4, 8, 16]


def test_never_repeats_a_tried_combo(mock_env):
    agent = make_agent(mock_env)
    seen = []
    for _ in range(agent.max_steps):
        step, action = agent.select_action(observation=None)
        seen.append(action["batch_size"])
        agent.update_policy({"trial_index": step, "value": 1.0 / action["batch_size"], "action": action})
    assert sorted(seen) == [1, 2, 4, 8, 16]


def test_stops_after_step_budget_exhausted(mock_env):
    agent = make_agent(mock_env)
    for _ in range(agent.max_steps):
        step, action = agent.select_action(observation=None)
        agent.update_policy({"trial_index": step, "value": 1.0, "action": action})
    assert agent.select_action(observation=None) is None


def test_neighborhood_candidates_stay_inside_the_grid(mock_env):
    """_bounded_neighborhood doubles/halves for Autotune's own continuous knobs; on CloudAI's
    fixed discrete grid, every candidate the agent proposes must still be a real grid value."""
    agent = make_agent(mock_env)
    for _ in range(agent.max_steps):
        step, action = agent.select_action(observation=None)
        assert action["batch_size"] in [1, 2, 4, 8, 16]
        # Reward peaks at batch_size=4, doubling/halving from there would land off-grid (e.g. 32).
        reward = -abs(action["batch_size"] - 4)
        agent.update_policy({"trial_index": step, "value": reward, "action": action})


def test_falls_back_to_random_for_non_numeric_knob(mock_env):
    mock_env.define_action_space.return_value = {"optimizer": ["adamw", "sgd"]}
    agent = make_agent(mock_env)
    for _ in range(agent.max_steps):
        step, action = agent.select_action(observation=None)
        assert action["optimizer"] in ["adamw", "sgd"]
        agent.update_policy({"trial_index": step, "value": 1.0, "action": action})


def test_full_run_loop_via_base_agent_run(mock_env):
    """Drives the agent through CloudAI's own default BaseAgent.run() loop, not just the
    adapter's methods directly, to confirm the observation/reward/done contract lines up."""
    mock_env.reset.return_value = ([0.0], {})

    def fake_step(action):
        reward = -abs(action["batch_size"] - 4)
        return [reward], reward, False, {}

    mock_env.step.side_effect = fake_step

    agent = make_agent(mock_env)
    result = agent.run()

    assert result == 0
    assert mock_env.step.call_count == agent.max_steps
    assert len(agent._trials) == agent.max_steps
