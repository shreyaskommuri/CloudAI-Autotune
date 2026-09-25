"""CloudAI BaseAgent adapter around Autotune's fitted-surface optimizer.

Ships as a `cloudai.agents` entry point (see pyproject.toml) so CloudAI's own
`register_entrypoint_agents` discovers it at process startup with zero changes to CloudAI
itself. See the README's Roadmap "Later" section for why this replaces trying to port search
logic into CloudAI core: that path was proposed twice (issues #993, #997) and declined by
maintainers as an internal-overlap product decision, not a scoping problem. This module is
only imported when CloudAI actually loads the entry point, so it requires `cloudai` to be
installed (the `cloudai-agent` extra) but the rest of Autotune does not.

Unlike `optimizer.suggest_joint_optimize`, which fits Autotune's own `throughput/latency`
efficiency formula from SQLite-backed `Experiment` rows, this agent fits the same quadratic
response surface directly against CloudAI's per-trial `reward` — the scalar CloudAI's own
configurable `reward_function` already computes. That keeps the agent workload-agnostic instead
of assuming any specific pair of named metrics.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, Optional, Tuple

from cloudai.configurator.base_agent import BaseAgent, BaseAgentConfig
from cloudai.configurator.base_gym import BaseGym

from autotune.database import Experiment
from autotune.optimizer import _fit_efficiency_surface
from autotune.recommender import ComboResult, _bounded_neighborhood, _knob_value


class AutotuneOptimizeAgentConfig(BaseAgentConfig):
    """Config for `AutotuneOptimizeAgent`. No fields of its own yet — reuses
    `BaseAgentConfig`'s `random_seed`/`start_action`/`rewards`."""


class AutotuneOptimizeAgent(BaseAgent):
    """Fits `reward ~ b0 + sum(bi*xi + ci*xi^2)` over every completed trial so far via least
    squares, then picks the untried candidate in a bounded neighborhood of the best trial with
    the highest predicted reward. Falls back to a random untried combination when there isn't
    enough data to fit reliably yet, or when a knob's value isn't numeric (the fit requires
    orderable values) — same explore-first-then-exploit shape as `--optimize`'s own fallback.
    """

    def __init__(self, env: BaseGym, config: AutotuneOptimizeAgentConfig):
        self.env = env
        self.config = config
        self.action_space: Dict[str, list[Any]] = {}
        self.knobs: list[str] = []
        self.max_steps = 0
        self.index = 0
        self._trials: list[Experiment] = []
        self._rng = random.Random(config.random_seed)
        self.configure(env.define_action_space())

    @staticmethod
    def get_config_class() -> type[BaseAgentConfig]:
        return AutotuneOptimizeAgentConfig

    def configure(self, config: Dict[str, Any]) -> None:
        """Set the action space and the step budget (one step per point in the full grid,
        matching how many distinct combinations exist to try)."""
        self.action_space = config
        self.knobs = sorted(config.keys())
        self.max_steps = 1
        for values in config.values():
            self.max_steps *= max(len(values), 1)

    def select_action(self, observation: list[float] | None = None) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Select the next untried combination. Stateless w.r.t. `observation` — this agent
        learns from `update_policy` feedback instead, same division of labor as the reward
        argument `run()` passes to `update_policy`."""
        del observation
        self.index += 1
        if self.index > self.max_steps:
            return None

        completed = [t for t in self._trials if t.status == "completed"]
        tried_values = {tuple(_knob_value(t, k) for k in self.knobs) for t in completed}

        action = self._next_action(completed, tried_values)
        if action is None:
            return None
        return self.index, action

    def _next_action(
        self, completed: list[Experiment], tried_values: set[tuple[Optional[float], ...]]
    ) -> Optional[Dict[str, Any]]:
        if not completed:
            return self._random_untried(tried_values)

        best = max(completed, key=lambda t: t.metrics["reward"])
        best_values = {k: _knob_value(best, k) for k in self.knobs}
        if any(v is None for v in best_values.values()):
            return self._random_untried(tried_values)  # a non-numeric knob is in play

        # _bounded_neighborhood generates candidates by doubling/halving (built for Autotune's
        # own continuous-valued config knobs), but CloudAI's action space is a fixed discrete
        # grid — only candidates whose every knob value is actually in that grid are legal here.
        neighborhood = [
            candidate
            for candidate in _bounded_neighborhood(completed, self.knobs, best_values, tried_values)
            if all(candidate[k] in self.action_space[k] for k in self.knobs)
        ]
        if not neighborhood:
            return self._random_untried(tried_values)

        combos = [
            ComboResult(
                values={k: _knob_value(t, k) for k in self.knobs},
                experiment_id=t.id,
                throughput=0.0,
                latency=0.0,
                efficiency=t.metrics["reward"],
            )
            for t in completed
        ]
        predict = _fit_efficiency_surface(combos, self.knobs)
        if predict is None:
            return dict(self._rng.choice(neighborhood))

        return dict(max(neighborhood, key=predict))

    def _random_untried(self, tried_values: set[tuple[Optional[float], ...]]) -> Optional[Dict[str, Any]]:
        all_combos = list(itertools.product(*[self.action_space[k] for k in self.knobs]))
        untried = [c for c in all_combos if c not in tried_values]
        if not untried:
            return None
        return dict(zip(self.knobs, self._rng.choice(untried), strict=True))

    def update_policy(self, _feedback: Dict[str, Any]) -> None:
        """Record the trial's action and reward so the next `select_action` can fit against it.
        `run()` (the default `BaseAgent` loop) always calls this with `action`/`value` set."""
        action = _feedback["action"]
        reward = _feedback["value"]
        self._trials.append(
            Experiment(
                id=len(self._trials),
                created_at=None,
                scenario="cloudai-agent",
                backend="cloudai",
                config_path="",
                config=dict(action),
                status="completed",
                metrics={"reward": reward},
            )
        )
