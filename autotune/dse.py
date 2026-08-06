"""Parse CloudAI DSE trajectory output.

CloudAI's DSE/sweep jobs (``cloudai.configurator.cloudai_gym.CloudAIGymEnv``) log one
row per trial to ``<results_root>/<test_name>/<iteration>/trajectory.csv``, with columns
``step, action, reward, observation``. ``action`` and ``observation`` are Python literal
reprs (dict / list), not flat columns, since CloudAI's own ``Trajectory`` class writes
them as whole objects rather than dot-flattened fields.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

TRAJECTORY_FILE_NAME = "trajectory.csv"


@dataclass
class DSETrial:
    step: int
    action: dict[str, Any]
    reward: float
    observation: list[float]


def find_trajectory_files(results_root: Path) -> list[Path]:
    """Find all trajectory.csv files under a CloudAI results directory."""
    return sorted(Path(results_root).rglob(TRAJECTORY_FILE_NAME))


def parse_trajectory(trajectory_path: Path) -> list[DSETrial]:
    """Parse one trajectory.csv into trials, ordered by step."""
    df = pd.read_csv(trajectory_path)
    trials = []
    for _, row in df.iterrows():
        trials.append(
            DSETrial(
                step=int(row["step"]),
                action=_literal(row.get("action"), default={}),
                reward=float(row["reward"]),
                observation=_literal(row.get("observation"), default=[]),
            )
        )
    return trials


def best_trial(trials: list[DSETrial]) -> Optional[DSETrial]:
    """Return the trial with the highest reward, or None if empty."""
    if not trials:
        return None
    return max(trials, key=lambda t: t.reward)


def derive_test_name(trajectory_path: Path) -> str:
    """Derive the CloudAI test name from a trajectory.csv path.

    Path shape is ``<results_root>/<test_name>/<iteration>/trajectory.csv``.
    """
    return trajectory_path.parent.parent.name


def _literal(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return default
    return default
