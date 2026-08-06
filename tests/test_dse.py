from pathlib import Path

from autotune.dse import (
    best_trial,
    derive_test_name,
    find_trajectory_files,
    parse_trajectory,
)

TRAJECTORY_CSV = (
    "step,action,reward,observation\n"
    "1,\"{'batch_size': 1}\",0.5,[100.0]\n"
    "2,\"{'batch_size': 4}\",1.2,[330.0]\n"
    "3,\"{'batch_size': 8}\",0.9,[430.0]\n"
)


def _write_trajectory(root: Path, test_name: str, iteration: str = "0") -> Path:
    trajectory_dir = root / test_name / iteration
    trajectory_dir.mkdir(parents=True)
    trajectory_path = trajectory_dir / "trajectory.csv"
    trajectory_path.write_text(TRAJECTORY_CSV)
    return trajectory_path


def test_parse_trajectory_reads_steps_actions_and_rewards(tmp_path: Path):
    trajectory_path = _write_trajectory(tmp_path, "nemo_run_sweep")

    trials = parse_trajectory(trajectory_path)

    assert len(trials) == 3
    assert trials[0].step == 1
    assert trials[0].action == {"batch_size": 1}
    assert trials[0].reward == 0.5
    assert trials[0].observation == [100.0]
    assert trials[1].action == {"batch_size": 4}


def test_best_trial_picks_highest_reward(tmp_path: Path):
    trajectory_path = _write_trajectory(tmp_path, "nemo_run_sweep")
    trials = parse_trajectory(trajectory_path)

    best = best_trial(trials)

    assert best is not None
    assert best.step == 2
    assert best.reward == 1.2
    assert best.action == {"batch_size": 4}


def test_best_trial_of_empty_list_is_none():
    assert best_trial([]) is None


def test_find_trajectory_files_discovers_nested_files(tmp_path: Path):
    _write_trajectory(tmp_path, "test_a", iteration="0")
    _write_trajectory(tmp_path, "test_b", iteration="0")
    (tmp_path / "test_a" / "0" / "not-a-trajectory.csv").write_text("noise")

    found = find_trajectory_files(tmp_path)

    assert len(found) == 2
    assert all(path.name == "trajectory.csv" for path in found)


def test_derive_test_name_reads_grandparent_dir(tmp_path: Path):
    trajectory_path = _write_trajectory(tmp_path, "nemo_run_sweep", iteration="2")

    assert derive_test_name(trajectory_path) == "nemo_run_sweep"


def test_parse_trajectory_falls_back_to_empty_on_malformed_action(tmp_path: Path):
    trajectory_dir = tmp_path / "broken_sweep" / "0"
    trajectory_dir.mkdir(parents=True)
    trajectory_path = trajectory_dir / "trajectory.csv"
    trajectory_path.write_text("step,action,reward,observation\n1,not a literal,0.5,[1.0]\n")

    trials = parse_trajectory(trajectory_path)

    assert trials[0].action == {}
    assert trials[0].reward == 0.5
