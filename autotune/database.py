"""SQLite-backed store for benchmark experiment runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .dse import DSETrial

DEFAULT_DB_PATH = Path("autotune.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    scenario TEXT NOT NULL,
    backend TEXT NOT NULL,
    config_path TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    report_path TEXT,
    metrics_json TEXT,
    notes TEXT,
    metadata_json TEXT
);
"""

DSE_TRIALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS dse_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    step INTEGER NOT NULL,
    action_json TEXT NOT NULL,
    reward REAL NOT NULL,
    observation_json TEXT NOT NULL,
    UNIQUE(experiment_id, step)
);
"""

SEARCH_SUGGESTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    knobs_key TEXT NOT NULL,
    suggested_json TEXT NOT NULL,
    predicted_efficiency REAL,
    resolved_experiment_id INTEGER REFERENCES experiments(id),
    actual_efficiency REAL
);
"""


@dataclass
class Experiment:
    id: Optional[int]
    created_at: Optional[str]
    scenario: str
    backend: str
    config_path: str
    config: dict[str, Any]
    status: str = "pending"
    report_path: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Experiment":
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            scenario=row["scenario"],
            backend=row["backend"],
            config_path=row["config_path"],
            config=json.loads(row["config_json"]),
            status=row["status"],
            report_path=row["report_path"],
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            notes=row["notes"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )


@dataclass
class SuggestionRecord:
    id: Optional[int]
    created_at: Optional[str]
    knobs: list[str]
    suggested_values: dict[str, float]
    predicted_efficiency: Optional[float]
    resolved_experiment_id: Optional[int]
    actual_efficiency: Optional[float]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SuggestionRecord":
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            knobs=row["knobs_key"].split(","),
            suggested_values=json.loads(row["suggested_json"]),
            predicted_efficiency=row["predicted_efficiency"],
            resolved_experiment_id=row["resolved_experiment_id"],
            actual_efficiency=row["actual_efficiency"],
        )


@dataclass
class DSETrialRow:
    id: Optional[int]
    experiment_id: int
    step: int
    action: dict[str, Any]
    reward: float
    observation: list[Any]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DSETrialRow":
        return cls(
            id=row["id"],
            experiment_id=row["experiment_id"],
            step=row["step"],
            action=json.loads(row["action_json"]),
            reward=row["reward"],
            observation=json.loads(row["observation_json"]),
        )


class ExperimentDB:
    """Thin wrapper around a SQLite database of benchmark experiments."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
        self._conn.execute(DSE_TRIALS_SCHEMA)
        self._conn.execute(SEARCH_SUGGESTIONS_SCHEMA)
        self._ensure_columns()
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentDB":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def add_experiment(
        self,
        scenario: str,
        backend: str,
        config_path: str,
        config: dict[str, Any],
        status: str = "pending",
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO experiments (scenario, backend, config_path, config_json, status, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                scenario,
                backend,
                config_path,
                json.dumps(config),
                status,
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def _ensure_columns(self) -> None:
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(experiments)").fetchall()}
        if "metadata_json" not in columns:
            self._conn.execute("ALTER TABLE experiments ADD COLUMN metadata_json TEXT")

    def update_result(
        self,
        experiment_id: int,
        status: str,
        report_path: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            UPDATE experiments
            SET status = ?, report_path = ?, metrics_json = ?, notes = ?
            WHERE id = ?
            """,
            (
                status,
                report_path,
                json.dumps(metrics) if metrics is not None else None,
                notes,
                experiment_id,
            ),
        )
        self._conn.commit()

    def get(self, experiment_id: int) -> Optional[Experiment]:
        row = self._conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        return Experiment.from_row(row) if row else None

    def list_experiments(self, scenario: Optional[str] = None) -> list[Experiment]:
        if scenario:
            rows = self._conn.execute(
                "SELECT * FROM experiments WHERE scenario = ? ORDER BY id", (scenario,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
        return [Experiment.from_row(row) for row in rows]

    def add_dse_trials(self, experiment_id: int, trials: list[DSETrial]) -> None:
        """Store DSE trials for an experiment, replacing any trials already stored for it."""
        self._conn.execute("DELETE FROM dse_trials WHERE experiment_id = ?", (experiment_id,))
        self._conn.executemany(
            """
            INSERT INTO dse_trials (experiment_id, step, action_json, reward, observation_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (experiment_id, trial.step, json.dumps(trial.action), trial.reward, json.dumps(trial.observation))
                for trial in trials
            ],
        )
        self._conn.commit()

    def get_dse_trials(self, experiment_id: int) -> list[DSETrialRow]:
        rows = self._conn.execute(
            "SELECT * FROM dse_trials WHERE experiment_id = ? ORDER BY step", (experiment_id,)
        ).fetchall()
        return [DSETrialRow.from_row(row) for row in rows]

    def add_suggestion(
        self,
        knobs: list[str],
        suggested_values: dict[str, Any],
        predicted_efficiency: Optional[float],
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO search_suggestions (knobs_key, suggested_json, predicted_efficiency)
            VALUES (?, ?, ?)
            """,
            (",".join(sorted(knobs)), json.dumps(suggested_values), predicted_efficiency),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_suggestions(self, knobs: list[str]) -> list[SuggestionRecord]:
        """Suggestions previously recorded for this exact set of knobs, order-independent."""
        rows = self._conn.execute(
            "SELECT * FROM search_suggestions WHERE knobs_key = ? ORDER BY id",
            (",".join(sorted(knobs)),),
        ).fetchall()
        return [SuggestionRecord.from_row(row) for row in rows]

    def resolve_suggestion(self, suggestion_id: int, experiment_id: int, actual_efficiency: float) -> None:
        self._conn.execute(
            """
            UPDATE search_suggestions
            SET resolved_experiment_id = ?, actual_efficiency = ?
            WHERE id = ?
            """,
            (experiment_id, actual_efficiency, suggestion_id),
        )
        self._conn.commit()
