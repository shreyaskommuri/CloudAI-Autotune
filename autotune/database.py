"""SQLite-backed store for benchmark experiment runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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


class ExperimentDB:
    """Thin wrapper around a SQLite database of benchmark experiments."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
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
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
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
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return Experiment.from_row(row) if row else None

    def list_experiments(self, scenario: Optional[str] = None) -> list[Experiment]:
        if scenario:
            rows = self._conn.execute(
                "SELECT * FROM experiments WHERE scenario = ? ORDER BY id", (scenario,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
        return [Experiment.from_row(row) for row in rows]
