"""Persist traces as JSON files and a SQLite index.

JSON is the source of truth for full span detail. SQLite is a query index
(trace_id, timestamp, final_status, avg_confidence). Every write commits
explicitly — a missed commit is a silent empty /traces list.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from models import Trace, TraceSummary

ROOT = Path(__file__).resolve().parent
TRACES_DIR = ROOT / "traces"
DB_PATH = TRACES_DIR / "traces.db"


def _connect() -> sqlite3.Connection:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                final_status TEXT NOT NULL,
                avg_confidence REAL NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_trace(trace: Trace) -> None:
    """Write traces/{trace_id}.json and upsert the SQLite index in one place."""
    init_db()
    json_path = TRACES_DIR / f"{trace.trace_id}.json"
    json_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO traces (trace_id, timestamp, final_status, avg_confidence)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                final_status = excluded.final_status,
                avg_confidence = excluded.avg_confidence
            """,
            (
                trace.trace_id,
                trace.timestamp.isoformat(),
                trace.final_status,
                trace.avg_confidence,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_traces() -> list[TraceSummary]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT trace_id, timestamp, final_status, avg_confidence
            FROM traces
            ORDER BY timestamp DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [TraceSummary(**dict(row)) for row in rows]


def load_trace(trace_id: str) -> Trace | None:
    json_path = TRACES_DIR / f"{trace_id}.json"
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return Trace.model_validate(data)
