"""Case-review queue: persist analyst decisions in SQLite."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    txn_id      TEXT PRIMARY KEY,
    customer_id TEXT,
    amount      REAL,
    merchant    TEXT,
    category    TEXT,
    risk_score  REAL,
    reasons     TEXT,
    status      TEXT DEFAULT 'open',      -- open | fraud | legit
    reviewed_at TEXT,
    reviewer    TEXT DEFAULT ''
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def upsert_cases(conn: sqlite3.Connection, scored) -> int:
    """Insert flagged transactions as open cases (idempotent). Returns # new."""
    rows = scored[scored["flagged"]].to_dict("records")
    new = 0
    for r in rows:
        cur = conn.execute(
            """INSERT INTO cases
               (txn_id, customer_id, amount, merchant, category, risk_score, reasons)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(txn_id) DO NOTHING""",
            (r["txn_id"], r["customer_id"], float(r["amount"]), r["merchant"],
             r["category"], float(r["risk_score"]), r["top_reasons"]),
        )
        new += cur.rowcount
    conn.commit()
    return new


def list_cases(conn: sqlite3.Connection, status: str | None = None,
               limit: int = 25) -> list[dict]:
    if status is not None and status not in ("open", "fraud", "legit"):
        raise ValueError("status must be one of 'open', 'fraud', 'legit'")
    q = "SELECT * FROM cases"
    params: list = []
    if status:
        q += " WHERE status = ?"
        params.append(status)
    q += " ORDER BY risk_score DESC LIMIT ?"
    params.append(limit)
    cur = conn.execute(q, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def decide(conn: sqlite3.Connection, txn_id: str, verdict: str,
           reviewer: str = "") -> bool:
    """Mark a case as fraud/legit. Returns True if the case existed."""
    if verdict not in ("fraud", "legit"):
        raise ValueError("verdict must be 'fraud' or 'legit'")
    cur = conn.execute(
        "UPDATE cases SET status = ?, reviewed_at = ?, reviewer = ? WHERE txn_id = ?",
        (verdict, datetime.now(timezone.utc).isoformat(), reviewer, txn_id),
    )
    conn.commit()
    return cur.rowcount > 0


def stats(conn: sqlite3.Connection) -> dict:
    cur = conn.execute(
        "SELECT status, COUNT(*) FROM cases GROUP BY status")
    return {row[0]: row[1] for row in cur.fetchall()}
