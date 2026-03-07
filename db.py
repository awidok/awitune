"""
SQLite database for experiment tracking.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

_lock = threading.Lock()
_conn = None
_db_path: Path | None = None


def configure(experiments_dir: Path):
    """Set the database path. Must be called before any DB operations."""
    global _db_path
    _db_path = experiments_dir / "experiments.db"


def get_connection():
    conn = sqlite3.connect(str(_db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_db():
    global _conn
    if _conn is None:
        _db_path.parent.mkdir(exist_ok=True)
        _conn = get_connection()
        init_db(_conn)
    return _conn


def init_db(conn=None):
    if conn is None:
        conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'queued',
            prompt TEXT DEFAULT '',
            base_solution TEXT DEFAULT '',
            parent_experiment TEXT DEFAULT '',
            task_type TEXT DEFAULT 'experiment',
            gpu_id INTEGER,
            container_name TEXT,
            container_id TEXT,
            exp_dir TEXT,
            workspace_dir TEXT,
            output_dir TEXT,

            created_at TEXT,
            started_at TEXT,
            finished_at TEXT,

            exit_code INTEGER,
            elapsed_min REAL,
            test_score REAL,
            val_score REAL,
            improved INTEGER DEFAULT 0,
            notes TEXT DEFAULT '',

            config_json TEXT,
            eval_json TEXT
        );

        CREATE TABLE IF NOT EXISTS experiment_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            timestamp TEXT,
            level TEXT DEFAULT 'info',
            message TEXT,
            FOREIGN KEY (experiment_name) REFERENCES experiments(name)
        );

        CREATE TABLE IF NOT EXISTS global_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
        CREATE INDEX IF NOT EXISTS idx_logs_exp ON experiment_logs(experiment_name);
    """)
    conn.commit()


def create_experiment(name, prompt="", base_solution="", gpu_id=None,
                      parent_experiment="", task_type="experiment"):
    db = get_db()
    with _lock:
        db.execute("""
            INSERT OR REPLACE INTO experiments
            (name, status, prompt, base_solution, parent_experiment, task_type, gpu_id, created_at)
            VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
        """, (name, prompt, base_solution, parent_experiment or "",
              task_type or "experiment", gpu_id, datetime.now().isoformat()))
        db.commit()


def update_experiment(name, **kwargs):
    db = get_db()
    allowed = {
        "status", "gpu_id", "container_name", "container_id",
        "exp_dir", "workspace_dir", "output_dir",
        "started_at", "finished_at",
        "exit_code", "elapsed_min", "test_score", "val_score",
        "improved", "notes", "config_json", "eval_json",
        "parent_experiment", "task_type",
    }
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [name]
    with _lock:
        db.execute(f"UPDATE experiments SET {set_clause} WHERE name = ?", values)
        db.commit()


def get_experiment(name):
    db = get_db()
    row = db.execute("SELECT * FROM experiments WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def get_all_experiments(limit=100, status=None):
    db = get_db()
    if status:
        rows = db.execute(
            "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_best_experiment(direction="DESC"):
    """Get best experiment. direction='DESC' for maximize, 'ASC' for minimize."""
    db = get_db()
    row = db.execute(
        f"SELECT * FROM experiments WHERE test_score IS NOT NULL "
        f"ORDER BY test_score {direction} LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def delete_experiment(name):
    db = get_db()
    with _lock:
        db.execute("DELETE FROM experiment_logs WHERE experiment_name = ?", (name,))
        db.execute("DELETE FROM experiments WHERE name = ?", (name,))
        db.commit()


def add_log(experiment_name, message, level="info"):
    db = get_db()
    with _lock:
        db.execute("""
            INSERT INTO experiment_logs (experiment_name, timestamp, level, message)
            VALUES (?, ?, ?, ?)
        """, (experiment_name, datetime.now().isoformat(), level, message))
        db.commit()


def get_logs(experiment_name, limit=500):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM experiment_logs WHERE experiment_name = ? ORDER BY id DESC LIMIT ?",
        (experiment_name, limit)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def set_global(key, value):
    db = get_db()
    with _lock:
        db.execute(
            "INSERT OR REPLACE INTO global_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value) if not isinstance(value, str) else value)
        )
        db.commit()


def get_global(key, default=None):
    db = get_db()
    row = db.execute("SELECT value FROM global_state WHERE key = ?", (key,)).fetchone()
    if row:
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]
    return default


def get_stats(direction="DESC"):
    db = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM experiments").fetchone()["c"]
    running = db.execute("SELECT COUNT(*) as c FROM experiments WHERE status = 'running'").fetchone()["c"]
    completed = db.execute("SELECT COUNT(*) as c FROM experiments WHERE status = 'completed'").fetchone()["c"]
    failed = db.execute("SELECT COUNT(*) as c FROM experiments WHERE status = 'failed'").fetchone()["c"]
    improved = db.execute("SELECT COUNT(*) as c FROM experiments WHERE improved = 1").fetchone()["c"]
    best = get_best_experiment(direction)
    return {
        "total": total,
        "running": running,
        "completed": completed,
        "failed": failed,
        "improved": improved,
        "best_score": best["test_score"] if best else 0,
        "best_experiment": best["name"] if best else "",
    }
