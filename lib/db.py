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
    cols = conn.execute("PRAGMA table_info(experiments)").fetchall()
    col_names = {row[1] for row in cols}
    if "cv_score" not in col_names:
        conn.execute("ALTER TABLE experiments ADD COLUMN cv_score REAL")
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
        "exit_code", "elapsed_min", "test_score", "val_score", "cv_score",
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


def get_dashboard_experiments(limit=500, status=None):
    """
    Lightweight experiment list for dashboard state polling.
    Excludes large JSON blobs (eval/config) and truncates prompt.
    """
    db = get_db()
    base_query = """
        SELECT
            name,
            status,
            substr(prompt, 1, 300) AS prompt,
            base_solution,
            parent_experiment,
            task_type,
            gpu_id,
            created_at,
            started_at,
            finished_at,
            exit_code,
            elapsed_min,
            test_score,
            val_score,
            cv_score,
            improved,
            notes,
            workspace_dir
        FROM experiments
    """
    if status:
        rows = db.execute(
            base_query + " WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = db.execute(
            base_query + " ORDER BY created_at DESC LIMIT ?",
            (limit,),
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


def try_acquire_oof_lock(experiment_name: str, owner: str = "") -> bool:
    """Atomically acquire OOF lock for experiment. Returns True if acquired."""
    db = get_db()
    key = f"oof_lock:{experiment_name}"
    value = owner or datetime.now().isoformat()
    with _lock:
        cur = db.execute(
            "INSERT OR IGNORE INTO global_state (key, value) VALUES (?, ?)",
            (key, value),
        )
        db.commit()
        return cur.rowcount == 1


def release_oof_lock(experiment_name: str):
    db = get_db()
    key = f"oof_lock:{experiment_name}"
    with _lock:
        db.execute("DELETE FROM global_state WHERE key = ?", (key,))
        db.commit()


def get_oof_lock_owner(experiment_name: str):
    return get_global(f"oof_lock:{experiment_name}")


def list_global_keys(prefix: str = "") -> list[str]:
    db = get_db()
    if prefix:
        rows = db.execute("SELECT key FROM global_state WHERE key LIKE ?", (f"{prefix}%",)).fetchall()
    else:
        rows = db.execute("SELECT key FROM global_state").fetchall()
    return [str(r["key"]) for r in rows]


def get_stats(direction="DESC"):
    db = get_db()
    try:
        total_row = db.execute("SELECT COUNT(*) as c FROM experiments").fetchone()
        total = total_row["c"] if total_row else 0
        running_row = db.execute("SELECT COUNT(*) as c FROM experiments WHERE status = 'running'").fetchone()
        running = running_row["c"] if running_row else 0
        completed_row = db.execute("SELECT COUNT(*) as c FROM experiments WHERE status = 'completed'").fetchone()
        completed = completed_row["c"] if completed_row else 0
        failed_row = db.execute("SELECT COUNT(*) as c FROM experiments WHERE status = 'failed'").fetchone()
        failed = failed_row["c"] if failed_row else 0
        improved_row = db.execute("SELECT COUNT(*) as c FROM experiments WHERE improved = 1").fetchone()
        improved = improved_row["c"] if improved_row else 0
    except Exception:
        # Database might not be initialized yet
        total = running = completed = failed = improved = 0
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
