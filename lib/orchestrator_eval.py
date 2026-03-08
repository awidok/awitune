"""Evaluation and event-detection helpers for orchestration runs."""

import json
import os
import re
import subprocess
from pathlib import Path


def run_evaluate(cfg, out: Path, log_path: Path = None):
    venv_py = Path(os.environ.get("VIRTUAL_ENV", "")) / "bin" / "python3"
    py = str(venv_py) if venv_py.exists() else "python3"
    result = subprocess.run(
        [py, str(cfg.evaluate_script), str(out), "--data-dir", str(cfg.data_dir)],
        capture_output=True, text=True, timeout=300,
    )
    if log_path:
        eval_out = (result.stdout or "").strip() + "\n" + (result.stderr or "").strip()
        if eval_out.strip():
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write("\n\n=== evaluate.py (host) ===\n")
                lf.write(eval_out)
                lf.write("\n")
    return result


def read_eval_results(out: Path) -> dict:
    ep = out / "eval_results.json"
    if ep.exists():
        try:
            return json.loads(ep.read_text())
        except Exception:
            pass
    return {}


def extract_metrics(cfg, ev: dict) -> tuple[float, float]:
    test = ev.get(cfg.test_metric_key, 0) or 0
    val = ev.get(cfg.val_metric_key, 0) or 0
    return test, val


def has_result_event(events_file: Path, tail_bytes: int = 65536) -> bool:
    if not events_file.exists():
        return False
    try:
        with open(events_file, "rb") as ef:
            ef.seek(0, 2)
            sz = ef.tell()
            ef.seek(max(0, sz - tail_bytes))
            tail = ef.read().decode("utf-8", errors="ignore")

        if re.search(r'"type"\s*:\s*"result"', tail):
            return True

        for line in tail.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                return True
    except Exception:
        return False
    return False
