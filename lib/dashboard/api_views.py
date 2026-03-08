"""Data builders for dashboard API routes."""

import json
import re
from datetime import datetime
from pathlib import Path

from .. import db
from ..orchestrator.workspace import analyst_reports_dir


def build_state_payload(rt):
    direction = rt.cfg.best_score_sort_key()
    stats = db.get_stats(direction)
    exps = db.get_all_experiments(limit=2000)

    def _exp_sort_key(e):
        status = e.get("status", "")
        name = e.get("name", "")
        created = e.get("created_at", "")
        is_auto = name.startswith("auto_")
        if status == "running":
            return (0, created)
        if status == "queued" and not is_auto:
            return (1, created)
        if status == "queued" and is_auto:
            return (2, created)
        return (3, "")

    exps.sort(key=_exp_sort_key)

    with rt.lock:
        raw_queue = list(rt.manual_queue) + [{"auto": True, **i} for i in rt.auto_queue]
        running = {str(k): v for k, v in rt.running_gpus.items()}
        used_slots = {str(k): len(v) for k, v in rt.running_gpus.items()}
        used_gpus = list(rt.running_gpus.keys())
        worker_running = rt.worker_running
        proxy_running = rt.proxy_proc is not None and rt.proxy_proc.poll() is None
        auto_queue_size = len(rt.auto_queue)
        ideas_used = len(rt.used_idea_names)

    queue_items = []
    for q in raw_queue:
        qid = q.get("id", "")
        display_name = q.get("name") or q.get("idea_name") or qid
        queue_items.append({**q, "name": display_name})

    return {
        "project_name": rt.cfg.name,
        "metric": rt.cfg.test_metric_key,
        "queue": queue_items,
        "auto_queue_size": auto_queue_size,
        "ideas_total": -1,
        "ideas_used": ideas_used,
        "running": running,
        "history": exps,
        "best_score": stats.get("best_score", 0) or 0,
        "best_experiment": stats.get("best_experiment", ""),
        "available_gpus": rt.cfg.gpus,
        "slots_per_gpu": rt.cfg.slots_per_gpu,
        "used_slots": used_slots,
        "used_gpus": used_gpus,
        "worker_running": worker_running,
        "proxy_running": proxy_running,
        "base_solution": str(rt.cfg.solutions_dir / "baseline"),
        "timeout_minutes": rt.cfg.timeout_minutes,
        "stats": stats,
    }


def build_graph_payload():
    all_exps = db.get_all_experiments(limit=500)
    nodes = []
    edges = []
    node_map = {}

    for exp in all_exps:
        name = exp.get("name", "")
        status = exp.get("status", "unknown")
        test_score = exp.get("test_score")
        improved = exp.get("improved", False)
        parent = exp.get("parent_experiment", "")
        task_type = exp.get("task_type", "experiment")
        created_at = exp.get("created_at", "")

        group = "other"
        if "dcnv2" in name.lower() or "dcn" in name.lower():
            group = "dcnv2"
        elif "transformer" in name.lower() or "attention" in name.lower():
            group = "transformer"
        elif "tabnet" in name.lower():
            group = "tabnet"
        elif "mlp" in name.lower() or "residual" in name.lower():
            group = "mlp"
        elif "lgbm" in name.lower() or "lightgbm" in name.lower() or "catboost" in name.lower():
            group = "tree"
        elif "analysis" in name.lower() or task_type == "analysis":
            group = "analysis"

        node = {
            "id": name,
            "status": status,
            "score": test_score,
            "improved": improved,
            "group": group,
            "task_type": task_type,
            "created_at": created_at,
            "parent": parent,
        }
        nodes.append(node)
        node_map[name] = node

    for node in nodes:
        parent = node.get("parent", "")
        if parent and parent in node_map:
            edges.append({"source": parent, "target": node["id"]})

    baseline_name = "baseline"
    if baseline_name not in node_map:
        nodes.insert(0, {
            "id": baseline_name,
            "status": "root",
            "score": None,
            "improved": False,
            "group": "baseline",
            "task_type": "experiment",
            "created_at": "",
            "parent": "",
        })
        node_map[baseline_name] = nodes[0]

    for node in nodes:
        if node["id"] != baseline_name and not node.get("parent"):
            edges.append({"source": baseline_name, "target": node["id"]})

    best_score = max((n["score"] for n in nodes if n["score"]), default=0)
    return {"nodes": nodes, "edges": edges, "best_score": best_score}


def read_events(rt, exp_name):
    events_path = rt.cfg.experiments_dir / exp_name / "output" / "events.jsonl"
    if not events_path.exists():
        return {"events": [], "raw_lines": 0}

    events = []
    raw_lines = 0
    try:
        for line in events_path.read_text(errors="replace").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            raw_lines += 1
            try:
                ev = json.loads(line)
                ev_type = ev.get("type", "unknown")
                if ev_type == "assistant":
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            events.append({
                                "type": "text",
                                "text": block.get("text", "")[:500],
                                "turn": ev.get("turn", "?"),
                            })
                        elif block.get("type") == "tool_use":
                            events.append({
                                "type": "tool_use",
                                "tool": block.get("name", "?"),
                                "input": json.dumps(block.get("input", {}))[:300],
                                "turn": ev.get("turn", "?"),
                            })
                elif ev_type == "result":
                    msg = ev.get("result", "")
                    events.append({
                        "type": "result",
                        "text": (msg if isinstance(msg, str) else json.dumps(msg))[:500],
                    })
                elif ev_type == "tool_result":
                    events.append({
                        "type": "tool_result",
                        "output": str(ev.get("output", ""))[:500],
                        "tool": ev.get("tool", "?"),
                    })
                else:
                    events.append({"type": ev_type, "raw": json.dumps(ev)[:300]})
            except json.JSONDecodeError:
                events.append({"type": "raw", "text": line[:300]})
    except Exception as exc:
        events.append({"type": "error", "text": str(exc)})

    return {"events": events[-200:], "raw_lines": raw_lines}


def list_experiment_files(rt, exp_name, subdir):
    exp_dir = rt.cfg.experiments_dir / exp_name
    dp = exp_dir / subdir
    files = []

    if subdir == "workspace":
        for name in ("CLAUDE.md", "EXPERIMENT_INFO.md"):
            fp = exp_dir / name
            if fp.exists():
                try:
                    sz = fp.stat().st_size
                    szs = f"{sz}B" if sz < 1024 else (f"{sz/1024:.1f}KB" if sz < 1048576 else f"{sz/1048576:.1f}MB")
                    files.append({"name": name, "path": f"../{name}", "size": szs})
                except OSError:
                    pass

    if dp.exists():
        try:
            for file_path in sorted(dp.rglob("*")):
                try:
                    if file_path.is_file():
                        sz = file_path.stat().st_size
                        szs = f"{sz}B" if sz < 1024 else (
                            f"{sz/1024:.1f}KB" if sz < 1048576 else f"{sz/1048576:.1f}MB"
                        )
                        files.append({
                            "name": str(file_path.relative_to(dp)),
                            "path": str(file_path.relative_to(dp)),
                            "size": szs,
                        })
                except (PermissionError, OSError):
                    pass
        except (PermissionError, OSError):
            pass
    return {"files": files}


def read_experiment_file(rt, exp_name, rel):
    fp = (rt.cfg.experiments_dir / exp_name / rel).resolve()
    if not str(fp).startswith(str(rt.cfg.experiments_dir.resolve())):
        return {"error": "Denied", "status": 403}
    if not fp.exists() or not fp.is_file():
        return {"error": "Not found", "status": 404}
    if fp.suffix in (".parquet", ".pt", ".cbm", ".bin"):
        return {"content": f"Binary: {fp.name} ({fp.stat().st_size}B)", "status": 200}
    try:
        return {"content": fp.read_text(errors="replace")[:200_000], "status": 200}
    except Exception as exc:
        return {"content": str(exc), "status": 200}


def submission_blob(rt, exp_name):
    fp = (rt.cfg.experiments_dir / exp_name / "output" / "submission.parquet").resolve()
    if not str(fp).startswith(str(rt.cfg.experiments_dir.resolve())):
        return {"error": "Denied", "status": 403}
    if not fp.exists():
        return {"error": "Not found", "status": 404}
    return {"bytes": fp.read_bytes(), "status": 200}


def list_analyst_reports(rt):
    reports_dir = analyst_reports_dir(rt.cfg)
    if not reports_dir.exists():
        return {"reports": []}

    reports = []
    for report_file in sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        reports.append({
            "name": report_file.stem,
            "file": report_file.name,
            "size": report_file.stat().st_size,
            "modified": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat(),
        })
    return {"reports": reports}


def read_analyst_report(rt, name):
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "", name)
    report = analyst_reports_dir(rt.cfg) / f"{safe}.md"
    if report.exists():
        return {"content": report.read_text(), "status": 200}
    return {"error": "Not found", "status": 404}
