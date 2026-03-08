"""
awitune Dashboard — Web UI & Control Panel for ML experiment orchestration.
Generic: all project-specific values come from ProjectConfig.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from .lib import db
from .lib.config import ProjectConfig
from .lib.dashboard_proxy import start_proxy as runtime_start_proxy, stop_proxy as runtime_stop_proxy
from .lib.dashboard_runtime import (
    MAX_AUTO_QUEUE_SIZE,
    ORCHESTRATOR_LOG_PATH,
    RuntimeState,
    get_docker_cmd,
    orchestrator_log,
)
from .lib.orchestrator_eval import (
    extract_metrics as eval_extract_metrics,
    has_result_event as eval_has_result_event,
    read_eval_results as eval_read_results,
    run_evaluate as eval_run_evaluate,
)
from .lib.orchestrator_queue import (
    collect_used_idea_names as queue_collect_used_idea_names,
    queue_idea as queue_queue_idea,
)
from .lib.notifications import send_telegram_notification as notify_telegram
from .lib.orchestrator_service import OrchestratorService
from .lib.orchestrator_workspace import (
    analyst_reports_dir as ws_analyst_reports_dir,
    build_reference_code_section as ws_build_reference_code_section,
    copy_analyst_reports_to_workspace as ws_copy_analyst_reports_to_workspace,
    get_analyst_reports_summary as ws_get_analyst_reports_summary,
    prepare_analyst_workspace as ws_prepare_analyst_workspace,
    prepare_workspace as ws_prepare_workspace,
    resolve_base_solution as ws_resolve_base_solution,
)

load_dotenv()

AWITUNE_DIR = Path(__file__).resolve().parent
PROXY_HOST = os.getenv("PROXY_HOST", "localhost")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = Flask(__name__, template_folder=str(AWITUNE_DIR / "templates"))
CORS(app)


# ---- Telegram notifications ----
def send_telegram_notification(message: str, parse_mode: str = "HTML"):
    notify_telegram(rt.cfg, message, parse_mode=parse_mode)


def trim_auto_queue(max_size: int = MAX_AUTO_QUEUE_SIZE) -> int:
    return orch.trim_auto_queue(max_size)


DOCKER_CMD = get_docker_cmd()
rt = RuntimeState(DOCKER_CMD)
orch = OrchestratorService(rt, DOCKER_CMD, AWITUNE_DIR, PROXY_HOST, OPENAI_API_KEY, orchestrator_log)


def _force_rmtree(path: Path):
    orch._force_rmtree(path)


# ---- Proxy ----
def start_proxy():
    orch.start_proxy()


def stop_proxy():
    orch.stop_proxy()


# ---- Workspace ----
def resolve_base_solution(base_experiment: str) -> str:
    return orch.resolve_base_solution(base_experiment)


def build_reference_code_section(reference_code: dict) -> str:
    return ws_build_reference_code_section(rt.cfg, reference_code)


def _analyst_reports_dir():
    return ws_analyst_reports_dir(rt.cfg)


def get_analyst_reports_summary() -> str:
    return ws_get_analyst_reports_summary(rt.cfg)


def copy_analyst_reports_to_workspace(ws: Path):
    ws_copy_analyst_reports_to_workspace(rt.cfg, ws)


def prepare_analyst_workspace(exp_dir, analysis_focus, prev_exps):
    return ws_prepare_analyst_workspace(rt.cfg, exp_dir, analysis_focus)


def prepare_workspace(base_path, exp_dir, custom_prompt, best_score, prev_exps,
                      reference_code=None):
    return ws_prepare_workspace(
        rt.cfg,
        base_path,
        exp_dir,
        custom_prompt,
        best_score,
        prev_exps,
        reference_code=reference_code,
    )


# ---- Evaluation helpers ----
def _run_evaluate(out: Path, log_path: Path = None):
    return eval_run_evaluate(rt.cfg, out, log_path)


def _read_eval_results(out: Path) -> dict:
    return eval_read_results(out)


def _extract_metrics(ev: dict) -> tuple[float, float]:
    return eval_extract_metrics(rt.cfg, ev)


def _has_result_event(events_file: Path, tail_bytes: int = 65536) -> bool:
    return eval_has_result_event(events_file, tail_bytes=tail_bytes)


# ---- Agent runner ----
def run_agent_in_thread(exp_name, prompt, base_solution, gpu_id,
                        reference_code=None, task_type="experiment"):
    return orch.run_agent_in_thread(exp_name, prompt, base_solution, gpu_id, reference_code=reference_code, task_type=task_type)


def _finalize_analysis(exp_name, exp_dir, out, elapsed, ec):
    return orch._finalize_analysis(exp_name, exp_dir, out, elapsed, ec)


def _finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp):
    return orch._finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp)


# ---- Auto-feeder ----
def _get_idea_feeder():
    return orch._get_idea_feeder()


# ---- Worker ----
def _collect_used_idea_names() -> set:
    return collect_used_idea_names(rt)


def _queue_idea(idea: dict, idx: int) -> bool:
    return queue_idea(rt, rt.cfg, idea, idx, resolve_base_solution, orchestrator_log)


def worker_loop():
    return orch.worker_loop()


def start_worker():
    return orch.start_worker()


def stop_worker():
    return orch.stop_worker()


# ---- Routes ----

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    # Don't sync on every request - only at startup via cli.py
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
        elif status == "queued" and not is_auto:
            return (1, created)
        elif status == "queued" and is_auto:
            return (2, created)
        else:
            return (3, "")
    exps.sort(key=_exp_sort_key)

    queue_items = []
    with rt.lock:
        raw_queue = list(rt.manual_queue) + [{"auto": True, **i} for i in rt.auto_queue]
    for q in raw_queue:
        qid = q.get("id", "")
        display_name = q.get("name") or q.get("idea_name") or qid
        queue_items.append({
            **q,
            "name": display_name,
        })

    return jsonify({
        "project_name": rt.cfg.name,
        "metric": rt.cfg.test_metric_key,
        "queue": queue_items,
        "auto_queue_size": len(rt.auto_queue),
        "ideas_total": -1,  # No longer using static ideas
        "ideas_used": len(rt.used_idea_names),
        "running": {str(k): v for k, v in rt.running_gpus.items()},
        "history": exps,
        "best_score": stats.get("best_score", 0) or 0,
        "best_experiment": stats.get("best_experiment", ""),
        "available_gpus": rt.cfg.gpus,
        "slots_per_gpu": rt.cfg.slots_per_gpu,
        "used_slots": {str(k): len(v) for k, v in rt.running_gpus.items()},
        "used_gpus": list(rt.running_gpus.keys()),
        "worker_running": rt.worker_running,
        "proxy_running": rt.proxy_proc is not None and rt.proxy_proc.poll() is None,
        "base_solution": str(rt.cfg.solutions_dir / "baseline"),
        "timeout_minutes": rt.cfg.timeout_minutes,
        "stats": stats,
    })


@app.route("/graph")
def graph_page():
    """Experiment graph visualization page."""
    return render_template("graph.html")


@app.route("/api/launch", methods=["POST"])
def api_launch():
    d = request.get_json() or {}
    prompt = d.get("prompt", "")
    name = d.get("name", "")
    base_experiment = d.get("base_experiment", "")
    task_type = d.get("task_type", "experiment")
    base = resolve_base_solution(base_experiment)
    parent = base_experiment if base_experiment and base_experiment != "default" else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "analyst" if task_type == "analysis" else "agent"
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name or "")[:40] if name else ""
    eid = f"{prefix}_{safe_name}_{ts}" if safe_name else f"{prefix}_{ts}"
    db.create_experiment(eid, prompt=prompt, base_solution=base,
                         parent_experiment=parent, task_type=task_type)
    db.add_log(eid, f"Queued {task_type}: {prompt[:100]}")
    item = {"id": eid, "prompt": prompt, "base_solution": base, "task_type": task_type}
    with rt.lock:
        rt.manual_queue.append(item)
    if not rt.worker_running:
        start_worker()
    return jsonify({"status": "queued", "id": eid})


@app.route("/api/queue/remove", methods=["POST"])
def api_queue_remove():
    d = request.get_json() or {}
    idx = d.get("index", -1)
    with rt.lock:
        combined = list(rt.manual_queue) + list(rt.auto_queue)
        if 0 <= idx < len(combined):
            removed = combined[idx]
            if idx < len(rt.manual_queue):
                del rt.manual_queue[idx]
            else:
                del rt.auto_queue[idx - len(rt.manual_queue)]
            db.update_experiment(removed["id"], status="cancelled")
            return jsonify({"status": "removed"})
    return jsonify({"status": "error"}), 400


@app.route("/api/worker/start", methods=["POST"])
def api_worker_start():
    start_worker()
    return jsonify({"status": "started"})


@app.route("/api/worker/stop", methods=["POST"])
def api_worker_stop():
    stop_worker()
    return jsonify({"status": "stopped"})


@app.route("/api/proxy/start", methods=["POST"])
def api_proxy_start():
    start_proxy()
    return jsonify({"status": "started"})


@app.route("/api/proxy/stop", methods=["POST"])
def api_proxy_stop():
    stop_proxy()
    return jsonify({"status": "stopped"})


@app.route("/api/log/<exp_name>/agent")
def api_agent_log(exp_name):
    p = rt.cfg.experiments_dir / exp_name / "agent.log"
    if p.exists():
        c = p.read_text(errors="replace")
        return Response(c[-200_000:] if len(c) > 200_000 else c, mimetype="text/plain")
    return Response("No log", mimetype="text/plain")


@app.route("/api/log/proxy")
def api_proxy_log():
    p = Path("/tmp/awitune_proxy.log")
    if p.exists():
        c = p.read_text(errors="replace")
        return Response(c[-200_000:] if len(c) > 200_000 else c, mimetype="text/plain")
    return Response("No proxy log", mimetype="text/plain")


@app.route("/api/log/orchestrator")
def api_orchestrator_log():
    """Get orchestrator/planner log."""
    if ORCHESTRATOR_LOG_PATH.exists():
        c = ORCHESTRATOR_LOG_PATH.read_text(errors="replace")
        return Response(c[-200_000:] if len(c) > 200_000 else c, mimetype="text/plain")
    return Response("No orchestrator log yet", mimetype="text/plain")


@app.route("/api/events/<exp_name>")
def api_events(exp_name):
    events_path = rt.cfg.experiments_dir / exp_name / "output" / "events.jsonl"
    if not events_path.exists():
        return jsonify({"events": [], "raw_lines": 0})
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
                            events.append({"type": "text", "text": block.get("text", "")[:500],
                                           "turn": ev.get("turn", "?")})
                        elif block.get("type") == "tool_use":
                            events.append({"type": "tool_use", "tool": block.get("name", "?"),
                                           "input": json.dumps(block.get("input", {}))[:300],
                                           "turn": ev.get("turn", "?")})
                elif ev_type == "result":
                    msg = ev.get("result", "")
                    events.append({"type": "result", "text": (msg if isinstance(msg, str) else json.dumps(msg))[:500]})
                elif ev_type == "tool_result":
                    events.append({"type": "tool_result",
                                   "output": str(ev.get("output", ""))[:500],
                                   "tool": ev.get("tool", "?")})
                else:
                    events.append({"type": ev_type, "raw": json.dumps(ev)[:300]})
            except json.JSONDecodeError:
                events.append({"type": "raw", "text": line[:300]})
    except Exception as e:
        events.append({"type": "error", "text": str(e)})
    return jsonify({"events": events[-200:], "raw_lines": raw_lines})


@app.route("/api/graph")
def api_graph():
    """Get experiment graph data for visualization."""
    all_exps = db.get_all_experiments(limit=500)
    
    nodes = []
    edges = []
    node_map = {}
    
    # Build nodes
    for exp in all_exps:
        name = exp.get("name", "")
        status = exp.get("status", "unknown")
        test_score = exp.get("test_score")
        improved = exp.get("improved", False)
        parent = exp.get("parent_experiment", "")
        task_type = exp.get("task_type", "experiment")
        created_at = exp.get("created_at", "")
        
        # Determine node group (solution family)
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
    
    # Build edges
    for node in nodes:
        parent = node.get("parent", "")
        if parent and parent in node_map:
            edges.append({
                "source": parent,
                "target": node["id"],
            })
    
    # Add baseline as root node if not present
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
    
    # Connect experiments without parent to baseline
    for node in nodes:
        if node["id"] != baseline_name and not node.get("parent"):
            edges.append({
                "source": baseline_name,
                "target": node["id"],
            })
    
    # Calculate best score
    best_score = max((n["score"] for n in nodes if n["score"]), default=0)
    
    return jsonify({
        "nodes": nodes,
        "edges": edges,
        "best_score": best_score,
    })


@app.route("/api/files/<exp_name>/<subdir>")
def api_files(exp_name, subdir):
    exp_dir = rt.cfg.experiments_dir / exp_name
    dp = exp_dir / subdir
    files = []
    # For workspace tab, also show experiment-level files (CLAUDE.md, EXPERIMENT_INFO.md)
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
            for f in sorted(dp.rglob("*")):
                try:
                    if f.is_file():
                        sz = f.stat().st_size
                        szs = f"{sz}B" if sz < 1024 else (f"{sz/1024:.1f}KB" if sz < 1048576 else f"{sz/1048576:.1f}MB")
                        files.append({"name": str(f.relative_to(dp)), "path": str(f.relative_to(dp)), "size": szs})
                except (PermissionError, OSError):
                    pass
        except (PermissionError, OSError):
            pass
    return jsonify({"files": files})


@app.route("/api/file/<exp_name>")
def api_file_content(exp_name):
    rel = request.args.get("path", "")
    if not rel:
        return Response("No path", status=400)
    fp = (rt.cfg.experiments_dir / exp_name / rel).resolve()
    if not str(fp).startswith(str(rt.cfg.experiments_dir.resolve())):
        return Response("Denied", status=403)
    if fp.exists() and fp.is_file():
        if fp.suffix in (".parquet", ".pt", ".cbm", ".bin"):
            return Response(f"Binary: {fp.name} ({fp.stat().st_size}B)", mimetype="text/plain")
        try:
            return Response(fp.read_text(errors="replace")[:200_000], mimetype="text/plain")
        except Exception as e:
            return Response(str(e), mimetype="text/plain")
    return Response("Not found", status=404)


@app.route("/api/download/<exp_name>/submission")
def api_download_submission(exp_name):
    fp = (rt.cfg.experiments_dir / exp_name / "output" / "submission.parquet").resolve()
    if not str(fp).startswith(str(rt.cfg.experiments_dir.resolve())):
        return Response("Denied", status=403)
    if not fp.exists():
        return Response("Not found", status=404)
    return Response(
        fp.read_bytes(),
        mimetype="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="submission_{exp_name}.parquet"'},
    )


def _find_claude_tasks_dir(container_name):
    try:
        r = subprocess.run(
            DOCKER_CMD + ["exec", container_name, "sh", "-c",
                          "ls -d /tmp/claude-*/-app-workspace/tasks 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return None


@app.route("/api/tasks/<exp_name>")
def api_tasks(exp_name):
    events_path = rt.cfg.experiments_dir / exp_name / "output" / "events.jsonl"
    tasks = {}
    if events_path.exists():
        try:
            for line in events_path.read_text(errors="replace").strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if ev.get("type") != "assistant":
                        continue
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") != "tool_use" or block.get("name") != "TaskOutput":
                            continue
                        task_id = block.get("input", {}).get("task_id", "")
                        if task_id and task_id not in tasks:
                            tasks[task_id] = {"id": task_id, "description": f"task {task_id}",
                                              "turn": ev.get("turn", "?")}
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass

    exp = db.get_experiment(exp_name)
    cn = exp.get("container_name", "") if exp else ""
    task_files = []
    if cn:
        try:
            tasks_dir = _find_claude_tasks_dir(cn)
            if tasks_dir:
                # Get list of .output files with size and modification time
                cmd = DOCKER_CMD + ["exec", cn, "sh", "-c",
                                   f"for f in {tasks_dir}/*.output; do stat -c '%n|%s|%Y' \"$f\"; done"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    for line in r.stdout.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parts = line.split("|")
                            if len(parts) >= 3:
                                fname = parts[0]
                                tid = fname.replace(".output", "").split("/")[-1]
                                size = parts[1]
                                timestamp = int(parts[2])
                                # Convert to local time (Moscow UTC+3)
                                utc_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                                moscow_tz = timezone(timedelta(hours=3))
                                local_dt = utc_dt.astimezone(moscow_tz)
                                modified = local_dt.strftime("%b %d %H:%M")
                                task_files.append({"id": tid, "size": size, "modified": modified, "timestamp": timestamp})
                                if tid not in tasks:
                                    tasks[tid] = {"id": tid, "description": "background task", "turn": "?"}
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
    return jsonify({"tasks": sorted(tasks.values(), key=lambda t: str(t.get("turn", ""))),
                    "task_files": task_files, "container": cn})


@app.route("/api/task_output/<exp_name>/<task_id>")
def api_task_output(exp_name, task_id):
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', task_id)
    if not safe_id:
        return Response("Invalid task_id", status=400)
    exp = db.get_experiment(exp_name)
    cn = exp.get("container_name", "") if exp else ""
    if not cn:
        return Response("No container found", mimetype="text/plain")
    tasks_dir = _find_claude_tasks_dir(cn)
    if not tasks_dir:
        return Response("Could not find tasks directory", mimetype="text/plain")
    try:
        r = subprocess.run(DOCKER_CMD + ["exec", cn, "cat", f"{tasks_dir}/{safe_id}.output"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return Response(r.stdout[-200_000:], mimetype="text/plain")
        return Response(f"Not found: {r.stderr.strip()}", mimetype="text/plain")
    except subprocess.TimeoutExpired:
        return Response("Timeout", mimetype="text/plain")
    except Exception as e:
        return Response(f"Error: {e}", mimetype="text/plain")


@app.route("/api/docker")
def api_docker():
    cs = []
    try:
        r = subprocess.run(DOCKER_CMD + ["ps", "--filter", "name=agent-", "--format", "{{.Names}}\t{{.Status}}"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split("\t", 1)
                cs.append({"name": parts[0], "status": parts[1] if len(parts) > 1 else "?"})
    except Exception:
        pass
    return jsonify({"containers": cs})


@app.route("/api/kill/<exp_name>", methods=["POST"])
def api_kill(exp_name):
    try:
        for gpu in rt.cfg.gpus:
            subprocess.run(DOCKER_CMD + ["kill", f"agent-{exp_name}-gpu{gpu}"], capture_output=True, timeout=5)
        db.update_experiment(exp_name, status="killed", notes="Manually killed")
        db.add_log(exp_name, "Manually killed", level="warning")
        return jsonify({"status": "killed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/restart/<exp_name>", methods=["POST"])
def api_restart(exp_name):
    try:
        exp = db.get_experiment(exp_name)
        if not exp:
            return jsonify({"status": "error", "message": "Not found"}), 404
        for gpu in rt.cfg.gpus:
            subprocess.run(DOCKER_CMD + ["kill", f"agent-{exp_name}-gpu{gpu}"],
                           capture_output=True, timeout=5)
        with rt.lock:
            for gpu_id in list(rt.running_gpus.keys()):
                rt.remove_experiment_from_gpu(gpu_id, exp_name)
        exp_dir = rt.cfg.experiments_dir / exp_name
        for subdir in ["output", "workspace"]:
            d = exp_dir / subdir
            if d.exists():
                _force_rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o777)
        prompt = exp.get("prompt", "")
        base = exp.get("base_solution", "") or str(rt.cfg.solutions_dir / "baseline")
        db.update_experiment(exp_name, status="queued",
                             started_at=None, finished_at=None,
                             exit_code=None, elapsed_min=None,
                             test_score=None, val_score=None,
                             improved=0, notes="", eval_json=None)
        db.add_log(exp_name, "Restarted")
        with rt.lock:
            rt.manual_queue.append({"id": exp_name, "prompt": prompt, "base_solution": base})
        if not rt.worker_running:
            start_worker()
        return jsonify({"status": "queued", "id": exp_name})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/delete/<exp_name>", methods=["POST"])
def api_delete(exp_name):
    try:
        for gpu in rt.cfg.gpus:
            subprocess.run(DOCKER_CMD + ["kill", f"agent-{exp_name}-gpu{gpu}"],
                           capture_output=True, timeout=5)
        with rt.lock:
            for gpu_id in list(rt.running_gpus.keys()):
                rt.remove_experiment_from_gpu(gpu_id, exp_name)
        db.delete_experiment(exp_name)
        exp_dir = rt.cfg.experiments_dir / exp_name
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/experiment/<exp_name>")
def api_experiment_detail(exp_name):
    exp = db.get_experiment(exp_name)
    logs = db.get_logs(exp_name, limit=100)
    return jsonify({"experiment": exp, "logs": logs})


@app.route("/api/experiments/bases")
def api_experiment_bases():
    exps = db.get_all_experiments(limit=200, status="completed")
    bases = []
    for e in exps:
        ws = e.get("workspace_dir", "")
        if ws and Path(ws).is_dir() and (Path(ws) / "run.py").exists():
            bases.append({
                "name": e["name"],
                "test_score": e.get("test_score"),
                "improved": e.get("improved"),
                "parent": e.get("parent_experiment", ""),
            })
    bases.sort(key=lambda x: x.get("test_score") or 0, reverse=True)
    return jsonify({"bases": bases})


@app.route("/api/analyst_reports")
def api_analyst_reports():
    d = _analyst_reports_dir()
    if not d.exists():
        return jsonify({"reports": []})
    reports = []
    for f in sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        reports.append({
            "name": f.stem, "file": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return jsonify({"reports": reports})


@app.route("/api/analyst_reports/<name>")
def api_analyst_report(name):
    safe = re.sub(r'[^a-zA-Z0-9_.-]', '', name)
    report = _analyst_reports_dir() / f"{safe}.md"
    if report.exists():
        return report.read_text(), 200, {"Content-Type": "text/markdown; charset=utf-8"}
    return "Not found", 404


# ---- Orphan recovery ----

def _check_agent_finished(events_file):
    return eval_has_result_event(events_file)


def _get_events_file_age(events_file: Path) -> float:
    return orch._get_events_file_age(events_file)


def _find_container_name(exp_name, active_containers):
    return orch._find_container_name(exp_name, active_containers)


def _finalize_experiment(exp_name, best_score):
    return orch._finalize_experiment(exp_name, best_score)


def _monitor_orphaned_container(exp_name, container_name, gpu_id):
    return orch._monitor_orphaned_container(exp_name, container_name, gpu_id)


def recover_orphaned_experiments():
    return orch.recover_orphaned_experiments()


# ---- Main init (called from cli.py) ----
def init_app(cfg: ProjectConfig):
    """Initialize the dashboard with a project config."""
    rt.cfg = cfg
    db.configure(cfg.experiments_dir)
    db.init_db()
