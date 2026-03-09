"""
awitune Dashboard — Web UI & Control Panel for ML experiment orchestration.
Generic: all project-specific values come from ProjectConfig.
"""

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from .. import db
from ..config import ProjectConfig
from .proxy import start_proxy as runtime_start_proxy, stop_proxy as runtime_stop_proxy
from .runtime import (
    MAX_AUTO_QUEUE_SIZE,
    ORCHESTRATOR_LOG_PATH,
    RuntimeState,
    get_docker_cmd,
    orchestrator_log,
)
from ..notifications import send_telegram_notification as notify_telegram
from ..orchestrator.management import (
    delete_experiment as orch_delete_experiment,
    kill_experiment as orch_kill_experiment,
    list_docker_containers as orch_list_docker_containers,
    read_task_output as orch_read_task_output,
    read_tasks as orch_read_tasks,
    restart_experiment as orch_restart_experiment,
)
from ..orchestrator.eval import (
    extract_metrics as eval_extract_metrics,
    has_result_event as eval_has_result_event,
    read_eval_results as eval_read_results,
    run_evaluate as eval_run_evaluate,
)
from ..orchestrator.queue import (
    collect_used_idea_names as queue_collect_used_idea_names,
    queue_idea as queue_queue_idea,
)
from ..orchestrator.service import OrchestratorService
from ..orchestrator.workspace import (
    analyst_reports_dir as ws_analyst_reports_dir,
    build_reference_code_section as ws_build_reference_code_section,
    copy_analyst_reports_to_workspace as ws_copy_analyst_reports_to_workspace,
    get_analyst_reports_summary as ws_get_analyst_reports_summary,
    prepare_analyst_workspace as ws_prepare_analyst_workspace,
    prepare_workspace as ws_prepare_workspace,
    resolve_base_solution as ws_resolve_base_solution,
)
from .api_views import (
    build_graph_payload as dashboard_build_graph_payload,
    build_state_payload as dashboard_build_state_payload,
    list_analyst_reports as dashboard_list_analyst_reports,
    list_experiment_files as dashboard_list_experiment_files,
    read_analyst_report as dashboard_read_analyst_report,
    read_events as dashboard_read_events,
    read_experiment_file as dashboard_read_experiment_file,
    submission_blob as dashboard_submission_blob,
)

load_dotenv()

AWITUNE_DIR = Path(__file__).resolve().parents[2]
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


def prepare_workspace(base_path, exp_dir, custom_prompt, best_score, prev_exps, reference_code=None):
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
def run_agent_in_thread(exp_name, prompt, base_solution, gpu_id, reference_code=None, task_type="experiment"):
    return orch.run_agent_in_thread(
        exp_name,
        prompt,
        base_solution,
        gpu_id,
        reference_code=reference_code,
        task_type=task_type,
    )


def _finalize_analysis(exp_name, exp_dir, out, elapsed, ec):
    return orch._finalize_analysis(exp_name, exp_dir, out, elapsed, ec)


def _finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp):
    return orch._finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp)


# ---- Auto-feeder ----
def _get_idea_feeder():
    return orch._get_idea_feeder()


# ---- Worker ----
def _collect_used_idea_names() -> set:
    return queue_collect_used_idea_names(rt)


def _queue_idea(idea: dict, idx: int) -> bool:
    return queue_queue_idea(rt, rt.cfg, idea, idx, resolve_base_solution, orchestrator_log)


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
    return jsonify(dashboard_build_state_payload(rt))


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
    stack_sources = d.get("stack_sources", [])
    if not isinstance(stack_sources, list):
        stack_sources = []
    base = resolve_base_solution(base_experiment)
    parent = base_experiment if base_experiment and base_experiment != "default" else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "analyst" if task_type == "analysis" else "agent"
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name or "")[:40] if name else ""
    eid = f"{prefix}_{safe_name}_{ts}" if safe_name else f"{prefix}_{ts}"
    db.create_experiment(eid, prompt=prompt, base_solution=base, parent_experiment=parent, task_type=task_type)
    db.add_log(eid, f"Queued {task_type}: {prompt[:100]}")
    item = {
        "id": eid,
        "prompt": prompt,
        "base_solution": base,
        "task_type": task_type,
        "stack_sources": [str(x).strip() for x in stack_sources if str(x).strip()],
    }
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


@app.route("/api/oof/<exp_name>", methods=["POST"])
def api_oof(exp_name):
    exp = db.get_experiment(exp_name)
    if not exp:
        return jsonify({"status": "error", "message": "Experiment not found"}), 404
    if exp.get("task_type") in ("analysis", "oof_fold"):
        return jsonify({"status": "error", "message": "OOF is available only for model experiments"}), 400
    result = orch.enqueue_oof_for_experiment(exp_name, manual=True, requester="manual_api")
    status = result.get("status")
    if status == "queued":
        if not rt.worker_running:
            start_worker()
        return jsonify(result), 200
    if status == "already_running":
        return jsonify(result), 409
    return jsonify(result), 400


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
    exp_dir = rt.cfg.experiments_dir / exp_name
    exp = db.get_experiment(exp_name) or {}
    if exp.get("task_type") == "oof_fold":
        candidates = [exp_dir / "oof_fold.log", exp_dir / "agent.log"]
    else:
        candidates = [exp_dir / "agent.log", exp_dir / "oof_fold.log"]
    for p in candidates:
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
    return jsonify(dashboard_read_events(rt, exp_name))


@app.route("/api/graph")
def api_graph():
    return jsonify(dashboard_build_graph_payload())


@app.route("/api/files/<exp_name>/<subdir>")
def api_files(exp_name, subdir):
    return jsonify(dashboard_list_experiment_files(rt, exp_name, subdir))


@app.route("/api/file/<exp_name>")
def api_file_content(exp_name):
    rel = request.args.get("path", "")
    if not rel:
        return Response("No path", status=400)
    file_data = dashboard_read_experiment_file(rt, exp_name, rel)
    if "error" in file_data:
        return Response(file_data["error"], status=file_data["status"])
    return Response(file_data["content"], mimetype="text/plain", status=file_data["status"])


@app.route("/api/download/<exp_name>/submission")
def api_download_submission(exp_name):
    payload = dashboard_submission_blob(rt, exp_name)
    if "error" in payload:
        return Response(payload["error"], status=payload["status"])
    return Response(
        payload["bytes"],
        mimetype="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="submission_{exp_name}.parquet"'},
    )


@app.route("/api/tasks/<exp_name>")
def api_tasks(exp_name):
    return jsonify(orch_read_tasks(rt, DOCKER_CMD, exp_name))


@app.route("/api/task_output/<exp_name>/<task_id>")
def api_task_output(exp_name, task_id):
    result = orch_read_task_output(DOCKER_CMD, exp_name, task_id)
    if "error" in result:
        return Response(result["error"], status=result["status"], mimetype="text/plain")
    return Response(result["content"], status=result["status"], mimetype="text/plain")


@app.route("/api/docker")
def api_docker():
    return jsonify(orch_list_docker_containers(DOCKER_CMD))


@app.route("/api/kill/<exp_name>", methods=["POST"])
def api_kill(exp_name):
    try:
        return jsonify(orch_kill_experiment(rt, DOCKER_CMD, exp_name))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/restart/<exp_name>", methods=["POST"])
def api_restart(exp_name):
    try:
        payload, status = orch_restart_experiment(rt, DOCKER_CMD, _force_rmtree, start_worker, exp_name)
        return jsonify(payload), status
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/delete/<exp_name>", methods=["POST"])
def api_delete(exp_name):
    try:
        return jsonify(orch_delete_experiment(rt, DOCKER_CMD, exp_name))
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
    return jsonify(dashboard_list_analyst_reports(rt))


@app.route("/api/analyst_reports/<name>")
def api_analyst_report(name):
    report = dashboard_read_analyst_report(rt, name)
    if "error" in report:
        return report["error"], report["status"]
    return report["content"], report["status"], {"Content-Type": "text/markdown; charset=utf-8"}


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
