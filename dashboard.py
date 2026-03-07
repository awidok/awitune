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
from collections import deque
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from . import db
from .config import ProjectConfig

load_dotenv()

AWITUNE_DIR = Path(__file__).resolve().parent
PROXY_HOST = os.getenv("PROXY_HOST", "localhost")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = Flask(__name__, template_folder=str(AWITUNE_DIR / "templates"))
CORS(app)


# ---- Orchestrator log ----
ORCHESTRATOR_LOG_PATH = Path("/tmp/awitune_orchestrator.log")


def orchestrator_log(message: str):
    """Log orchestrator activity to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with open(ORCHESTRATOR_LOG_PATH, "a") as f:
        f.write(line)
    print(f"[orchestrator] {message}")


# ---- Global runtime state ----
class RuntimeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.manual_queue = deque()
        self.auto_queue = deque()
        self.running_gpus = {}
        self.auto_feed_enabled = True
        self.used_idea_names = set()
        self.proxy_proc = None
        self.worker_running = False
        self.worker_thread = None
        self.cfg: ProjectConfig = None

    def sync_running_from_docker(self):
        try:
            r = subprocess.run(
                DOCKER_CMD + ["ps", "--filter", "name=agent-", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5)
            active_containers = set(r.stdout.strip().split("\n")) if r.stdout.strip() else set()
            zombies_to_kill = []
            with self.lock:
                for cn in active_containers:
                    if not cn:
                        continue
                    parts = cn.rsplit("-gpu", 1)
                    if len(parts) == 2:
                        try:
                            gpu = int(parts[1])
                            exp_name = parts[0].replace("agent-", "", 1)
                            if gpu not in self.running_gpus:
                                exp = db.get_experiment(exp_name)
                                if exp and exp.get("status") in ("completed", "failed", "killed", "cancelled"):
                                    zombies_to_kill.append(cn)
                                else:
                                    self.running_gpus[gpu] = exp_name
                        except ValueError:
                            pass
            for cn in zombies_to_kill:
                try:
                    subprocess.run(DOCKER_CMD + ["kill", cn], capture_output=True, timeout=10)
                except Exception:
                    pass
        except Exception:
            pass


rt = RuntimeState()


# ---- Docker ----
def get_docker_cmd():
    for cmd in [["docker"], ["sudo", "docker"]]:
        try:
            r = subprocess.run(cmd + ["info"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return cmd
        except Exception:
            pass
    return ["docker"]


DOCKER_CMD = get_docker_cmd()


def _force_rmtree(path: Path):
    try:
        shutil.rmtree(path)
    except PermissionError:
        subprocess.run(["chmod", "-R", "u+rwX", str(path)], capture_output=True, timeout=30)
        shutil.rmtree(path, ignore_errors=True)


# ---- Proxy ----
def start_proxy():
    if rt.proxy_proc and rt.proxy_proc.poll() is None:
        return
    venv_py = Path(os.environ.get("VIRTUAL_ENV", "")) / "bin" / "python3"
    py = str(venv_py) if venv_py.exists() else "python3"
    env = os.environ.copy()
    env["PROXY_HOST"] = PROXY_HOST
    env["PROXY_PORT"] = str(rt.cfg.proxy_port)
    rt.proxy_proc = subprocess.Popen(
        [py, str(AWITUNE_DIR / "proxy.py")], cwd=str(AWITUNE_DIR), env=env,
        stdout=open("/tmp/awitune_proxy.log", "a"), stderr=subprocess.STDOUT)
    time.sleep(2)


def stop_proxy():
    if rt.proxy_proc and rt.proxy_proc.poll() is None:
        rt.proxy_proc.terminate()
        try:
            rt.proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rt.proxy_proc.kill()
    rt.proxy_proc = None


# ---- Workspace ----
def resolve_base_solution(base_experiment: str) -> str:
    if not base_experiment or base_experiment == "default":
        return str(rt.cfg.solutions_dir / "baseline")
    exp = db.get_experiment(base_experiment)
    if exp and exp.get("workspace_dir"):
        ws_path = Path(exp["workspace_dir"])
        if ws_path.is_dir() and (ws_path / "run.py").exists():
            return str(ws_path)
    return str(rt.cfg.solutions_dir / "baseline")


def build_reference_code_section(reference_code: dict) -> str:
    if not reference_code:
        return ""
    ref_exp_name = reference_code.get("experiment", "")
    what_to_take = reference_code.get("what_to_take", "")
    if not ref_exp_name:
        return ""

    if rt.cfg.reference_dir:
        ref_dir = rt.cfg.reference_dir / ref_exp_name
        if ref_dir.is_dir():
            files = reference_code.get("files")
            if not files:
                files = [f.name for f in sorted(ref_dir.glob("*.py"))]
            section = f"\n\n## Reference Code: {ref_exp_name}\n"
            if what_to_take:
                section += f"**Specifically**: {what_to_take}\n"
            for fname in files:
                fpath = ref_dir / fname
                if fpath.exists():
                    code = fpath.read_text(errors="replace")
                    if len(code) > 20000:
                        code = code[:20000] + "\n# ... (truncated) ...\n"
                    section += f"\n### {fname}\n```python\n{code}\n```\n"
            return section

    ref_exp = db.get_experiment(ref_exp_name)
    if not ref_exp or not ref_exp.get("workspace_dir"):
        return ""
    ref_run_py = Path(ref_exp["workspace_dir"]) / "run.py"
    if not ref_run_py.exists():
        return ""
    ref_code = ref_run_py.read_text(errors="replace")
    ref_score = ref_exp.get("test_score", "?")
    section = f"\n\n## Reference Code (from experiment {ref_exp_name}, score {ref_score})\n"
    if what_to_take:
        section += f"**Specifically**: {what_to_take}\n"
    section += f"\n```python\n{ref_code}\n```\n"
    return section


def _analyst_reports_dir():
    return rt.cfg.data_dir / "analyst_reports"


def get_analyst_reports_summary() -> str:
    d = _analyst_reports_dir()
    if not d.exists():
        return "No previous analysis has been done yet."
    reports = sorted(d.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not reports:
        return "No previous analysis has been done yet."
    parts = []
    for r in reports[:5]:
        content = r.read_text(errors="replace")
        if len(content) > 3000:
            content = content[:3000] + "\n... (truncated) ...\n"
        parts.append(f"### {r.stem}\n{content}\n")
    return "\n".join(parts)


def copy_analyst_reports_to_workspace(ws: Path):
    d = _analyst_reports_dir()
    if not d.exists():
        return
    reports = sorted(d.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not reports:
        return
    dest = ws / "analyst_reports"
    dest.mkdir(exist_ok=True)
    for r in reports[:10]:
        shutil.copy2(r, dest / r.name)
    for j in d.glob("*.json"):
        shutil.copy2(j, dest / j.name)
    os.chmod(dest, 0o777)
    for f in dest.iterdir():
        os.chmod(f, 0o666)


def prepare_analyst_workspace(exp_dir, analysis_focus, prev_exps):
    ws = exp_dir / "workspace"
    if ws.exists():
        subprocess.run(["chmod", "-R", "u+rwX", str(ws)], capture_output=True, timeout=30)
    ws.mkdir(parents=True, exist_ok=True)
    os.chmod(ws, 0o777)
    copy_analyst_reports_to_workspace(ws)

    if rt.cfg.analyst_prompt and rt.cfg.analyst_prompt.exists():
        tpl = rt.cfg.analyst_prompt.read_text()
        prompt = tpl.replace("{ANALYSIS_FOCUS}", analysis_focus or "General dataset exploration")
        prompt = prompt.replace("{PREVIOUS_ANALYSIS}", get_analyst_reports_summary())
    else:
        prompt = f"# Analysis Task\n\n{analysis_focus}\n"

    (exp_dir / "CLAUDE.md").write_text(prompt)
    od = exp_dir / "output"
    od.mkdir(exist_ok=True)
    os.chmod(od, 0o777)
    return ws


def prepare_workspace(base_path, exp_dir, custom_prompt, best_score, prev_exps,
                      reference_code=None):
    ws = exp_dir / "workspace"
    if ws.exists():
        subprocess.run(["chmod", "-R", "u+rwX", str(ws)], capture_output=True, timeout=30)
    ws.mkdir(parents=True, exist_ok=True)
    base = Path(base_path)
    if base.is_dir():
        for f in base.iterdir():
            if f.is_file():
                shutil.copy2(f, ws / f.name)

    project_readme = rt.cfg.project_dir / "README.md"
    if project_readme.exists():
        shutil.copy2(project_readme, ws / "README.md")

    os.chmod(ws, 0o777)
    for f in ws.iterdir():
        os.chmod(f, 0o666)

    exp_info = f"# Experiment: {exp_dir.name}\n\n"
    exp_info += f"**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    exp_info += f"**Base solution**: {base_path}\n"
    exp_info += f"**Best score at start**: {best_score:.6f}\n\n"
    if custom_prompt:
        exp_info += f"## Custom Prompt\n\n{custom_prompt}\n\n"
    (exp_dir / "EXPERIMENT_INFO.md").write_text(exp_info)

    prev_md = f"# Previous Experiments\n\nBest score: **{best_score:.6f}**\n\n"
    if prev_exps:
        prev_md += "| # | Name | Score | Notes |\n|---|------|-------|-------|\n"
        for i, e in enumerate(prev_exps[-20:]):
            score = e.get("test_score") or "?"
            prev_md += f"| {i+1} | {e.get('name','')} | {score} | {e.get('notes','')} |\n"
    (ws / "prev_experiments.md").write_text(prev_md)

    copy_analyst_reports_to_workspace(ws)

    tpl = rt.cfg.agent_prompt.read_text()
    prompt = tpl.replace("{BEST_SCORE}", f"{best_score:.6f}")
    if custom_prompt:
        prompt += f"\n\n## SPECIFIC TASK FOR THIS RUN\n{custom_prompt}\n"
    ref_section = build_reference_code_section(reference_code)
    if ref_section:
        prompt += ref_section
    if reference_code and reference_code.get("experiment") and rt.cfg.reference_dir:
        ref_dir = rt.cfg.reference_dir / reference_code["experiment"]
        if ref_dir.is_dir():
            ref_dest = ws / "reference"
            ref_dest.mkdir(exist_ok=True)
            for f in ref_dir.glob("*.py"):
                shutil.copy2(f, ref_dest / f.name)
            os.chmod(ref_dest, 0o777)
            for f in ref_dest.iterdir():
                os.chmod(f, 0o666)
    (exp_dir / "CLAUDE.md").write_text(prompt)

    od = exp_dir / "output"
    od.mkdir(exist_ok=True)
    os.chmod(od, 0o777)
    return ws


# ---- Evaluation helpers ----
def _run_evaluate(out: Path, log_path: Path = None):
    """Run the project's evaluate.py on an output directory."""
    venv_py = Path(os.environ.get("VIRTUAL_ENV", "")) / "bin" / "python3"
    py = str(venv_py) if venv_py.exists() else "python3"
    result = subprocess.run(
        [py, str(rt.cfg.evaluate_script), str(out), "--data-dir", str(rt.cfg.data_dir)],
        capture_output=True, text=True, timeout=300
    )
    if log_path:
        eval_out = (result.stdout or "").strip() + "\n" + (result.stderr or "").strip()
        if eval_out.strip():
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write("\n\n=== evaluate.py (host) ===\n")
                lf.write(eval_out)
                lf.write("\n")
    return result


def _read_eval_results(out: Path) -> dict:
    ep = out / "eval_results.json"
    if ep.exists():
        try:
            return json.loads(ep.read_text())
        except Exception:
            pass
    return {}


def _extract_metrics(ev: dict) -> tuple[float, float]:
    """Extract test and val scores from eval_results using config metric keys."""
    test = ev.get(rt.cfg.test_metric_key, 0) or 0
    val = ev.get(rt.cfg.val_metric_key, 0) or 0
    return test, val


# ---- Agent runner ----
def run_agent_in_thread(exp_name, prompt, base_solution, gpu_id,
                        reference_code=None, task_type="experiment"):
    cfg = rt.cfg
    exp_dir = cfg.experiments_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    is_analysis = task_type == "analysis"

    db.update_experiment(exp_name, status="running", gpu_id=gpu_id,
                         started_at=datetime.now().isoformat(),
                         exp_dir=str(exp_dir))
    db.add_log(exp_name, f"Starting {'analysis' if is_analysis else 'experiment'} on GPU {gpu_id}")

    t0 = time.time()
    try:
        start_proxy()
        direction = cfg.best_score_sort_key()
        stats = db.get_stats(direction)
        best_score = stats.get("best_score", 0) or 0
        prev_exps = db.get_all_experiments(limit=20, status="completed")

        if is_analysis:
            ws = prepare_analyst_workspace(exp_dir, prompt, prev_exps)
        else:
            ws = prepare_workspace(base_solution or str(cfg.solutions_dir / "baseline"),
                                   exp_dir, prompt, best_score, prev_exps,
                                   reference_code=reference_code)
        db.update_experiment(exp_name, workspace_dir=str(ws), output_dir=str(exp_dir / "output"))

        cn = f"agent-{exp_name}-gpu{gpu_id}"
        db.update_experiment(exp_name, container_name=cn)
        out = exp_dir / "output"

        proxy_port = cfg.proxy_port
        cmd = DOCKER_CMD + [
            "run", "--rm", "--network=host", "--name", cn,
            "--gpus", f'device={gpu_id}',
            "--ulimit", "core=0",
            "-v", f"{cfg.data_dir}:/app/data:ro",
            "-v", f"{ws}:/app/workspace",
            "-v", f"{out}:/app/output",
            "-v", f"{exp_dir/'CLAUDE.md'}:/app/CLAUDE.md:ro",
            "-v", "/etc/ssl/certs:/etc/ssl/certs:ro",
            "-v", "/etc/pki:/etc/pki:ro",
            "-e", f"PROXY_HOST={PROXY_HOST}", "-e", f"PROXY_PORT={proxy_port}",
            "-e", f"OPENAI_API_KEY={OPENAI_API_KEY}",
            "-e", f"ANTHROPIC_BASE_URL=http://{PROXY_HOST}:{proxy_port}",
            "-e", f"ANTHROPIC_API_KEY={OPENAI_API_KEY or 'dummy-key'}",
            "-e", f"ANTHROPIC_AUTH_TOKEN={OPENAI_API_KEY or 'dummy-key'}",
            "-e", "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt",
            "-e", "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
            "-e", "NODE_TLS_REJECT_UNAUTHORIZED=0",
            "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "CUDA_DEVICE=cuda:0",
            "-e", f"MAX_TURNS={cfg.max_turns}", cfg.docker_image,
        ]

        db.add_log(exp_name, f"Docker command: {' '.join(cmd[:10])}...")
        lp = exp_dir / "agent.log"
        events_file = out / "events.jsonl"
        GRACE_PERIOD = 120
        with open(lp, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
            deadline = time.time() + cfg.timeout_minutes * 60
            agent_done_at = None
            ec = None
            while True:
                ret = proc.poll()
                if ret is not None:
                    ec = ret
                    break
                if time.time() > deadline:
                    subprocess.run(DOCKER_CMD + ["kill", cn], capture_output=True)
                    proc.wait(timeout=30)
                    ec = -1
                    db.add_log(exp_name, "TIMEOUT — container killed", level="error")
                    break
                if agent_done_at is None and events_file.exists():
                    try:
                        with open(events_file, "rb") as ef:
                            ef.seek(0, 2)
                            sz = ef.tell()
                            ef.seek(max(0, sz - 8192))
                            tail = ef.read().decode("utf-8", errors="ignore")
                        if '"type":"result"' in tail:
                            agent_done_at = time.time()
                            db.add_log(exp_name, "Agent finished, waiting for container to exit...")
                    except Exception:
                        pass
                if agent_done_at and time.time() - agent_done_at > GRACE_PERIOD:
                    db.add_log(exp_name, "Container stuck after agent finished — killing", level="warning")
                    subprocess.run(DOCKER_CMD + ["kill", cn], capture_output=True)
                    proc.wait(timeout=30)
                    ec = proc.returncode
                    break
                time.sleep(10)

        elapsed = time.time() - t0
        db.add_log(exp_name, f"Finished in {elapsed/60:.1f}min, exit={ec}")

        if is_analysis:
            _finalize_analysis(exp_name, exp_dir, out, elapsed, ec)
        else:
            _finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp)

    except Exception as e:
        db.update_experiment(exp_name,
                             status="failed",
                             finished_at=datetime.now().isoformat(),
                             exit_code=-2,
                             elapsed_min=round((time.time() - t0) / 60, 2),
                             notes=f"ERROR: {e}")
        db.add_log(exp_name, f"ERROR: {e}", level="error")
    finally:
        with rt.lock:
            rt.running_gpus.pop(gpu_id, None)


def _finalize_analysis(exp_name, exp_dir, out, elapsed, ec):
    reports_dir = _analyst_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_src = out / "analysis_report.md"
    data_src = out / "analysis_data.json"
    if report_src.exists():
        dest = reports_dir / f"{exp_name}.md"
        shutil.copy2(report_src, dest)
        db.add_log(exp_name, f"Analysis report saved to {dest}", level="success")
    if data_src.exists():
        shutil.copy2(data_src, reports_dir / f"{exp_name}.json")

    has_report = report_src.exists()
    db.update_experiment(exp_name,
                         status="completed",
                         finished_at=datetime.now().isoformat(),
                         exit_code=ec,
                         elapsed_min=round(elapsed / 60, 2),
                         notes="analysis complete" if has_report else "analysis finished (no report)")


def _finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp):
    _run_evaluate(out, lp)
    ev = _read_eval_results(out)
    test, val = _extract_metrics(ev)

    improved = False
    notes = "no improvement"
    if test and rt.cfg.is_better(test, best_score):
        improved = True
        notes = f"IMPROVED {best_score:.6f} → {test:.6f}"
        db.set_global("best_score", test)
        db.set_global("best_experiment", str(exp_dir / "workspace"))
        db.add_log(exp_name, f"★ {notes}", level="success")

    db.update_experiment(exp_name,
                         status="completed",
                         finished_at=datetime.now().isoformat(),
                         exit_code=ec,
                         elapsed_min=round(elapsed / 60, 2),
                         test_score=test if test else None,
                         val_score=val if val else None,
                         improved=1 if improved else 0,
                         notes=notes,
                         eval_json=json.dumps(ev))


# ---- Auto-feeder ----
def _get_idea_feeder():
    try:
        from . import idea_feeder
        return idea_feeder
    except ImportError:
        return None


def refill_auto_queue():
    if not rt.auto_feed_enabled:
        return
    with rt.lock:
        if len(rt.auto_queue) > 0:
            return

    used_idea_names = set()
    all_exps = db.get_all_experiments(limit=2000)
    for e in all_exps:
        exp_name = e.get("name", "")
        if exp_name.startswith("auto_"):
            parts = exp_name[5:]
            segments = parts.rsplit("_", 2)
            if len(segments) >= 3:
                used_idea_names.add(segments[0])

    with rt.lock:
        for item in rt.manual_queue:
            if item.get("idea_name"):
                used_idea_names.add(item["idea_name"])
        for item in rt.auto_queue:
            if item.get("idea_name"):
                used_idea_names.add(item["idea_name"])
        rt.used_idea_names = used_idea_names

    feeder = _get_idea_feeder()
    if not feeder:
        orchestrator_log("No feeder available")
        return
    
    orchestrator_log(f"Getting ideas (used: {len(used_idea_names)})...")
    unused = feeder.get_unused_prompts(used_idea_names)
    orchestrator_log(f"Got {len(unused)} idea(s)")
    if unused:
        idea = unused[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        eid = f"auto_{idea['name']}_{ts}"
        base_exp = idea.get("base_experiment", "default")
        base_sol = resolve_base_solution(base_exp)
        parent = base_exp if base_exp and base_exp != "default" else ""
        idea_task_type = idea.get("task_type", "experiment")
        db.create_experiment(eid, prompt=idea["prompt"][:500], base_solution=base_sol,
                             parent_experiment=parent, task_type=idea_task_type)
        db.add_log(eid, f"Auto-queued {idea_task_type}: {idea['name']}")
        orchestrator_log(f"Queued: {eid} ({idea_task_type}) - {idea.get('reasoning', '')[:100]}")
        item = {"id": eid, "prompt": idea["prompt"], "base_solution": base_sol,
                "idea_name": idea["name"], "auto": True,
                "reference_code": idea.get("reference_code"),
                "task_type": idea_task_type}
        with rt.lock:
            rt.auto_queue.append(item)


# ---- Worker ----
def worker_loop():
    while rt.worker_running:
        item = None
        with rt.lock:
            if rt.manual_queue:
                item = rt.manual_queue.popleft()
            elif rt.auto_queue:
                item = rt.auto_queue.popleft()

        if item is None:
            refill_auto_queue()
            time.sleep(2)
            continue

        with rt.lock:
            gpu = None
            used = set(rt.running_gpus.keys())
            for g in rt.cfg.gpus:
                if g not in used:
                    gpu = g
                    break
            if gpu is None:
                if item.get("auto"):
                    rt.auto_queue.appendleft(item)
                else:
                    rt.manual_queue.appendleft(item)
            else:
                rt.running_gpus[gpu] = item["id"]

        if gpu is None:
            time.sleep(2)
            continue

        threading.Thread(
            target=run_agent_in_thread,
            args=(item["id"], item.get("prompt", ""), item.get("base_solution", ""), gpu),
            kwargs={
                "reference_code": item.get("reference_code"),
                "task_type": item.get("task_type", "experiment"),
            },
            daemon=True
        ).start()
        time.sleep(1)


def start_worker():
    if rt.worker_running:
        return
    rt.worker_running = True
    rt.worker_thread = threading.Thread(target=worker_loop, daemon=True)
    rt.worker_thread.start()


def stop_worker():
    rt.worker_running = False


# ---- Routes ----

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    rt.sync_running_from_docker()
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

    return jsonify({
        "project_name": rt.cfg.name,
        "metric": rt.cfg.test_metric_key,
        "queue": list(rt.manual_queue) + [{"auto": True, **i} for i in rt.auto_queue],
        "auto_feed_enabled": rt.auto_feed_enabled,
        "auto_queue_size": len(rt.auto_queue),
        "ideas_total": -1,  # No longer using static ideas
        "ideas_used": len(rt.used_idea_names),
        "running": {str(k): v for k, v in rt.running_gpus.items()},
        "history": exps,
        "best_score": stats.get("best_score", 0) or 0,
        "best_experiment": stats.get("best_experiment", ""),
        "available_gpus": rt.cfg.gpus,
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


@app.route("/api/autofeed/toggle", methods=["POST"])
def api_autofeed_toggle():
    rt.auto_feed_enabled = not rt.auto_feed_enabled
    return jsonify({"auto_feed_enabled": rt.auto_feed_enabled})


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
                r = subprocess.run(DOCKER_CMD + ["exec", cn, "ls", "-1", tasks_dir],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    for fname in r.stdout.strip().split("\n"):
                        fname = fname.strip()
                        if fname and fname.endswith(".output"):
                            tid = fname.replace(".output", "")
                            task_files.append(tid)
                            if tid not in tasks:
                                tasks[tid] = {"id": tid, "description": "background task", "turn": "?"}
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
            for k in [k for k, v in rt.running_gpus.items() if v == exp_name]:
                del rt.running_gpus[k]
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
            for k in [k for k, v in rt.running_gpus.items() if v == exp_name]:
                del rt.running_gpus[k]
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
    if not events_file.exists():
        return False
    try:
        with open(events_file, "rb") as ef:
            ef.seek(0, 2)
            sz = ef.tell()
            ef.seek(max(0, sz - 8192))
            tail = ef.read().decode("utf-8", errors="ignore")
        return '"type":"result"' in tail
    except Exception:
        return False


def _find_container_name(exp_name, active_containers):
    for cn in active_containers:
        if exp_name in cn:
            return cn
    return None


def _finalize_experiment(exp_name, best_score):
    cfg = rt.cfg
    exp_dir = cfg.experiments_dir / exp_name
    out = exp_dir / "output"

    elapsed_min = None
    exp = db.get_experiment(exp_name)
    if exp and exp.get("started_at"):
        try:
            started = datetime.fromisoformat(exp["started_at"])
            elapsed_min = round((datetime.now() - started).total_seconds() / 60, 2)
        except Exception:
            pass

    task_type = (exp.get("task_type") or "experiment") if exp else "experiment"
    if task_type == "analysis":
        _finalize_analysis(exp_name, exp_dir, out, (elapsed_min or 0) * 60, None)
        db.add_log(exp_name, "Recovered analyst task after restart")
        return "completed", 0

    if out.exists():
        _run_evaluate(out)

    ev = _read_eval_results(out)
    test, val = _extract_metrics(ev)

    improved = False
    notes = "recovered after restart"
    if test and cfg.is_better(test, best_score):
        improved = True
        notes = f"IMPROVED (recovered)"
        db.set_global("best_score", test)
        db.set_global("best_experiment", str(exp_dir / "workspace"))

    has_output = out.exists() and any(out.iterdir()) if out.exists() else False
    status = "completed" if has_output else "failed"
    db.update_experiment(exp_name,
                         status=status,
                         finished_at=datetime.now().isoformat(),
                         elapsed_min=elapsed_min,
                         test_score=test if test else None,
                         val_score=val if val else None,
                         improved=1 if improved else 0,
                         notes=notes,
                         eval_json=json.dumps(ev) if ev else None)
    db.add_log(exp_name, f"Recovered: {status}, test={test or 'N/A'}")
    return status, test


def _monitor_orphaned_container(exp_name, container_name, gpu_id):
    GRACE_PERIOD = 120
    events_file = rt.cfg.experiments_dir / exp_name / "output" / "events.jsonl"
    agent_done_at = None
    db.add_log(exp_name, f"Resumed monitoring (container {container_name})")
    while True:
        try:
            r = subprocess.run(
                DOCKER_CMD + ["inspect", container_name, "--format", "{{.State.Running}}"],
                capture_output=True, text=True, timeout=5)
            if r.stdout.strip() != "true":
                break
        except Exception:
            break
        if agent_done_at is None and _check_agent_finished(events_file):
            agent_done_at = time.time()
        if agent_done_at and time.time() - agent_done_at > GRACE_PERIOD:
            subprocess.run(DOCKER_CMD + ["kill", container_name], capture_output=True)
            break
        time.sleep(10)

    direction = rt.cfg.best_score_sort_key()
    best_score = db.get_stats(direction).get("best_score", 0) or 0
    _finalize_experiment(exp_name, best_score)
    with rt.lock:
        rt.running_gpus.pop(gpu_id, None)


def recover_orphaned_experiments():
    try:
        r = subprocess.run(DOCKER_CMD + ["ps", "--filter", "name=agent-", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=5)
        active_containers = set(r.stdout.strip().split("\n")) if r.stdout.strip() else set()
    except Exception:
        active_containers = set()

    running_exps = db.get_all_experiments(limit=200, status="running")
    if not running_exps:
        return

    direction = rt.cfg.best_score_sort_key()
    best_score = db.get_stats(direction).get("best_score", 0) or 0
    recovered = monitored = 0
    for exp in running_exps:
        exp_name = exp["name"]
        gpu_id = exp.get("gpu_id")
        cn = _find_container_name(exp_name, active_containers)
        if cn:
            events_file = rt.cfg.experiments_dir / exp_name / "output" / "events.jsonl"
            if _check_agent_finished(events_file):
                subprocess.run(DOCKER_CMD + ["kill", cn], capture_output=True)
                time.sleep(2)
                _finalize_experiment(exp_name, best_score)
                recovered += 1
            else:
                if gpu_id is not None:
                    with rt.lock:
                        rt.running_gpus[gpu_id] = exp_name
                threading.Thread(target=_monitor_orphaned_container,
                                 args=(exp_name, cn, gpu_id), daemon=True).start()
                monitored += 1
        else:
            _finalize_experiment(exp_name, best_score)
            recovered += 1
    if recovered:
        print(f"  Recovered {recovered} orphaned experiment(s)")
    if monitored:
        print(f"  Resumed monitoring for {monitored} experiment(s)")


# ---- Main init (called from cli.py) ----
def init_app(cfg: ProjectConfig):
    """Initialize the dashboard with a project config."""
    rt.cfg = cfg
    db.configure(cfg.experiments_dir)
    db.init_db()
