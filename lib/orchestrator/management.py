"""Orchestrator management helpers used by dashboard routes."""

import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

from .. import db


def find_claude_tasks_dir(docker_cmd, container_name):
    try:
        result = subprocess.run(
            docker_cmd + [
                "exec",
                container_name,
                "sh",
                "-c",
                "ls -d /tmp/claude-*/-app-workspace/tasks 2>/dev/null | head -1",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def read_tasks(rt, docker_cmd, exp_name):
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
                            tasks[task_id] = {
                                "id": task_id,
                                "description": f"task {task_id}",
                                "turn": ev.get("turn", "?"),
                            }
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass

    exp = db.get_experiment(exp_name)
    container_name = exp.get("container_name", "") if exp else ""
    task_files = []
    if container_name:
        try:
            tasks_dir = find_claude_tasks_dir(docker_cmd, container_name)
            if tasks_dir:
                cmd = docker_cmd + [
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    f"for f in {tasks_dir}/*.output; do stat -c '%n|%s|%Y' \"$f\"; done",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parts = line.split("|")
                            if len(parts) >= 3:
                                fname = parts[0]
                                task_id = fname.replace(".output", "").split("/")[-1]
                                size = parts[1]
                                timestamp = int(parts[2])
                                utc_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                                local_dt = utc_dt.astimezone(timezone(timedelta(hours=3)))
                                modified = local_dt.strftime("%b %d %H:%M")
                                task_files.append({
                                    "id": task_id,
                                    "size": size,
                                    "modified": modified,
                                    "timestamp": timestamp,
                                })
                                if task_id not in tasks:
                                    tasks[task_id] = {"id": task_id, "description": "background task", "turn": "?"}
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
    return {
        "tasks": sorted(tasks.values(), key=lambda t: str(t.get("turn", ""))),
        "task_files": task_files,
        "container": container_name,
    }


def read_task_output(docker_cmd, exp_name, task_id):
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", task_id)
    if not safe_id:
        return {"error": "Invalid task_id", "status": 400}
    exp = db.get_experiment(exp_name)
    container_name = exp.get("container_name", "") if exp else ""
    if not container_name:
        return {"content": "No container found", "status": 200}
    tasks_dir = find_claude_tasks_dir(docker_cmd, container_name)
    if not tasks_dir:
        return {"content": "Could not find tasks directory", "status": 200}
    try:
        result = subprocess.run(
            docker_cmd + ["exec", container_name, "cat", f"{tasks_dir}/{safe_id}.output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {"content": result.stdout[-200_000:], "status": 200}
        return {"content": f"Not found: {result.stderr.strip()}", "status": 200}
    except subprocess.TimeoutExpired:
        return {"content": "Timeout", "status": 200}
    except Exception as exc:
        return {"content": f"Error: {exc}", "status": 200}


def list_docker_containers(docker_cmd):
    containers = []
    try:
        result = subprocess.run(
            docker_cmd + ["ps", "--filter", "name=agent-", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split("\t", 1)
                containers.append({"name": parts[0], "status": parts[1] if len(parts) > 1 else "?"})
    except Exception:
        pass
    return {"containers": containers}


def kill_experiment(rt, docker_cmd, exp_name):
    for gpu in rt.cfg.gpus:
        subprocess.run(docker_cmd + ["kill", f"agent-{exp_name}-gpu{gpu}"], capture_output=True, timeout=5)
    db.update_experiment(exp_name, status="killed", notes="Manually killed")
    db.add_log(exp_name, "Manually killed", level="warning")
    return {"status": "killed"}


def restart_experiment(rt, docker_cmd, force_rmtree, start_worker, exp_name):
    exp = db.get_experiment(exp_name)
    if not exp:
        return {"status": "error", "message": "Not found"}, 404

    for gpu in rt.cfg.gpus:
        subprocess.run(docker_cmd + ["kill", f"agent-{exp_name}-gpu{gpu}"], capture_output=True, timeout=5)

    with rt.lock:
        for gpu_id in list(rt.running_gpus.keys()):
            rt.remove_experiment_from_gpu(gpu_id, exp_name)

    exp_dir = rt.cfg.experiments_dir / exp_name
    for subdir in ["output", "workspace"]:
        dpath = exp_dir / subdir
        if dpath.exists():
            force_rmtree(dpath)
        dpath.mkdir(parents=True, exist_ok=True)

    prompt = exp.get("prompt", "")
    base = exp.get("base_solution", "") or str(rt.cfg.solutions_dir / "baseline")
    db.update_experiment(
        exp_name,
        status="queued",
        started_at=None,
        finished_at=None,
        exit_code=None,
        elapsed_min=None,
        test_score=None,
        val_score=None,
        improved=0,
        notes="",
        eval_json=None,
    )
    db.add_log(exp_name, "Restarted")
    with rt.lock:
        rt.manual_queue.append({"id": exp_name, "prompt": prompt, "base_solution": base})
    if not rt.worker_running:
        start_worker()
    return {"status": "queued", "id": exp_name}, 200


def delete_experiment(rt, docker_cmd, exp_name):
    for gpu in rt.cfg.gpus:
        subprocess.run(docker_cmd + ["kill", f"agent-{exp_name}-gpu{gpu}"], capture_output=True, timeout=5)
    with rt.lock:
        for gpu_id in list(rt.running_gpus.keys()):
            rt.remove_experiment_from_gpu(gpu_id, exp_name)
    db.delete_experiment(exp_name)
    exp_dir = rt.cfg.experiments_dir / exp_name
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
    return {"status": "deleted"}
