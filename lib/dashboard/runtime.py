"""Runtime primitives for dashboard scheduling and orchestration."""

import re
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

from .. import db
from ..config import ProjectConfig

ORCHESTRATOR_LOG_PATH = Path("/tmp/awitune_orchestrator.log")
MAX_AUTO_QUEUE_SIZE = 5


def clean_output(text: str) -> str:
    """Remove control characters that break terminal-style logs."""
    text = text.replace("\r", "")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def orchestrator_log(message: str):
    """Append orchestrator message to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = clean_output(message)
    line = f"[{timestamp}] {message}\n"
    with open(ORCHESTRATOR_LOG_PATH, "a") as f:
        f.write(line)
    print(f"[orchestrator] {message}", flush=True)


def get_docker_cmd():
    for cmd in [["docker"], ["sudo", "docker"]]:
        try:
            r = subprocess.run(cmd + ["info"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return cmd
        except Exception:
            pass
    return ["docker"]


class RuntimeState:
    def __init__(self, docker_cmd: list[str]):
        self.docker_cmd = docker_cmd
        self.lock = threading.Lock()
        self.manual_queue = deque()
        self.auto_queue = deque()
        self.running_gpus = {}
        self.used_idea_names = set()
        self.proxy_proc = None
        self.worker_running = False
        self.worker_thread = None
        self.cfg: ProjectConfig = None

    def get_gpu_slots_used(self, gpu_id: int) -> int:
        return len(self.running_gpus.get(gpu_id, []))

    def get_available_gpu(self) -> int | None:
        slots_per_gpu = self.cfg.slots_per_gpu if self.cfg else 1
        for g in (self.cfg.gpus if self.cfg else [0]):
            if self.get_gpu_slots_used(g) < slots_per_gpu:
                return g
        return None

    def add_experiment_to_gpu(self, gpu_id: int, exp_name: str):
        if gpu_id not in self.running_gpus:
            self.running_gpus[gpu_id] = []
        if exp_name not in self.running_gpus[gpu_id]:
            self.running_gpus[gpu_id].append(exp_name)

    def remove_experiment_from_gpu(self, gpu_id: int, exp_name: str):
        if gpu_id in self.running_gpus:
            try:
                self.running_gpus[gpu_id].remove(exp_name)
                if not self.running_gpus[gpu_id]:
                    del self.running_gpus[gpu_id]
            except ValueError:
                pass

    def sync_running_from_docker(self):
        try:
            r = subprocess.run(
                self.docker_cmd + ["ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5
            )
            active_containers = set(r.stdout.strip().split("\n")) if r.stdout.strip() else set()
            zombies_to_kill = []
            with self.lock:
                rebuilt_running = {}
                for cn in active_containers:
                    if not cn:
                        continue
                    if not (cn.startswith("agent-") or cn.startswith("oof-")):
                        continue
                    parts = cn.rsplit("-gpu", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        gpu = int(parts[1])
                    except ValueError:
                        continue
                    if parts[0].startswith("agent-"):
                        exp_name = parts[0].replace("agent-", "", 1)
                    elif parts[0].startswith("oof-"):
                        exp_name = parts[0].replace("oof-", "", 1)
                    else:
                        continue
                    exp = db.get_experiment(exp_name)
                    if not exp:
                        zombies_to_kill.append(cn)
                    elif exp.get("status") in ("completed", "failed", "killed", "cancelled"):
                        zombies_to_kill.append(cn)
                    else:
                        rebuilt_running.setdefault(gpu, [])
                        if exp_name not in rebuilt_running[gpu]:
                            rebuilt_running[gpu].append(exp_name)
                self.running_gpus = rebuilt_running
            for cn in zombies_to_kill:
                try:
                    subprocess.run(self.docker_cmd + ["kill", cn], capture_output=True, timeout=10)
                except Exception:
                    pass
        except Exception:
            pass
