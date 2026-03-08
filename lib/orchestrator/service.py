"""Core orchestrator service (worker, runner, recovery)."""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from .. import db
from ..dashboard.runtime import MAX_AUTO_QUEUE_SIZE
from ..dashboard.proxy import start_proxy as runtime_start_proxy, stop_proxy as runtime_stop_proxy
from ..notifications import send_telegram_notification
from .eval import extract_metrics, has_result_event, read_eval_results, run_evaluate
from .queue import collect_used_idea_names, queue_idea
from .workspace import analyst_reports_dir, prepare_analyst_workspace, prepare_workspace, resolve_base_solution


class OrchestratorService:
    def __init__(self, rt, docker_cmd, awitune_dir: Path, proxy_host: str, openai_api_key: str, orchestrator_log):
        self.rt = rt
        self.docker_cmd = docker_cmd
        self.awitune_dir = awitune_dir
        self.proxy_host = proxy_host
        self.openai_api_key = openai_api_key
        self.orchestrator_log = orchestrator_log

    def _notify(self, message: str, parse_mode: str = "HTML"):
        send_telegram_notification(self.rt.cfg, message, parse_mode=parse_mode)

    def trim_auto_queue(self, max_size: int = MAX_AUTO_QUEUE_SIZE) -> int:
        removed = []
        with self.rt.lock:
            while len(self.rt.auto_queue) > max_size:
                removed.append(self.rt.auto_queue.pop())

        for item in removed:
            exp_id = item.get("id", "")
            if not exp_id:
                continue
            db.update_experiment(exp_id, status="cancelled", notes=f"Auto queue trimmed to {max_size}")
            db.add_log(exp_id, f"Removed from auto-queue (limit {max_size})", level="warning")

        if removed:
            self.orchestrator_log(f"Trimmed auto queue: removed {len(removed)} item(s), kept {max_size}")
        return len(removed)

    def start_proxy(self):
        runtime_start_proxy(self.rt, self.awitune_dir, self.proxy_host)

    def stop_proxy(self):
        runtime_stop_proxy(self.rt)

    def resolve_base_solution(self, base_experiment: str) -> str:
        return resolve_base_solution(self.rt.cfg, base_experiment)

    def _force_rmtree(self, path: Path):
        try:
            shutil.rmtree(path)
        except PermissionError:
            subprocess.run(["chmod", "-R", "u+rwX", str(path)], capture_output=True, timeout=30)
            shutil.rmtree(path, ignore_errors=True)

    def run_agent_in_thread(self, exp_name, prompt, base_solution, gpu_id, reference_code=None, task_type="experiment"):
        cfg = self.rt.cfg
        exp_dir = cfg.experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        is_analysis = task_type == "analysis"

        db.update_experiment(exp_name, status="running", gpu_id=gpu_id, started_at=datetime.now().isoformat(), exp_dir=str(exp_dir))
        db.add_log(exp_name, f"Starting {'analysis' if is_analysis else 'experiment'} on GPU {gpu_id}")

        t0 = time.time()
        try:
            self.start_proxy()
            direction = cfg.best_score_sort_key()
            stats = db.get_stats(direction)
            best_score = stats.get("best_score", 0) or 0
            prev_exps = db.get_all_experiments(limit=20, status="completed")

            if is_analysis:
                ws = prepare_analyst_workspace(cfg, exp_dir, prompt)
            else:
                ws = prepare_workspace(
                    cfg,
                    base_solution or str(cfg.solutions_dir / "baseline"),
                    exp_dir,
                    prompt,
                    best_score,
                    prev_exps,
                    reference_code=reference_code,
                )
            db.update_experiment(exp_name, workspace_dir=str(ws), output_dir=str(exp_dir / "output"))

            cn = f"agent-{exp_name}-gpu{gpu_id}"
            db.update_experiment(exp_name, container_name=cn)
            out = exp_dir / "output"

            proxy_port = cfg.proxy_port
            cmd = self.docker_cmd + [
                "run",
                "--rm",
                "--network=host",
                "--name",
                cn,
                "--gpus",
                f"device={gpu_id}",
                "--ulimit",
                "core=0",
                "-v",
                f"{cfg.data_dir}:/app/data:ro",
                "-v",
                f"{ws}:/app/workspace",
                "-v",
                f"{out}:/app/output",
                "-v",
                f"{exp_dir/'CLAUDE.md'}:/app/CLAUDE.md:ro",
                "-v",
                "/etc/ssl/certs:/etc/ssl/certs:ro",
                "-v",
                "/etc/pki:/etc/pki:ro",
                "-e",
                f"PROXY_HOST={self.proxy_host}",
                "-e",
                f"PROXY_PORT={proxy_port}",
                "-e",
                f"OPENAI_API_KEY={self.openai_api_key}",
                "-e",
                f"ANTHROPIC_BASE_URL=http://{self.proxy_host}:{proxy_port}",
                "-e",
                f"ANTHROPIC_API_KEY={self.openai_api_key or 'dummy-key'}",
                "-e",
                f"ANTHROPIC_AUTH_TOKEN={self.openai_api_key or 'dummy-key'}",
                "-e",
                "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt",
                "-e",
                "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
                "-e",
                "NODE_TLS_REJECT_UNAUTHORIZED=0",
                "-e",
                "CUDA_VISIBLE_DEVICES=0",
                "-e",
                "CUDA_DEVICE=cuda:0",
                "-e",
                f"MAX_TURNS={cfg.max_turns}",
                cfg.docker_image,
            ]

            db.add_log(exp_name, f"Docker command: {' '.join(cmd[:10])}...")
            lp = exp_dir / "agent.log"
            events_file = out / "events.jsonl"
            grace_period = 180
            with open(lp, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                deadline = time.time() + cfg.timeout_minutes * 60
                agent_done_at = None
                ec = None
                while True:
                    ret = proc.poll()
                    if ret is not None:
                        ec = ret
                        break
                    if time.time() > deadline:
                        db.add_log(exp_name, "TIMEOUT — stopping container gracefully", level="error")
                        subprocess.run(self.docker_cmd + ["stop", "-t", "30", cn], capture_output=True, timeout=60)
                        proc.wait(timeout=60)
                        ec = -1
                        break
                    if agent_done_at is None and has_result_event(events_file):
                        agent_done_at = time.time()
                        db.add_log(exp_name, "Agent finished, waiting for container to exit...")
                    if agent_done_at and time.time() - agent_done_at > grace_period:
                        db.add_log(exp_name, f"Container still running after {grace_period}s, stopping gracefully", level="warning")
                        subprocess.run(self.docker_cmd + ["stop", "-t", "30", cn], capture_output=True, timeout=60)
                        proc.wait(timeout=60)
                        ec = proc.returncode
                        break
                    time.sleep(10)

            elapsed = time.time() - t0
            db.add_log(exp_name, f"Finished in {elapsed/60:.1f}min, exit={ec}")

            if is_analysis:
                self._finalize_analysis(exp_name, exp_dir, out, elapsed, ec)
            else:
                self._finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp)
        except Exception as e:
            db.update_experiment(
                exp_name,
                status="failed",
                finished_at=datetime.now().isoformat(),
                exit_code=-2,
                elapsed_min=round((time.time() - t0) / 60, 2),
                notes=f"ERROR: {e}",
            )
            db.add_log(exp_name, f"ERROR: {e}", level="error")
            if self.rt.cfg.notify_on_failure:
                self._notify(
                    f"❌ <b>Experiment FAILED</b>\n\n"
                    f"📊 Project: <b>{self.rt.cfg.name}</b>\n"
                    f"🧪 Experiment: <code>{exp_name}</code>\n"
                    f"⏱️ Time: {(time.time() - t0)/60:.1f} min\n"
                    f"⚠️ Error: <pre>{str(e)[:500]}</pre>"
                )
        finally:
            with self.rt.lock:
                self.rt.remove_experiment_from_gpu(gpu_id, exp_name)

    def _finalize_analysis(self, exp_name, exp_dir, out, elapsed, ec):
        reports_dir = analyst_reports_dir(self.rt.cfg)
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
        db.update_experiment(
            exp_name,
            status="completed",
            finished_at=datetime.now().isoformat(),
            exit_code=ec,
            elapsed_min=round(elapsed / 60, 2),
            notes="analysis complete" if has_report else "analysis finished (no report)",
        )

    def _finalize_experiment_run(self, exp_name, exp_dir, out, elapsed, ec, best_score, lp):
        run_evaluate(self.rt.cfg, out, lp)
        ev = read_eval_results(out)
        test, val = extract_metrics(self.rt.cfg, ev)

        improved = False
        notes = "no improvement"
        if test and self.rt.cfg.is_better(test, best_score):
            improved = True
            notes = f"IMPROVED {best_score:.6f} → {test:.6f}"
            db.set_global("best_score", test)
            db.set_global("best_experiment", str(exp_dir / "workspace"))
            db.add_log(exp_name, f"★ {notes}", level="success")
            if self.rt.cfg.notify_on_improvement:
                self._notify(
                    f"🎉 <b>NEW BEST SCORE!</b>\n\n"
                    f"📊 Project: <b>{self.rt.cfg.name}</b>\n"
                    f"📈 Score: <b>{test:.6f}</b> (was {best_score:.6f})\n"
                    f"🧪 Experiment: <code>{exp_name}</code>\n"
                    f"⏱️ Time: {elapsed/60:.1f} min"
                )

        db.update_experiment(
            exp_name,
            status="completed",
            finished_at=datetime.now().isoformat(),
            exit_code=ec,
            elapsed_min=round(elapsed / 60, 2),
            test_score=test if test else None,
            val_score=val if val else None,
            improved=1 if improved else 0,
            notes=notes,
            eval_json=json.dumps(ev),
        )

        if self.rt.cfg.notify_on_completion and not improved:
            status_emoji = "✅" if ec == 0 else "❌"
            self._notify(
                f"{status_emoji} <b>Experiment completed</b>\n\n"
                f"📊 Project: <b>{self.rt.cfg.name}</b>\n"
                f"📈 Score: {test:.6f if test else 'N/A'}\n"
                f"🧪 Experiment: <code>{exp_name}</code>\n"
                f"⏱️ Time: {elapsed/60:.1f} min\n"
                f"📝 Notes: {notes}"
            )

    def _get_idea_feeder(self):
        try:
            from .. import idea_feeder
            return idea_feeder
        except ImportError:
            return None

    def worker_loop(self):
        last_sync_at = 0.0
        while self.rt.worker_running:
            try:
                now = time.time()
                if now - last_sync_at >= 15:
                    self.rt.sync_running_from_docker()
                    last_sync_at = now

                started_any = False
                while True:
                    item = None
                    with self.rt.lock:
                        if self.rt.manual_queue:
                            item = self.rt.manual_queue.popleft()
                        elif self.rt.auto_queue:
                            item = self.rt.auto_queue.popleft()
                    if item is None:
                        break

                    with self.rt.lock:
                        gpu = self.rt.get_available_gpu()
                        if gpu is None:
                            if item.get("auto"):
                                self.rt.auto_queue.appendleft(item)
                            else:
                                self.rt.manual_queue.appendleft(item)
                            break
                        self.rt.add_experiment_to_gpu(gpu, item["id"])

                    threading.Thread(
                        target=self.run_agent_in_thread,
                        args=(item["id"], item.get("prompt", ""), item.get("base_solution", ""), gpu),
                        kwargs={
                            "reference_code": item.get("reference_code"),
                            "task_type": item.get("task_type", "experiment"),
                        },
                        daemon=True,
                    ).start()
                    started_any = True
                    time.sleep(0.5)

                if not started_any:
                    self.trim_auto_queue(MAX_AUTO_QUEUE_SIZE)
                    with self.rt.lock:
                        if len(self.rt.auto_queue) > 0:
                            time.sleep(2)
                            continue
                    used_idea_names = collect_used_idea_names(self.rt)
                    feeder = self._get_idea_feeder()
                    if not feeder:
                        self.orchestrator_log("No feeder available")
                        time.sleep(2)
                        continue

                    slots_per_gpu = self.rt.cfg.slots_per_gpu if self.rt.cfg else 1
                    total_slots = len(self.rt.cfg.gpus) * slots_per_gpu if self.rt.cfg else 1
                    with self.rt.lock:
                        running_count = sum(len(exps) for exps in self.rt.running_gpus.values())
                        queued_auto_count = len(self.rt.auto_queue)
                        queued_manual_count = len(self.rt.manual_queue)
                        needed = total_slots - running_count - queued_auto_count - queued_manual_count
                        max_auto_to_add = MAX_AUTO_QUEUE_SIZE - queued_auto_count
                        needed = min(needed, max_auto_to_add)

                    if needed <= 0:
                        time.sleep(2)
                        continue

                    self.orchestrator_log(f"Getting ideas (used: {len(used_idea_names)}, need {needed})...")
                    unused = feeder.get_unused_prompts(used_idea_names, limit=needed)
                    self.orchestrator_log(f"Got {len(unused)} idea(s)")
                    queued_now = 0
                    for idx, idea in enumerate(unused):
                        if queue_idea(self.rt, self.rt.cfg, idea, idx, self.resolve_base_solution, self.orchestrator_log):
                            queued_now += 1
                    if queued_now:
                        self.orchestrator_log(f"Queued total: {queued_now}/{len(unused)}")
                    time.sleep(2)
                else:
                    time.sleep(1)
            except Exception as e:
                self.orchestrator_log(f"Worker loop error (continuing): {e}")
                time.sleep(2)

    def start_worker(self):
        if self.rt.worker_running:
            return
        self.rt.worker_running = True
        self.rt.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.rt.worker_thread.start()

    def stop_worker(self):
        self.rt.worker_running = False

    def _get_events_file_age(self, events_file: Path) -> float:
        if not events_file.exists():
            return float("inf")
        try:
            mtime = os.path.getmtime(events_file)
            return time.time() - mtime
        except Exception:
            return float("inf")

    @staticmethod
    def _find_container_name(exp_name, active_containers):
        for cn in active_containers:
            if exp_name in cn:
                return cn
        return None

    def _finalize_experiment(self, exp_name, best_score):
        cfg = self.rt.cfg
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
            self._finalize_analysis(exp_name, exp_dir, out, (elapsed_min or 0) * 60, None)
            db.add_log(exp_name, "Recovered analyst task after restart")
            return "completed", 0

        if out.exists():
            run_evaluate(self.rt.cfg, out)

        ev = read_eval_results(out)
        test, val = extract_metrics(self.rt.cfg, ev)

        improved = False
        notes = "recovered after restart"
        if test and cfg.is_better(test, best_score):
            improved = True
            notes = "IMPROVED (recovered)"
            db.set_global("best_score", test)
            db.set_global("best_experiment", str(exp_dir / "workspace"))

        submission_path = out / "submission.parquet"
        has_submission = submission_path.exists()
        has_eval = bool(ev)
        has_result = has_result_event(out / "events.jsonl")
        status = "completed" if (has_eval or has_submission) else "failed"
        if status == "failed":
            if has_result:
                notes = "recovered after restart (agent finished but no eval/submission)"
            else:
                notes = "recovered after restart (stuck or interrupted before result/eval)"

        recovered_exit_code = exp.get("exit_code") if exp else None
        if status == "failed" and recovered_exit_code is None:
            recovered_exit_code = -3

        db.update_experiment(
            exp_name,
            status=status,
            finished_at=datetime.now().isoformat(),
            exit_code=recovered_exit_code,
            elapsed_min=elapsed_min,
            test_score=test if test else None,
            val_score=val if val else None,
            improved=1 if improved else 0,
            notes=notes,
            eval_json=json.dumps(ev) if ev else None,
        )
        db.add_log(exp_name, f"Recovered: {status}, test={test or 'N/A'}")
        return status, test

    def _monitor_orphaned_container(self, exp_name, container_name, gpu_id):
        stuck_threshold = 600
        events_file = self.rt.cfg.experiments_dir / exp_name / "output" / "events.jsonl"
        agent_done_logged = False
        db.add_log(exp_name, f"Resumed monitoring (container {container_name})")

        while True:
            try:
                r = subprocess.run(
                    self.docker_cmd + ["inspect", container_name, "--format", "{{.State.Running}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if r.stdout.strip() != "true":
                    break
            except Exception:
                break

            if not agent_done_logged and has_result_event(events_file):
                agent_done_logged = True
                db.add_log(exp_name, "Agent finished (found 'result' in events.jsonl), waiting for container to exit...")

            events_age = self._get_events_file_age(events_file)
            if events_age > stuck_threshold and not agent_done_logged:
                db.add_log(exp_name, f"Container appears stuck (no events.jsonl updates for {events_age:.0f}s)", level="warning")

            time.sleep(10)

        direction = self.rt.cfg.best_score_sort_key()
        best_score = db.get_stats(direction).get("best_score", 0) or 0
        self._finalize_experiment(exp_name, best_score)
        with self.rt.lock:
            self.rt.remove_experiment_from_gpu(gpu_id, exp_name)

    def recover_orphaned_experiments(self):
        try:
            r = subprocess.run(self.docker_cmd + ["ps", "--filter", "name=agent-", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5)
            active_containers = set(r.stdout.strip().split("\n")) if r.stdout.strip() else set()
        except Exception:
            active_containers = set()

        running_exps = db.get_all_experiments(limit=200, status="running")
        if not running_exps:
            return

        direction = self.rt.cfg.best_score_sort_key()
        best_score = db.get_stats(direction).get("best_score", 0) or 0
        recovered = monitored = 0
        for exp in running_exps:
            exp_name = exp["name"]
            gpu_id = exp.get("gpu_id")
            cn = self._find_container_name(exp_name, active_containers)
            if cn:
                if gpu_id is not None:
                    with self.rt.lock:
                        self.rt.add_experiment_to_gpu(gpu_id, exp_name)
                db.add_log(exp_name, f"Resuming monitoring for orphaned experiment on GPU {gpu_id}")
                threading.Thread(target=self._monitor_orphaned_container, args=(exp_name, cn, gpu_id), daemon=True).start()
                monitored += 1
            else:
                self._finalize_experiment(exp_name, best_score)
                recovered += 1
        if recovered:
            print(f"  Recovered {recovered} orphaned experiment(s)")
        if monitored:
            print(f"  Resumed monitoring for {monitored} experiment(s)")
