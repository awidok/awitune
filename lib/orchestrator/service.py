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
from .eval import extract_metrics, has_result_event, has_result_in_agent_log, read_eval_results, run_evaluate
from .queue import collect_used_idea_names, queue_idea
from .workspace import analyst_reports_dir, prepare_analyst_workspace, prepare_workspace, prepare_stacking_workspace, resolve_base_solution

OOF_FILE_PATTERNS = (
    "oof_predictions.parquet",
    "oof_predictions.csv",
    "oof_preds.parquet",
    "oof_preds.csv",
)


class OrchestratorService:
    def __init__(self, rt, docker_cmd, awitune_dir: Path, proxy_host: str, openai_api_key: str, orchestrator_log):
        self.rt = rt
        self.docker_cmd = docker_cmd
        self.awitune_dir = awitune_dir
        self.proxy_host = proxy_host
        self.openai_api_key = openai_api_key
        self.orchestrator_log = orchestrator_log
        self._gpu_numa_affinity = self._detect_gpu_numa_affinity()

    def _notify(self, message: str, parse_mode: str = "HTML"):
        send_telegram_notification(self.rt.cfg, message, parse_mode=parse_mode)

    def trim_auto_queue(self, max_size: int = MAX_AUTO_QUEUE_SIZE) -> int:
        # Deprecated: do not cancel queued tasks anymore.
        # We now cap auto additions before enqueueing new ideas.
        return 0

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

    @staticmethod
    def _parse_cpu_set(cpu_set: str) -> list[int]:
        values = []
        for part in str(cpu_set or "").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                values.extend(range(int(a), int(b) + 1))
            else:
                values.append(int(part))
        return sorted(set(values))

    @staticmethod
    def _compress_cpu_set(values: list[int]) -> str:
        if not values:
            return ""
        values = sorted(set(values))
        ranges = []
        start = prev = values[0]
        for v in values[1:]:
            if v == prev + 1:
                prev = v
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = v
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ",".join(ranges)

    def _detect_gpu_numa_affinity(self) -> dict[int, dict]:
        mapping: dict[int, dict] = {}
        line_re = re.compile(r"^GPU(\d+)\s+.*?\s+(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\s+(\d+)\s+")
        try:
            r = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                return mapping
            for raw in r.stdout.splitlines():
                line = raw.strip()
                m = line_re.match(line)
                if not m:
                    continue
                gpu_id = int(m.group(1))
                cpu_affinity = m.group(2)
                numa_affinity = m.group(3)
                mapping[gpu_id] = {
                    "cpu_affinity": cpu_affinity,
                    "numa_affinity": numa_affinity,
                }
        except Exception:
            return {}
        return mapping

    def _docker_numa_args(self, gpu_id: int, slot_index: int) -> list[str]:
        info = self._gpu_numa_affinity.get(int(gpu_id))
        if not info:
            return []
        cpus = self._parse_cpu_set(info.get("cpu_affinity", ""))
        if not cpus:
            return []
        slots = int(self.rt.cfg.slots_per_gpu) if self.rt.cfg and self.rt.cfg.slots_per_gpu else 1
        slots = max(1, slots)
        idx = max(0, min(int(slot_index), slots - 1))
        chunk = max(1, len(cpus) // slots)
        start = idx * chunk
        end = (idx + 1) * chunk if idx < (slots - 1) else len(cpus)
        selected = cpus[start:end]
        cpuset = self._compress_cpu_set(selected) or info.get("cpu_affinity", "")
        args = ["--cpuset-cpus", cpuset]
        numa = str(info.get("numa_affinity", "")).strip()
        if numa.isdigit():
            args += ["--cpuset-mems", numa]
        return args

    @staticmethod
    def _events_file(exp_dir: Path) -> Path:
        primary = exp_dir / "events" / "events.jsonl"
        if primary.exists():
            return primary
        fallback = exp_dir / "output" / "events.jsonl"
        if fallback.exists():
            return fallback
        return primary

    def _has_result_event_any(self, exp_dir: Path, out_dir: Path) -> bool:
        # Primary location: exp_dir/events/events.jsonl (agent writes to /app/events/)
        primary = self._events_file(exp_dir)
        if has_result_event(primary):
            return True
        # Fallback: check agent.log for result event
        if has_result_in_agent_log(exp_dir):
            return True
        return False

    def _cleanup_post_result_processes(self, exp_name: str, container_name: str) -> bool:
        """Stop known background tail/watch loops that can keep container alive after result."""
        cleanup_cmd = (
            "pids=$(ps -eo pid,args | awk '"
            "/\\/home\\/agent\\/\\.claude\\/shell-snapshots\\/.*tasks\\/.*\\.output/ {print $1}; "
            "/tail -f \\/tmp\\/claude-.*\\/-app-workspace\\/tasks\\/.*\\.output/ {print $1}; "
            "/while true; do clear; tail -n 100 \\/tmp\\/claude-.*\\/-app-workspace\\/tasks\\/.*\\.output; sleep 300; done/ {print $1}; "
            "/^ *[0-9]+ sleep 300$/ {print $1}' | sort -u); "
            "if [ -n \"$pids\" ]; then kill $pids >/dev/null 2>&1; echo \"$pids\"; exit 0; fi; "
            "exit 3"
        )
        try:
            result = subprocess.run(
                self.docker_cmd + ["exec", container_name, "sh", "-c", cleanup_cmd],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                killed = " ".join(result.stdout.split())
                db.add_log(exp_name, f"Post-result cleanup: stopped background processes [{killed}]")
                return True
        except Exception as exc:
            db.add_log(exp_name, f"Post-result cleanup failed: {exc}", level="warning")
        return False

    def _terminate_post_result_agent(self, exp_name: str, container_name: str) -> bool:
        """Gracefully terminate claude process after result if it lingers."""
        try:
            result = subprocess.run(
                self.docker_cmd + ["exec", container_name, "sh", "-c", "pkill -TERM -x claude"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                db.add_log(exp_name, "Post-result cleanup: sent TERM to lingering claude process")
                return True
        except Exception as exc:
            db.add_log(exp_name, f"Post-result claude TERM failed: {exc}", level="warning")
        return False

    def _force_finish_container_after_result(self, exp_name: str, container_name: str):
        """If result is already produced, free resources by finishing container now."""
        db.add_log(exp_name, "Result is ready but container still alive; stopping container to free resources", level="warning")
        try:
            subprocess.run(
                self.docker_cmd + ["stop", "-t", "30", container_name],
                capture_output=True,
                timeout=60,
            )
        except Exception as exc:
            db.add_log(exp_name, f"Container stop after result failed: {exc}", level="warning")

        try:
            state = subprocess.run(
                self.docker_cmd + ["inspect", container_name, "--format", "{{.State.Running}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if state.stdout.strip() == "true":
                subprocess.run(
                    self.docker_cmd + ["kill", container_name],
                    capture_output=True,
                    timeout=15,
                )
                db.add_log(exp_name, "Container kill after result applied (stop was insufficient)", level="warning")
        except Exception as exc:
            db.add_log(exp_name, f"Container kill after result failed: {exc}", level="warning")

    def run_agent_in_thread(
        self,
        exp_name,
        prompt,
        base_solution,
        gpu_id,
        reference_code=None,
        task_type="experiment",
        task_payload=None,
    ):
        if task_type == "oof_fold":
            self._run_oof_fold_in_thread(exp_name, gpu_id, task_payload or {})
            return

        cfg = self.rt.cfg
        exp_dir = cfg.experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        is_analysis = task_type == "analysis"
        is_stacking = task_type == "stacking"

        db.update_experiment(exp_name, status="running", gpu_id=gpu_id, started_at=datetime.now().isoformat(), exp_dir=str(exp_dir))
        db.add_log(exp_name, f"Starting {'analysis' if is_analysis else 'experiment'} on GPU {gpu_id}")

        t0 = time.time()
        try:
            self.start_proxy()
            direction = cfg.best_score_sort_key()
            stats = db.get_stats(direction)
            best_score = stats.get("best_score", 0) or 0
            # Get val score of the best experiment to show to the agent
            # (agent should only see val score, not test score)
            best_exp = db.get_best_experiment(direction)
            best_val_score = (best_exp.get("val_score") or best_score) if best_exp else best_score
            prev_exps = db.get_all_experiments(limit=20, status="completed")

            if is_analysis:
                ws = prepare_analyst_workspace(cfg, exp_dir, prompt)
            elif is_stacking:
                stack_sources = (task_payload or {}).get("stack_sources") or []
                ws = prepare_stacking_workspace(
                    cfg,
                    exp_dir,
                    prompt,
                    best_val_score,
                    prev_exps,
                    stack_sources=stack_sources,
                )
            else:
                ws = prepare_workspace(
                    cfg,
                    base_solution or str(cfg.solutions_dir / "baseline"),
                    exp_dir,
                    prompt,
                    best_val_score,
                    prev_exps,
                    reference_code=reference_code,
                )
            db.update_experiment(exp_name, workspace_dir=str(ws), output_dir=str(exp_dir / "output"))

            cn = f"agent-{exp_name}-gpu{gpu_id}"
            db.update_experiment(exp_name, container_name=cn)
            out = exp_dir / "output"
            events_dir = exp_dir / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(events_dir, 0o777)  # Allow agent to write events
            slot_index = int((task_payload or {}).get("slot_index", 0) or 0)
            numa_args = self._docker_numa_args(gpu_id, slot_index)

            proxy_port = cfg.proxy_port
            # For stacking experiments, mount the pre-joined stacking data as /app/data/
            # This ensures stacking uses the same /app/data/ interface as regular experiments,
            # enabling stacking-on-stacking.
            if is_stacking:
                data_mount_dir = exp_dir / "stacking_data"
            else:
                data_mount_dir = cfg.data_dir
            cmd = self.docker_cmd + [
                "run",
                "--rm",
                "--network=host",
                "--name",
                cn,
                "--gpus",
                f"device={gpu_id}",
            ] + numa_args + [
                "--ulimit",
                "core=0",
                "-v",
                f"{data_mount_dir}:/app/data:ro",
                "-v",
                f"{ws}:/app/workspace",
                "-v",
                f"{out}:/app/output",
                "-v",
                f"{events_dir}:/app/events",
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
            events_file = self._events_file(exp_dir)  # Agent writes to /app/events/events.jsonl
            grace_period = 180
            with open(lp, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                deadline = time.time() + cfg.timeout_minutes * 60
                agent_done_at = None
                last_cleanup_at = 0.0
                agent_term_sent = False
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
                    if agent_done_at is None and self._has_result_event_any(exp_dir, out):
                        agent_done_at = time.time()
                        db.add_log(exp_name, "Agent finished, waiting for container to exit...")
                        if self._cleanup_post_result_processes(exp_name, cn):
                            last_cleanup_at = time.time()
                    if agent_done_at and (time.time() - last_cleanup_at) >= 30:
                        if self._cleanup_post_result_processes(exp_name, cn):
                            last_cleanup_at = time.time()
                    if agent_done_at and not agent_term_sent and (time.time() - agent_done_at) > grace_period:
                        agent_term_sent = self._terminate_post_result_agent(exp_name, cn)
                    if agent_done_at and time.time() - agent_done_at > grace_period:
                        self._force_finish_container_after_result(exp_name, cn)
                        proc.wait(timeout=60)
                        ec = proc.returncode
                        break
                    time.sleep(10)

            elapsed = time.time() - t0
            db.add_log(exp_name, f"Finished in {elapsed/60:.1f}min, exit={ec}")

            if is_analysis:
                self._finalize_analysis(exp_name, exp_dir, out, elapsed, ec)
            else:
                self._finalize_experiment_run(exp_name, exp_dir, out, elapsed, ec, best_score, lp, task_type=task_type)
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

    def _finalize_experiment_run(self, exp_name, exp_dir, out, elapsed, ec, best_score, lp, task_type="experiment"):
        run_evaluate(self.rt.cfg, out, lp)
        ev = read_eval_results(out)
        test, val = extract_metrics(self.rt.cfg, ev)
        self._register_oof_predictions(exp_name, out, test)

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

    @staticmethod
    def _infer_family(idea_or_exp: dict) -> str:
        task_type = str(idea_or_exp.get("task_type") or "").lower()
        if task_type == "analysis":
            return "analysis"
        if task_type == "oof_fold":
            return "oof_fold"
        if task_type == "stacking":
            return "stacking"
        text = " ".join(
            [
                str(idea_or_exp.get("name") or ""),
                str(idea_or_exp.get("idea_name") or ""),
                str(idea_or_exp.get("prompt") or ""),
                str(idea_or_exp.get("notes") or ""),
                str(idea_or_exp.get("base_experiment") or ""),
            ]
        ).lower()
        if any(k in text for k in ("stack", "blend", "blending", "meta-model", "ensemble", "oof")):
            return "stacking"
        if any(k in text for k in ("lightgbm", "lgbm", "xgboost", "catboost", "gbdt", "tree")):
            return "tree"
        if any(k in text for k in ("transformer", "attention", "tabtransformer", "fttransformer")):
            return "transformer"
        if any(k in text for k in ("dcnv2", "dcn", "crossnet")):
            return "dcnv2"
        if "tabnet" in text:
            return "tabnet"
        if any(k in text for k in ("mlp", "residual", "feedforward", "dense")):
            return "mlp"
        if any(k in text for k in ("linear", "ridge", "lasso", "logistic")):
            return "linear"
        return "other"

    @staticmethod
    def _is_stacking_item(idea_or_exp: dict) -> bool:
        return OrchestratorService._infer_family(idea_or_exp) == "stacking"

    def _select_diverse_ideas(self, ideas: list[dict], limit: int) -> list[dict]:
        if limit <= 0:
            return []
        by_family = {}
        for idx, idea in enumerate(ideas):
            family = self._infer_family(idea)
            by_family.setdefault(family, []).append((idx, idea))

        selected = []
        seen = set()
        for family, group in by_family.items():
            if family in seen:
                continue
            selected.append(group[0][1])
            seen.add(family)
            if len(selected) >= limit:
                return selected

        for idea in ideas:
            if len(selected) >= limit:
                break
            if idea not in selected:
                selected.append(idea)
        return selected

    def _count_stacking_mix(self) -> tuple[int, int]:
        running = db.get_all_experiments(limit=500, status="running")
        running_total = len(running)
        running_stacking = sum(1 for exp in running if self._is_stacking_item(exp))
        with self.rt.lock:
            queued_items = list(self.rt.manual_queue) + list(self.rt.auto_queue)
        queued_total = len(queued_items)
        queued_stacking = sum(1 for item in queued_items if self._is_stacking_item(item))
        return running_total + queued_total, running_stacking + queued_stacking

    def _filter_and_prioritize_auto_ideas(self, ideas: list[dict], needed: int) -> list[dict]:
        if needed <= 0 or not ideas:
            return []

        diverse = self._select_diverse_ideas(ideas, len(ideas))
        if not self.rt.cfg.enable_stacking_mode:
            return diverse[:needed]

        base_candidates = [i for i in diverse if not self._is_stacking_item(i)]
        stacking_candidates = [i for i in diverse if self._is_stacking_item(i)]
        current_total, current_stacking = self._count_stacking_mix()
        max_stack_share = max(0.0, min(1.0, float(self.rt.cfg.stacking_max_queue_share)))
        min_base_share = max(0.0, min(1.0, float(self.rt.cfg.stacking_min_base_share)))

        selected = []
        min_base_count = int(round(needed * min_base_share))
        min_base_count = min(needed, max(1, min_base_count)) if needed > 1 else needed

        for idea in base_candidates:
            if len(selected) >= min_base_count:
                break
            selected.append(idea)

        for idea in stacking_candidates:
            if len(selected) >= needed:
                break
            simulated_total = current_total + len(selected) + 1
            simulated_stacking = current_stacking + sum(1 for s in selected if self._is_stacking_item(s)) + 1
            if simulated_total <= 0:
                continue
            stack_share = simulated_stacking / simulated_total
            base_share = (simulated_total - simulated_stacking) / simulated_total
            if stack_share <= max_stack_share and base_share >= min_base_share:
                selected.append(idea)
            else:
                self.orchestrator_log(
                    f"Skip stacking idea due to quota: stack_share={stack_share:.2f}, base_share={base_share:.2f}"
                )

        for idea in base_candidates:
            if len(selected) >= needed:
                break
            if idea not in selected:
                selected.append(idea)

        return selected[:needed]

    def _oof_registry_path(self) -> Path:
        return self.rt.cfg.data_dir / "stacking_oof_registry.json"

    def _load_oof_registry(self) -> list[dict]:
        fp = self._oof_registry_path()
        if not fp.exists():
            return []
        try:
            data = json.loads(fp.read_text(errors="replace"))
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
        except Exception:
            return []
        return []

    def _save_oof_registry(self, rows: list[dict]):
        fp = self._oof_registry_path()
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    def _register_oof_predictions(self, exp_name: str, output_dir: Path, test_score):
        found = []
        for pattern in OOF_FILE_PATTERNS:
            for match in output_dir.glob(pattern):
                if match.is_file():
                    found.append(match)
        if not found:
            return

        # Validate OOF coverage: check customer_id overlap with local_train
        min_coverage = 0.9  # Require at least 90% customer_id coverage
        train_path = self.rt.cfg.data_dir / "local_train.parquet"
        train_customer_ids = None
        train_row_count = None
        if train_path.exists():
            try:
                import polars as pl
                train_ids_df = pl.read_parquet(str(train_path), columns=["customer_id"])
                train_customer_ids = set(train_ids_df["customer_id"].to_list())
                train_row_count = len(train_customer_ids)
            except Exception:
                pass

        rows = self._load_oof_registry()
        rows = [row for row in rows if Path(row.get("path", "")).exists()]
        existing = {(row.get("experiment"), row.get("path")) for row in rows}

        for fp in found:
            # Check OOF customer_id coverage against local_train
            oof_rows = None
            coverage = None
            if train_customer_ids:
                try:
                    import polars as pl
                    oof_df = pl.read_parquet(str(fp), columns=["customer_id"])
                    oof_rows = len(oof_df)
                    oof_ids = set(oof_df["customer_id"].to_list())
                    matched = len(train_customer_ids & oof_ids)
                    coverage = matched / len(train_customer_ids) if train_customer_ids else 0
                    if coverage < min_coverage:
                        db.add_log(
                            exp_name,
                            f"OOF file {fp.name} has low customer_id coverage: "
                            f"{matched}/{len(train_customer_ids)} train customers matched "
                            f"({coverage:.1%}). Minimum required: {min_coverage:.0%}. Skipping registration.",
                            level="warning",
                        )
                        continue
                except Exception as exc:
                    db.add_log(exp_name, f"Could not validate OOF file {fp.name}: {exc}", level="warning")

            exp = db.get_experiment(exp_name) or {}
            row = {
                "experiment": exp_name,
                "path": str(fp.resolve()),
                "score": test_score,
                "updated_at": datetime.now().isoformat(),
                "splitter": self.rt.cfg.stacking_oof_splitter,
                "folds": int(self.rt.cfg.stacking_oof_folds),
                "family": self._infer_family({"name": exp_name, "task_type": exp.get("task_type", "")}),
                "oof_rows": oof_rows,
                "coverage": round(coverage, 4) if coverage is not None else None,
            }
            key = (row["experiment"], row["path"])
            if key not in existing:
                rows.append(row)
                existing.add(key)
                if coverage is not None:
                    db.add_log(
                        exp_name,
                        f"Registered OOF {fp.name}: {oof_rows} rows, {coverage:.1%} customer_id coverage",
                    )

        rows.sort(
            key=lambda r: (
                r.get("score") is None,
                -(float(r.get("score")) if isinstance(r.get("score"), (int, float)) else -1e9),
            )
        )
        self._save_oof_registry(rows)
        db.set_global("stacking_oof_registry", rows)
        db.add_log(exp_name, f"Registered {len(found)} OOF file(s) for stacking reuse")

    def _build_stacking_prompt(self, user_prompt: str, stack_sources: list[str] | None = None) -> str:
        rows = self._load_oof_registry()
        # Filter: OOF file exists, experiment exists in DB, not an OOF fold, has workspace
        available = []
        for r in rows:
            if not Path(r.get("path", "")).exists():
                continue
            exp_name = r.get("experiment", "")
            exp = db.get_experiment(exp_name) if exp_name else None
            if not exp or exp.get("status") != "completed":
                continue
            if exp.get("task_type") == "oof_fold":
                continue
            available.append(r)
        available.sort(
            key=lambda r: (
                r.get("score") is None,
                -(float(r.get("score")) if isinstance(r.get("score"), (int, float)) else -1e9),
            )
        )
        if stack_sources:
            src = {str(x).strip() for x in stack_sources if str(x).strip()}
            available = [r for r in available if str(r.get("experiment") or "") in src]
        top_rows = available[:20]

        registry_lines = []
        for row in top_rows:
            score = row.get("score")
            score_str = f"{score:.6f}" if isinstance(score, (float, int)) else "N/A"
            registry_lines.append(
                f"- {row.get('experiment')} | score={score_str} | family={row.get('family','other')} | {row.get('path')}"
            )

        oof_block = "\n".join(registry_lines) if registry_lines else "- no registered OOF files yet"
        stacking_guidance = (
            "## STACKING MODE\n"
            "This run is for stacking/blending. Ignore 'NO stacking' rules.\n"
            f"OOF splitter: {self.rt.cfg.stacking_oof_splitter}, folds: {self.rt.cfg.stacking_oof_folds}.\n"
            "Reuse OOF predictions from previous experiments.\n"
            "Save OOF to `/app/output/oof_predictions.parquet` (customer_id + predict_*).\n"
            "Save report to `/app/output/report.md` with sources used and per-target AUC.\n\n"
            "### Available OOF predictions\n"
            f"{oof_block}\n"
        )
        if user_prompt:
            return f"{stacking_guidance}\n\n## Specific stacking task\n{user_prompt}"
        return stacking_guidance

    def _extract_stacking_sources(self, item: dict) -> list[str]:
        direct = item.get("stack_sources")
        if isinstance(direct, list):
            sources = [str(x).strip() for x in direct if str(x).strip()]
            if sources:
                return self._filter_stacking_sources(sources)

        base_experiment = str(item.get("base_experiment") or "").strip()
        if "," in base_experiment:
            sources = [x.strip() for x in base_experiment.split(",") if x.strip()]
            if sources:
                return self._filter_stacking_sources(sources)

        prompt = str(item.get("prompt") or "")
        if not prompt:
            return []
        all_names = {e.get("name", "") for e in db.get_all_experiments(limit=2000)}
        detected = [name for name in all_names if name and name in prompt]
        return self._filter_stacking_sources(sorted(set(detected)))

    @staticmethod
    def _filter_stacking_sources(sources: list[str]) -> list[str]:
        """Filter out analysis and oof_fold experiments from stacking sources."""
        if not sources:
            return sources
        filtered = []
        for name in sources:
            exp = db.get_experiment(name)
            if not exp:
                filtered.append(name)  # keep unknown names for later validation
                continue
            task_type = str(exp.get("task_type") or "").lower()
            if task_type in ("analysis", "oof_fold"):
                continue  # skip analysis and oof_fold experiments
            filtered.append(name)
        return filtered

    def _has_ready_oof_for_experiment(self, exp_name: str) -> bool:
        # First check if experiment is completed successfully
        exp = db.get_experiment(exp_name) or {}
        if exp.get("status") != "completed":
            return False
        # Then check if OOF predictions exist
        rows = self._load_oof_registry()
        for row in rows:
            if str(row.get("experiment") or "") != exp_name:
                continue
            if Path(str(row.get("path") or "")).exists():
                return True
        return False

    def _build_oof_runner_script(self) -> str:
        return """import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--n-folds", type=int, required=True)
    parser.add_argument("--parent-output", default="", help="Parent experiment output dir for fold 4 reuse")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split indices from pre-generated file
    split_path = data_dir / "split_indices.json"
    if not split_path.exists():
        raise RuntimeError(f"split_indices.json not found at {split_path}")

    with open(split_path) as f:
        split_data = json.load(f)

    oof_folds = split_data.get("oof_folds", [])
    n_oof_folds = len(oof_folds)

    if args.fold_idx < 0 or args.fold_idx >= n_oof_folds:
        raise RuntimeError(f"Invalid fold index {args.fold_idx} for {n_oof_folds} OOF folds")

    # Get fold indices
    fold_info = oof_folds[args.fold_idx]
    train_idx = fold_info["train_idx"]
    val_idx = fold_info["val_idx"]

    print(f"OOF fold {args.fold_idx}: train={len(train_idx)}, val={len(val_idx)}")

    # Check if this is fold 4 and we can reuse parent predictions
    # Fold 4 uses same train data as baseline (folds 0-3), so we can reuse
    if args.fold_idx == 4 and args.parent_output:
        parent_output = Path(args.parent_output)
        val_pred = parent_output / "val_predictions.parquet"
        test_pred = parent_output / "test_predictions.parquet"
        submit_pred = parent_output / "submission.parquet"

        if val_pred.exists():
            print(f"Reusing predictions from parent experiment: {parent_output}")
            # For OOF fold 4, we need predictions on entire fold 4 (val + test)
            # Parent has val on fold4_val and test on fold4_test
            # Combine them for full fold 4 coverage
            holdout = split_data.get("holdout_fold", {})
            fold4_val_idx = holdout.get("val_idx", [])
            fold4_test_idx = holdout.get("test_idx", [])

            if fold4_val_idx and fold4_test_idx:
                # Load parent predictions
                val_df = pl.read_parquet(str(val_pred))
                test_df = pl.read_parquet(str(test_pred)) if test_pred.exists() else None

                # Load customer_ids from fold 4 data
                train_path = data_dir / "train.parquet"
                if not train_path.exists():
                    train_path = data_dir / "local_train.parquet"
                full_df = pl.read_parquet(str(train_path), columns=["customer_id"])

                # Build combined predictions for entire fold 4
                predict_cols = [c for c in val_df.columns if c.startswith("predict_")]

                # Get customer_ids for val and test portions
                val_customers = full_df[fold4_val_idx].select("customer_id")
                test_customers = full_df[fold4_test_idx].select("customer_id")

                # Join predictions with correct customer_ids
                val_preds_aligned = val_customers.join(val_df, on="customer_id", how="left")
                test_preds_aligned = test_customers.join(test_df, on="customer_id", how="left") if test_df else None

                if test_preds_aligned is not None:
                    combined = pl.concat([val_preds_aligned, test_preds_aligned])
                    combined.write_parquet(str(output_dir / "oof_fold_predictions.parquet"))
                    print(f"Combined val+test predictions for fold 4: {len(combined)} rows")
                else:
                    # Fallback: just use val predictions
                    (output_dir / "oof_fold_predictions.parquet").write_bytes(val_pred.read_bytes())

                # Copy submission
                if submit_pred.exists():
                    (output_dir / "submission_fold.parquet").write_bytes(submit_pred.read_bytes())

                print("Successfully reused parent predictions for OOF fold 4")
                return
            else:
                print("WARNING: Could not find holdout fold indices, running training instead")

    # Standard OOF training for folds 0-3
    # Load full training data (train.parquet contains all data)
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        # Fallback to local_train if train.parquet doesn't exist
        train_path = data_dir / "local_train.parquet"

    df = pl.read_parquet(str(train_path))
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if not target_cols:
        raise RuntimeError("No target_* columns found in training data")

    fold_data_dir = output_dir / "fold_data"
    fold_data_dir.mkdir(parents=True, exist_ok=True)
    train_prefix = fold_data_dir / "train"
    val_prefix = fold_data_dir / "val"

    df[train_idx].write_parquet(str(train_prefix) + ".parquet")
    df[val_idx].write_parquet(str(val_prefix) + ".parquet")
    for metadata_file in data_dir.glob("*.json"):
        target_meta = fold_data_dir / metadata_file.name
        target_meta.write_bytes(metadata_file.read_bytes())

    run_output = output_dir / "run_output"
    run_output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(workspace / "run.py"),
        "--train",
        str(train_prefix),
        "--val",
        str(val_prefix),
        "--test",
        str(data_dir / "local_test"),
        "--submit",
        str(data_dir / "contest_test"),
        str(run_output),
    ]
    subprocess.run(cmd, check=True)

    val_pred = run_output / "val_predictions.parquet"
    if not val_pred.exists():
        raise RuntimeError("run.py did not produce val_predictions.parquet")
    (output_dir / "oof_fold_predictions.parquet").write_bytes(val_pred.read_bytes())

    submit_pred = run_output / "submission.parquet"
    if submit_pred.exists():
        (output_dir / "submission_fold.parquet").write_bytes(submit_pred.read_bytes())

    test_pred = run_output / "test_predictions.parquet"
    if test_pred.exists():
        (output_dir / "test_fold_predictions.parquet").write_bytes(test_pred.read_bytes())


if __name__ == "__main__":
    main()
"""

    def _has_active_oof_jobs(self, parent_exp_name: str) -> bool:
        all_exps = db.get_all_experiments(limit=5000)
        for exp in all_exps:
            if exp.get("task_type") != "oof_fold":
                continue
            if exp.get("parent_experiment") != parent_exp_name:
                continue
            if exp.get("status") in ("queued", "running"):
                return True
        return False

    @staticmethod
    def _extract_oof_fold_idx(parent_exp_name: str, exp_name: str):
        pattern = rf"^oof_{re.escape(parent_exp_name)}_f(\d+)(?:_|$)"
        m = re.match(pattern, exp_name or "")
        if not m:
            return None
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None

    def _list_parent_oof_experiments(self, parent_exp_name: str):
        all_exps = db.get_all_experiments(limit=5000)
        return [
            e for e in all_exps
            if e.get("task_type") == "oof_fold" and e.get("parent_experiment") == parent_exp_name
        ]

    def _drop_oof_queue_items(self, parent_exp_name: str, keep_ids: set[str] | None = None):
        keep_ids = keep_ids or set()
        with self.rt.lock:
            self.rt.manual_queue = type(self.rt.manual_queue)(
                item for item in self.rt.manual_queue
                if not (
                    item.get("task_type") == "oof_fold"
                    and item.get("parent_experiment") == parent_exp_name
                    and item.get("id") not in keep_ids
                )
            )
            self.rt.auto_queue = type(self.rt.auto_queue)(
                item for item in self.rt.auto_queue
                if not (
                    item.get("task_type") == "oof_fold"
                    and item.get("parent_experiment") == parent_exp_name
                    and item.get("id") not in keep_ids
                )
            )

    def _prepare_inplace_oof_slots(self, parent_exp_name: str, n_folds: int, parent_workspace: str):
        parent_oof = self._list_parent_oof_experiments(parent_exp_name)
        by_fold_idx = {}
        extras = []
        for exp in sorted(parent_oof, key=lambda e: e.get("created_at") or "", reverse=True):
            name = str(exp.get("name") or "")
            fold_idx = self._extract_oof_fold_idx(parent_exp_name, name)
            if fold_idx is None or fold_idx < 0 or fold_idx >= n_folds:
                extras.append(name)
                continue
            if fold_idx not in by_fold_idx:
                by_fold_idx[fold_idx] = exp
            else:
                extras.append(name)

        for exp_name in extras:
            db.delete_experiment(exp_name)

        # Only create slots for folds 0 to n_folds-2 (last fold reuses parent predictions)
        # This allows OOF fold 4 to reuse baseline's val+test predictions
        slots: list[tuple[int, str]] = []
        for fold_idx in range(n_folds - 1):
            existing = by_fold_idx.get(fold_idx)
            if existing:
                eid = str(existing.get("name") or "")
                # Skip completed folds - don't re-queue them
                if existing.get("status") == "completed":
                    continue
            else:
                eid = f"oof_{parent_exp_name}_f{fold_idx}"
                db.create_experiment(
                    eid,
                    prompt=f"External OOF fold {fold_idx}/{n_folds - 1} for {parent_exp_name}",
                    base_solution=parent_workspace,
                    parent_experiment=parent_exp_name,
                    task_type="oof_fold",
                )
            db.update_experiment(
                eid,
                status="queued",
                started_at=None,
                finished_at=None,
                exit_code=None,
                elapsed_min=None,
                gpu_id=None,
                container_name=None,
                container_id=None,
                val_score=None,
                test_score=None,
                cv_score=None,
                improved=0,
                notes="",
                eval_json=None,
                parent_experiment=parent_exp_name,
                task_type="oof_fold",
            )
            slots.append((fold_idx, eid))
        return slots

    def enqueue_oof_for_experiment(self, parent_exp_name: str, manual: bool = True, requester: str = "manual") -> dict:
        return self._enqueue_external_oof_folds_for_source(
            parent_exp_name=parent_exp_name,
            requester=requester,
            manual=manual,
        )

    def _enqueue_external_oof_folds_for_source(self, parent_exp_name: str, requester: str = "", manual: bool = False) -> dict:
        parent = db.get_experiment(parent_exp_name) or {}

        # Never create OOF folds for experiments that are themselves OOF folds
        if parent.get("task_type") == "oof_fold":
            if requester:
                db.add_log(requester, f"Skipping OOF for {parent_exp_name}: it is itself an OOF fold", level="warning")
            return {"status": "error", "message": "cannot create OOF for an OOF fold experiment"}

        if not parent.get("workspace_dir"):
            if requester:
                db.add_log(requester, f"Cannot queue OOF for {parent_exp_name}: missing workspace", level="warning")
            return {"status": "error", "message": "missing workspace"}

        if self._has_active_oof_jobs(parent_exp_name):
            return {"status": "already_running", "message": "OOF already queued/running"}

        if db.get_oof_lock_owner(parent_exp_name) and not self._has_active_oof_jobs(parent_exp_name):
            # stale lock after restart/crash
            db.release_oof_lock(parent_exp_name)

        owner = requester or ("manual" if manual else "stacking_dependency")
        if not db.try_acquire_oof_lock(parent_exp_name, owner=owner):
            return {"status": "already_running", "message": "OOF lock is already acquired"}

        try:
            if not manual:
                with self.rt.lock:
                    queued_auto = len(self.rt.auto_queue)
                if queued_auto >= MAX_AUTO_QUEUE_SIZE:
                    db.release_oof_lock(parent_exp_name)
                    return {"status": "deferred", "message": f"auto queue is full ({queued_auto}/{MAX_AUTO_QUEUE_SIZE})"}

            parent_dir = self.rt.cfg.experiments_dir / parent_exp_name
            oof_runner_path = parent_dir / "oof_runner.py"
            oof_runner_path.write_text(self._build_oof_runner_script())

            n_folds = int(self.rt.cfg.stacking_oof_folds)
            parent_workspace = str(parent.get("workspace_dir", ""))
            slots = self._prepare_inplace_oof_slots(parent_exp_name, n_folds, parent_workspace)
            keep_ids = {eid for _, eid in slots}
            self._drop_oof_queue_items(parent_exp_name, keep_ids=keep_ids)

            if not slots:
                # All folds already completed — try to aggregate and release lock
                self._aggregate_parent_oof(parent_exp_name)
                db.release_oof_lock(parent_exp_name)
                self.orchestrator_log(f"All OOF folds already completed for {parent_exp_name}, aggregated and released lock")
                return {"status": "queued", "count": 0}

            for fold_idx, eid in slots:
                db.add_log(
                    eid,
                    f"Queued external OOF fold {fold_idx + 1}/{n_folds} for {parent_exp_name}"
                    + (f" (requested by {requester})" if requester else ""),
                )
                item = {
                    "id": eid,
                    "prompt": "",
                    "base_solution": parent_workspace,
                    "task_type": "oof_fold",
                    "auto": not manual,
                    "parent_experiment": parent_exp_name,
                    "fold_idx": fold_idx,
                    "n_folds": n_folds,
                    "oof_runner_path": str(oof_runner_path),
                }
                with self.rt.lock:
                    if manual:
                        self.rt.manual_queue.append(item)
                    else:
                        self.rt.auto_queue.append(item)
            self.orchestrator_log(
                f"Queued {len(slots)} external OOF fold task(s) for {parent_exp_name} "
                f"(queue={'manual' if manual else 'auto'}, fold 4 reuses parent predictions)"
            )
            return {"status": "queued", "count": len(slots)}
        except Exception as exc:
            db.release_oof_lock(parent_exp_name)
            return {"status": "error", "message": f"Failed to queue OOF: {exc}"}

    def _release_oof_lock_if_done(self, parent_exp_name: str):
        if not parent_exp_name:
            return
        if self._has_active_oof_jobs(parent_exp_name):
            return
        if db.get_oof_lock_owner(parent_exp_name):
            db.release_oof_lock(parent_exp_name)
            db.add_log(parent_exp_name, "Released OOF lock")

    def _cleanup_stale_oof_locks(self):
        for key in db.list_global_keys("oof_lock:"):
            parent_exp = key.replace("oof_lock:", "", 1)
            if not parent_exp:
                continue
            if not self._has_active_oof_jobs(parent_exp):
                db.release_oof_lock(parent_exp)
                db.add_log(parent_exp, "Released stale OOF lock", level="warning")

    def _regenerate_stacking_sources(self, item: dict) -> list[str]:
        """Ask LLM to regenerate stack_sources with correct experiment names."""
        try:
            from ..generate_ideas import configure as configure_ideas, call_openai_with_tools, parse_ideas
            from ..orchestrator.tools import configure as configure_tools

            configure_ideas(self.rt.cfg)
            configure_tools(self.rt.cfg)

            # Build list of available completed experiments with scores
            completed = db.get_all_experiments(limit=200, status="completed")
            exp_list = []
            for e in completed:
                if e.get("task_type") in ("analysis", "oof_fold"):
                    continue
                score = e.get("test_score")
                score_str = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
                exp_list.append(f"- {e['name']} (score={score_str}, family={self._infer_family(e)})")

            if not exp_list:
                return []

            prompt = (
                "The orchestrator tried to create a stacking experiment but the stack_sources "
                "contained invalid experiment names that don't exist in the database.\n\n"
                f"Original prompt: {item.get('prompt', '')[:500]}\n"
                f"Original stack_sources (INVALID): {item.get('stack_sources', [])}\n\n"
                "Here are ALL available completed experiments:\n"
                + "\n".join(exp_list[:50]) + "\n\n"
                "Pick 2-5 diverse experiments from the list above for stacking. "
                "Return ONLY a JSON object with a single key 'stack_sources' containing "
                "the EXACT experiment names from the list. Example:\n"
                '{"stack_sources": ["auto_exp_A_20260310_123456_0", "auto_exp_B_20260310_234567_1"]}\n'
                "Return ONLY valid JSON, no commentary."
            )

            system = "You are a helper that picks experiment names for stacking. Return only JSON."
            response = call_openai_with_tools(system, prompt, temperature=0.3, max_retries=2)
            if not response:
                return []

            import json
            text = response.strip()
            # Try to extract JSON object
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1:
                obj = json.loads(text[brace_start:brace_end + 1])
                sources = obj.get("stack_sources", [])
                if isinstance(sources, list):
                    return [str(x).strip() for x in sources if str(x).strip()]
            return []
        except Exception as exc:
            self.orchestrator_log(f"Failed to regenerate stacking sources: {exc}")
            return []

    def _ensure_stacking_dependencies(self, item: dict) -> bool:
        if item.get("task_type") != "stacking":
            return True
        sources = self._extract_stacking_sources(item)

        # If no sources found, ask LLM to regenerate with correct names
        if not sources:
            retry_count = item.get("_source_retry", 0)
            if retry_count < 1:
                db.add_log(item["id"], "No valid stack_sources found, asking LLM to regenerate with correct experiment names...")
                regenerated = self._regenerate_stacking_sources(item)
                if regenerated:
                    item["stack_sources"] = regenerated
                    item["_source_retry"] = retry_count + 1
                    db.add_log(item["id"], f"LLM regenerated stack_sources: {regenerated}")
                    sources = regenerated
                else:
                    db.add_log(item["id"], "LLM could not regenerate stack_sources. Stacking blocked.", level="error")
                    # Cancel the experiment instead of infinite re-queue
                    db.update_experiment(item["id"], status="failed", notes="No valid stack_sources — LLM regeneration failed")
                    return True  # Return True to not re-queue; experiment is already marked failed
            else:
                db.add_log(item["id"], "Stacking blocked after retry: still no valid stack_sources.", level="error")
                db.update_experiment(item["id"], status="failed", notes="No valid stack_sources after LLM retry")
                return True  # Don't re-queue

        if not sources:
            return True  # Already marked as failed above

        item["stack_sources"] = sources
        db.add_log(item["id"], f"Stacking sources: {sources}")

        missing = []
        failed_sources = item.get("_failed_oof_sources", set())
        if not isinstance(failed_sources, set):
            failed_sources = set(failed_sources)

        all_exps = db.get_all_experiments(limit=5000)
        required_fold_count = max(0, int(self.rt.cfg.stacking_oof_folds) - 1)

        for source_exp in sources:
            if self._has_ready_oof_for_experiment(source_exp):
                continue

            if source_exp in failed_sources:
                continue

            source_exp_row = db.get_experiment(source_exp) or {}
            source_status = str(source_exp_row.get("status") or "")
            source_notes = str(source_exp_row.get("notes") or "")

            source_fold_exps = [
                e for e in all_exps
                if e.get("task_type") == "oof_fold" and e.get("parent_experiment") == source_exp
            ]
            failed_fold_exps = [e for e in source_fold_exps if e.get("status") == "failed"]
            completed_fold_exps = [e for e in source_fold_exps if e.get("status") == "completed"]
            active_fold_exps = [e for e in source_fold_exps if e.get("status") in ("queued", "running")]

            if failed_fold_exps and not active_fold_exps and len(completed_fold_exps) < required_fold_count:
                failed_sources.add(source_exp)
                failed_names = ", ".join(e.get("name", "") for e in failed_fold_exps)
                db.add_log(
                    item["id"],
                    f"OOF dependency {source_exp} failed after 3 attempts; failed folds: {failed_names}",
                    level="error",
                )
                continue

            if source_status == "failed" and "oof" in source_notes.lower():
                failed_sources.add(source_exp)
                db.add_log(
                    item["id"],
                    f"OOF dependency {source_exp} failed permanently: {source_notes}",
                    level="error",
                )
                continue

            enqueue_result = self._enqueue_external_oof_folds_for_source(source_exp, requester=item["id"], manual=False)
            status = enqueue_result.get("status", "")
            if status in ("queued", "already_running", "deferred"):
                missing.append(source_exp)
            elif status == "error":
                failed_sources.add(source_exp)
                db.add_log(
                    item["id"],
                    f"OOF dependency {source_exp} failed permanently: {enqueue_result.get('message', 'unknown')}",
                    level="error",
                )
            else:
                missing.append(source_exp)

        item["_failed_oof_sources"] = list(failed_sources)

        available_sources = [s for s in sources if self._has_ready_oof_for_experiment(s)]
        unavailable_sources = [s for s in sources if s in failed_sources and not self._has_ready_oof_for_experiment(s)]
        if unavailable_sources:
            db.add_log(
                item["id"],
                f"Stacking dependencies failed permanently: {', '.join(unavailable_sources)}",
                level="error",
            )
            db.update_experiment(
                item["id"],
                status="failed",
                notes=(
                    f"OOF dependency failure after 3 attempts; ready_sources={len(available_sources)}; "
                    f"failed_sources={', '.join(sorted(unavailable_sources))}"
                ),
            )
            return True

        if missing:
            # Throttle logging — only log every 10th check
            wait_count = item.get("_wait_count", 0) + 1
            item["_wait_count"] = wait_count
            if wait_count <= 1 or wait_count % 10 == 0:
                db.add_log(
                    item["id"],
                    f"Waiting for OOF dependencies ({wait_count}): {', '.join(missing)}",
                )
            return False
        return True

    def _aggregate_parent_oof(self, parent_exp_name: str):
        try:
            import polars as pl
        except Exception:
            return

        all_exps = db.get_all_experiments(limit=5000)
        fold_exps = [
            e
            for e in all_exps
            if e.get("task_type") == "oof_fold"
            and e.get("parent_experiment") == parent_exp_name
            and e.get("status") == "completed"
        ]
        n_folds = int(self.rt.cfg.stacking_oof_folds)
        # We need n_folds-1 completed fold experiments (fold 4 reuses parent predictions)
        if len(fold_exps) < n_folds - 1:
            return

        # Per-fold validation: check each fold predicts the correct customer_ids
        split_path = self.rt.cfg.data_dir / "split_indices.json"
        full_train_path = self.rt.cfg.data_dir / "train.parquet"
        expected_ids_by_fold = {}
        if split_path.exists() and full_train_path.exists():
            try:
                import json as _json
                splits = _json.loads(split_path.read_text())
                full_train = pl.read_parquet(str(full_train_path), columns=["customer_id"])
                for fold_info in splits.get("oof_folds", []):
                    fidx = fold_info.get("fold")
                    val_idx = fold_info.get("val_idx", [])
                    if fidx is not None and val_idx:
                        expected_ids_by_fold[fidx] = set(full_train[val_idx]["customer_id"].to_list())
            except Exception:
                pass

        parts = []
        submit_parts = []
        fold_cv_scores = []
        bad_folds = []
        for e in fold_exps:
            fold_output = self.rt.cfg.experiments_dir / e["name"] / "output"
            fp = fold_output / "oof_fold_predictions.parquet"
            if not fp.exists():
                continue

            oof_df = pl.read_parquet(str(fp))

            # Validate fold predictions against expected customer_ids
            fold_idx = self._extract_oof_fold_idx(parent_exp_name, e["name"])
            if fold_idx is not None and fold_idx in expected_ids_by_fold and "customer_id" in oof_df.columns:
                expected = expected_ids_by_fold[fold_idx]
                actual = set(oof_df["customer_id"].to_list())
                match_pct = len(expected & actual) / len(expected) if expected else 0
                if match_pct < 0.9:
                    db.add_log(
                        e["name"],
                        f"Fold {fold_idx} predicted wrong customers: {match_pct:.1%} match. "
                        f"Resetting to queued for re-run.",
                        level="error",
                    )
                    db.update_experiment(e["name"], status="queued", notes=f"Reset: fold predicted wrong customers ({match_pct:.1%} match)")
                    try:
                        fp.unlink()
                    except Exception:
                        pass
                    bad_folds.append(e["name"])
                    continue

            parts.append(oof_df)
            if isinstance(e.get("cv_score"), (int, float)):
                fold_cv_scores.append(float(e["cv_score"]))
            submit_fp = fold_output / "submission_fold.parquet"
            if submit_fp.exists():
                submit_df = pl.read_parquet(str(submit_fp))
                submit_parts.append((e["name"], submit_df))

        if bad_folds:
            db.add_log(parent_exp_name, f"OOF aggregation deferred: {len(bad_folds)} fold(s) had wrong predictions and were reset: {bad_folds}", level="warning")
            return

        # Fold 4: reuse parent's val + test predictions
        parent_out = self.rt.cfg.experiments_dir / parent_exp_name / "output"
        val_fp = parent_out / "val_predictions.parquet"
        test_fp = parent_out / "test_predictions.parquet"
        if val_fp.exists() and test_fp.exists():
            val_df = pl.read_parquet(str(val_fp))
            test_df = pl.read_parquet(str(test_fp))
            fold4_df = pl.concat([val_df, test_df], how="vertical")
            parts.append(fold4_df)
            # Use parent's CV score for fold 4
            parent_exp = db.get_experiment(parent_exp_name) or {}
            if isinstance(parent_exp.get("val_score"), (int, float)):
                fold_cv_scores.append(float(parent_exp["val_score"]))

        if len(parts) < n_folds:
            return

        merged = pl.concat(parts, how="vertical")
        total_rows_before_dedup = len(merged)
        if "customer_id" in merged.columns:
            merged = merged.unique(subset=["customer_id"], keep="first")

        # Validate: after dedup, we should have close to the full train.parquet row count
        # If many rows were duplicates, it means folds predicted overlapping customers (buggy run.py)
        expected_rows = sum(len(p) for p in parts)
        dedup_ratio = len(merged) / total_rows_before_dedup if total_rows_before_dedup > 0 else 0
        if dedup_ratio < 0.8:
            db.add_log(
                parent_exp_name,
                f"OOF aggregation WARNING: {total_rows_before_dedup} rows before dedup → {len(merged)} after "
                f"({dedup_ratio:.1%} unique). Folds likely predicted overlapping customers. "
                f"This OOF file may have low coverage.",
                level="warning",
            )

        # Validate customer_id coverage against local_train
        train_path = self.rt.cfg.data_dir / "local_train.parquet"
        coverage_ok = True
        if train_path.exists() and "customer_id" in merged.columns:
            try:
                train_ids = set(pl.read_parquet(str(train_path), columns=["customer_id"])["customer_id"].to_list())
                oof_ids = set(merged["customer_id"].to_list())
                matched = len(train_ids & oof_ids)
                coverage = matched / len(train_ids) if train_ids else 0
                db.add_log(
                    parent_exp_name,
                    f"OOF coverage: {matched}/{len(train_ids)} train customers ({coverage:.1%})",
                )
                if coverage < 0.9:
                    db.add_log(
                        parent_exp_name,
                        f"OOF aggregation REJECTED: only {coverage:.1%} coverage. "
                        f"Some folds likely predicted wrong customers. Will not register.",
                        level="error",
                    )
                    coverage_ok = False
            except Exception as exc:
                db.add_log(parent_exp_name, f"Could not validate OOF coverage: {exc}", level="warning")

        parent_out.mkdir(parents=True, exist_ok=True)
        final_oof = parent_out / "oof_predictions.parquet"
        merged.write_parquet(str(final_oof))
        db.add_log(parent_exp_name, f"OOF aggregated: {len(merged)} unique rows from {len(parts)} folds")

        if submit_parts:
            submit_dir = parent_out / "submission_folds"
            submit_dir.mkdir(parents=True, exist_ok=True)
            aligned = []
            for fold_name, df in submit_parts:
                per_fold_path = submit_dir / f"{fold_name}.parquet"
                df.write_parquet(str(per_fold_path))
                aligned.append(df)
            if aligned:
                base = aligned[0].clone()
                pred_cols = [c for c in base.columns if c.startswith("predict_")]
                if pred_cols:
                    for col in pred_cols:
                        col_sum = None
                        for df in aligned:
                            expr = pl.col(col)
                            col_sum = expr if col_sum is None else (col_sum + expr)
                        base = base.with_columns((col_sum / len(aligned)).alias(col))
                    base.write_parquet(str(parent_out / "submission_oof_mean.parquet"))
                db.add_log(parent_exp_name, f"Saved per-fold submit predictions: {len(aligned)} fold file(s)")

        parent = db.get_experiment(parent_exp_name) or {}
        if coverage_ok:
            self._register_oof_predictions(parent_exp_name, parent_out, parent.get("test_score"))
        else:
            db.add_log(parent_exp_name, "Skipped OOF registration due to low coverage", level="warning")
        # Only update CV score if we have all n_folds scores (including fold 4 from parent)
        if len(fold_cv_scores) == n_folds:
            cv_score = float(sum(fold_cv_scores) / len(fold_cv_scores))
            db.update_experiment(parent_exp_name, cv_score=cv_score)
            db.add_log(parent_exp_name, f"Updated CV score from OOF folds: {cv_score:.6f}")
        else:
            db.add_log(parent_exp_name, f"CV score not updated: have {len(fold_cv_scores)}/{n_folds} fold scores")
        db.add_log(parent_exp_name, f"External OOF merged from {len(parts)} fold(s): {final_oof}")

    def _run_oof_fold_in_thread(self, exp_name: str, gpu_id: int, task_payload: dict):
        cfg = self.rt.cfg
        exp_dir = cfg.experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "oof_fold.log"
        out = exp_dir / "output"
        if out.exists():
            try:
                subprocess.run(["chmod", "-R", "a+rwX", str(out)], capture_output=True, timeout=15)
            except Exception:
                pass
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(out, 0o777)
        except Exception:
            pass

        parent_exp = str(task_payload.get("parent_experiment") or "")
        fold_idx = int(task_payload.get("fold_idx", 0))
        n_folds = int(task_payload.get("n_folds", max(2, int(cfg.stacking_oof_folds))))
        runner_path = Path(str(task_payload.get("oof_runner_path") or ""))
        parent = db.get_experiment(parent_exp) or {}
        parent_ws = Path(str(parent.get("workspace_dir") or ""))
        parent_output = Path(str(parent.get("exp_dir") or "")) / "output"

        # Simple retry logic: at most 3 total attempts
        max_retries = 2
        max_attempts = max_retries + 1
        exit_code = -1
        t0 = time.time()

        for retry_idx in range(max_retries + 1):
            if retry_idx > 0:
                db.add_log(exp_name, f"Retry {retry_idx}/{max_retries} (attempt {retry_idx + 1}/{max_attempts})")
                # Clean up output directory for retry
                try:
                    subprocess.run(["rm", "-rf", str(out)], capture_output=True, timeout=15)
                    out.mkdir(parents=True, exist_ok=True)
                    os.chmod(out, 0o777)
                except Exception:
                    pass

            db.update_experiment(exp_name, status="running", gpu_id=gpu_id, started_at=datetime.now().isoformat(), exp_dir=str(exp_dir))
            db.add_log(exp_name, f"Starting external OOF fold {fold_idx + 1}/{n_folds} on GPU {gpu_id}" + (f" (retry {retry_idx})" if retry_idx > 0 else ""))
            exit_code = -1
            try:
                if not parent_ws.exists() or not (parent_ws / "run.py").exists():
                    raise RuntimeError(f"Parent workspace/run.py not found for {parent_exp}")
                if not runner_path.exists():
                    if parent_exp:
                        parent_dir = cfg.experiments_dir / parent_exp
                        parent_dir.mkdir(parents=True, exist_ok=True)
                        runner_path = parent_dir / "oof_runner.py"
                        runner_path.write_text(self._build_oof_runner_script())
                    else:
                        raise RuntimeError(f"OOF runner script not found: {runner_path}")

                cn = f"oof-{exp_name}-gpu{gpu_id}"[:120]
                cmd = self.docker_cmd + [
                    "run",
                    "--rm",
                    "--network=host",
                    "--name",
                    cn,
                    "--gpus",
                    f"device={gpu_id}",
                ] + self._docker_numa_args(gpu_id, int(task_payload.get("slot_index", 0) or 0)) + [
                    "--user",
                    "0:0",
                    "--entrypoint",
                    "python",
                    "-v",
                    f"{cfg.data_dir}:/app/data:ro",
                    "-v",
                    f"{parent_ws}:/app/workspace:ro",
                    "-v",
                    f"{out}:/app/output",
                    "-v",
                    f"{runner_path}:/app/oof_runner.py:ro",
                    "-v",
                    f"{parent_output}:/app/parent_output:ro",
                    cfg.docker_image,
                    "/app/oof_runner.py",
                    "--workspace",
                    "/app/workspace",
                    "--data-dir",
                    "/app/data",
                    "--output-dir",
                    "/app/output",
                    "--fold-idx",
                    str(fold_idx),
                    "--n-folds",
                    str(n_folds),
                    "--parent-output",
                    "/app/parent_output" if parent_output.exists() else "",
                ]

                log_mode = "w" if retry_idx == 0 else "a"
                with open(log_path, log_mode, encoding="utf-8") as lf:
                    if retry_idx > 0:
                        lf.write(f"\n{'='*80}\n")
                    lf.write(f"[{datetime.now().isoformat()}] OOF launch (attempt {retry_idx + 1})\n")
                    lf.write(f"experiment: {exp_name}\n")
                    lf.write(f"container: {cn}\n")
                    lf.write(f"gpu: {gpu_id}\n")
                    lf.write(f"fold: {fold_idx + 1}/{n_folds}\n")
                    lf.write(f"command: {' '.join(cmd)}\n")
                    lf.write("-" * 80 + "\n")
                    lf.flush()
                    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                    deadline = time.time() + cfg.timeout_minutes * 60
                    while True:
                        ret = proc.poll()
                        if ret is not None:
                            exit_code = ret
                            break
                        if time.time() > deadline:
                            lf.write(f"\n[{datetime.now().isoformat()}] TIMEOUT reached, stopping container {cn}\n")
                            lf.flush()
                            subprocess.run(self.docker_cmd + ["stop", "-t", "30", cn], capture_output=True, timeout=60)
                            proc.wait(timeout=60)
                            exit_code = -1
                            break
                        time.sleep(5)
                    elapsed_min = (time.time() - t0) / 60
                    lf.write(f"\n[{datetime.now().isoformat()}] OOF finished with exit_code={exit_code}, elapsed={elapsed_min:.2f} min\n")
                    lf.flush()

            except Exception as exc:
                try:
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(f"\n[{datetime.now().isoformat()}] OOF ERROR: {exc}\n")
                except Exception:
                    pass
                exit_code = -2

            # Check if successful
            oof_pred = out / "oof_fold_predictions.parquet"
            if exit_code == 0 and oof_pred.exists():
                break  # Success
            elif retry_idx < max_retries:
                # Retry on any failure, but never exceed 3 total attempts
                time.sleep(5)
                continue
            else:
                break  # Max retries exceeded

        # Final status update
        oof_pred = out / "oof_fold_predictions.parquet"
        status = "completed" if exit_code == 0 and oof_pred.exists() else "failed"
        notes = f"external OOF fold {fold_idx + 1}/{n_folds}"
        if status == "failed" and not oof_pred.exists():
            notes = f"failed after {max_attempts} attempts: missing oof_fold_predictions.parquet"
        fold_cv = None
        metrics_fp = out / "run_output" / "metrics.json"
        if metrics_fp.exists():
            try:
                metrics = json.loads(metrics_fp.read_text(errors="replace"))
                raw_cv = metrics.get(cfg.val_metric_key)
                if isinstance(raw_cv, (int, float)):
                    fold_cv = float(raw_cv)
            except Exception:
                fold_cv = None

        db.update_experiment(
            exp_name,
            status=status,
            finished_at=datetime.now().isoformat(),
            exit_code=exit_code,
            elapsed_min=round((time.time() - t0) / 60, 2),
            notes=notes,
            cv_score=fold_cv,
        )
        db.add_log(exp_name, f"OOF fold finished with status={status}, exit={exit_code}, attempts={min(max_attempts, retry_idx + 1)}")
        if status == "completed" and parent_exp:
            self._aggregate_parent_oof(parent_exp)
        self._release_oof_lock_if_done(parent_exp)

    def worker_loop(self):
        last_sync_at = 0.0
        last_oof_cleanup_at = 0.0
        while self.rt.worker_running:
            try:
                now = time.time()
                if now - last_sync_at >= 15:
                    self.rt.sync_running_from_docker()
                    last_sync_at = now
                if now - last_oof_cleanup_at >= 30:
                    self._cleanup_stale_oof_locks()
                    last_oof_cleanup_at = now

                started_any = False
                blocked_stacking = []
                # --- Pass 1: drain queues, prioritise non-stacking items ---
                non_stacking_items = []
                stacking_items = []
                with self.rt.lock:
                    while self.rt.manual_queue or self.rt.auto_queue:
                        if self.rt.manual_queue:
                            item = self.rt.manual_queue.popleft()
                        else:
                            item = self.rt.auto_queue.popleft()
                        if item.get("task_type") == "stacking":
                            stacking_items.append(item)
                        else:
                            non_stacking_items.append(item)

                # Process non-stacking first, then stacking
                ordered_items = non_stacking_items + stacking_items

                # --- Pass 2: check stacking deps & start tasks ---
                deferred_items = []
                no_gpu = False
                for item in ordered_items:
                    # Once GPUs are exhausted, defer remaining items without checking deps
                    if no_gpu:
                        if item.get("task_type") == "stacking":
                            blocked_stacking.append(item)
                        else:
                            deferred_items.append(item)
                        continue

                    if item.get("task_type") == "stacking":
                        try:
                            if not self._ensure_stacking_dependencies(item):
                                blocked_stacking.append(item)
                                continue
                        except Exception as exc:
                            self.orchestrator_log(f"Stacking dep check error for {item.get('id')}: {exc}")
                            blocked_stacking.append(item)
                            continue

                    with self.rt.lock:
                        gpu = self.rt.get_available_gpu()
                        if gpu is None:
                            if item.get("task_type") == "stacking":
                                blocked_stacking.append(item)
                            else:
                                deferred_items.append(item)
                            no_gpu = True
                            continue
                        self.rt.add_experiment_to_gpu(gpu, item["id"])
                        item["slot_index"] = max(0, len(self.rt.running_gpus.get(gpu, [])) - 1)

                    threading.Thread(
                        target=self.run_agent_in_thread,
                        args=(item["id"], item.get("prompt", ""), item.get("base_solution", ""), gpu),
                        kwargs={
                            "reference_code": item.get("reference_code"),
                            "task_type": item.get("task_type", "experiment"),
                            "task_payload": item,
                        },
                        daemon=True,
                    ).start()
                    started_any = True
                    time.sleep(0.5)

                # --- Put back blocked stacking & deferred items ---
                with self.rt.lock:
                    for item in reversed(deferred_items):
                        if item.get("auto"):
                            self.rt.auto_queue.appendleft(item)
                        else:
                            self.rt.manual_queue.appendleft(item)
                    for item in blocked_stacking:
                        if item.get("auto"):
                            self.rt.auto_queue.append(item)
                        else:
                            self.rt.manual_queue.append(item)

                if not started_any:
                    # Count only non-blocked items for idea generation decision
                    with self.rt.lock:
                        non_blocked_queued = sum(
                            1 for item in list(self.rt.manual_queue) + list(self.rt.auto_queue)
                            if item.get("task_type") != "stacking"
                        )
                        total_blocked = len(blocked_stacking)
                    if non_blocked_queued > 0:
                        # There are runnable non-stacking items but no GPU — just wait
                        time.sleep(2)
                        continue
                    if total_blocked > 0 and non_blocked_queued == 0:
                        # Only blocked stacking items remain — need to generate new ideas
                        pass  # fall through to idea generation

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
                        # Don't count blocked stacking items as "queued" for slot calculation
                        queued_runnable = sum(
                            1 for item in list(self.rt.manual_queue) + list(self.rt.auto_queue)
                            if item.get("task_type") != "stacking"
                        )
                        queued_auto_count = len(self.rt.auto_queue)
                        needed = total_slots - running_count - queued_runnable
                        max_auto_to_add = MAX_AUTO_QUEUE_SIZE - max(0, queued_auto_count - total_blocked)
                        needed = min(needed, max_auto_to_add)

                    if needed <= 0:
                        time.sleep(2)
                        continue

                    self.orchestrator_log(
                        f"Getting ideas (used: {len(used_idea_names)}, need {needed}, stacking_mode={self.rt.cfg.enable_stacking_mode})..."
                    )
                    unused = feeder.get_unused_prompts(used_idea_names, limit=needed)
                    self.orchestrator_log(f"Got {len(unused)} idea(s) from feeder")
                    unused = self._filter_and_prioritize_auto_ideas(unused, needed)
                    self.orchestrator_log(f"After filtering: {len(unused)} idea(s)")
                    queued_now = 0
                    for idx, idea in enumerate(unused):
                        if queue_idea(self.rt, self.rt.cfg, idea, idx, self.resolve_base_solution, self.orchestrator_log):
                            queued_now += 1
                    if queued_now:
                        self.orchestrator_log(f"Queued total: {queued_now}/{len(unused)}")
                    else:
                        self.orchestrator_log(f"WARNING: No ideas queued! unused={len(unused)}")
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

    def _is_claude_active_in_container(self, container_name: str) -> bool:
        """Return True if claude process is currently running inside container."""
        try:
            result = subprocess.run(
                self.docker_cmd + ["exec", container_name, "sh", "-c", "pgrep -x claude >/dev/null 2>&1"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

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
        if task_type == "oof_fold":
            parent_exp = str(exp.get("parent_experiment") or "") if exp else ""
            oof_pred = out / "oof_fold_predictions.parquet"
            status = "completed" if oof_pred.exists() else "failed"
            notes = "recovered OOF after restart" if status == "completed" else "recovered OOF without predictions"
            fold_cv = None
            metrics_fp = out / "run_output" / "metrics.json"
            if metrics_fp.exists():
                try:
                    metrics = json.loads(metrics_fp.read_text(errors="replace"))
                    raw_cv = metrics.get(cfg.val_metric_key)
                    if isinstance(raw_cv, (int, float)):
                        fold_cv = float(raw_cv)
                except Exception:
                    fold_cv = None
            recovered_exit_code = exp.get("exit_code") if exp else None
            if recovered_exit_code is None:
                recovered_exit_code = 0 if status == "completed" else -3
            db.update_experiment(
                exp_name,
                status=status,
                finished_at=datetime.now().isoformat(),
                exit_code=recovered_exit_code,
                elapsed_min=elapsed_min,
                notes=notes,
                cv_score=fold_cv,
            )
            db.add_log(exp_name, f"Recovered OOF: {status}, pred_exists={oof_pred.exists()}")
            if status == "completed" and parent_exp:
                self._aggregate_parent_oof(parent_exp)
            self._release_oof_lock_if_done(parent_exp)
            return status, None

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
            db.add_log(exp_name, f"★ {notes}", level="success")
            if cfg.notify_on_improvement:
                self._notify(
                    f"🎉 <b>NEW BEST SCORE! (recovered)</b>\n\n"
                    f"📊 Project: <b>{cfg.name}</b>\n"
                    f"📈 Score: <b>{test:.6f}</b> (was {best_score:.6f})\n"
                    f"🧪 Experiment: <code>{exp_name}</code>"
                )

        submission_path = out / "submission.parquet"
        has_submission = submission_path.exists()
        has_eval = bool(ev)
        has_result = self._has_result_event_any(exp_dir, out)
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
        post_result_grace = 180
        exp_dir = self.rt.cfg.experiments_dir / exp_name
        out_dir = exp_dir / "output"
        events_file = self._events_file(exp_dir)  # Agent writes to /app/events/events.jsonl
        agent_done_logged = False
        agent_done_at = None
        last_cleanup_at = 0.0
        agent_term_sent = False
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

            claude_active = self._is_claude_active_in_container(container_name)
            if not claude_active and self._has_result_event_any(exp_dir, out_dir):
                db.add_log(
                    exp_name,
                    "Recovery: claude process is not active and result is present; finalizing experiment",
                    level="warning",
                )
                self._force_finish_container_after_result(exp_name, container_name)
                break

            if not agent_done_logged and self._has_result_event_any(exp_dir, out_dir):
                agent_done_logged = True
                agent_done_at = time.time()
                db.add_log(exp_name, "Agent finished (found 'result' in events.jsonl), waiting for container to exit...")
                if self._cleanup_post_result_processes(exp_name, container_name):
                    last_cleanup_at = time.time()
            if agent_done_logged and (time.time() - last_cleanup_at) >= 30:
                if self._cleanup_post_result_processes(exp_name, container_name):
                    last_cleanup_at = time.time()
            if agent_done_at and not agent_term_sent and (time.time() - agent_done_at) > post_result_grace:
                agent_term_sent = self._terminate_post_result_agent(exp_name, container_name)
            if agent_done_at and (time.time() - agent_done_at) > post_result_grace:
                self._force_finish_container_after_result(exp_name, container_name)

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
            r = subprocess.run(self.docker_cmd + ["ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5)
            active_containers = {
                cn for cn in (r.stdout.strip().split("\n") if r.stdout.strip() else set())
                if cn.startswith("agent-") or cn.startswith("oof-")
            }
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
            task_type = exp.get("task_type") or "experiment"
            cn = self._find_container_name(exp_name, active_containers)
            if cn:
                if task_type == "oof_fold":
                    if gpu_id is not None:
                        with self.rt.lock:
                            self.rt.add_experiment_to_gpu(gpu_id, exp_name)
                    db.add_log(exp_name, f"Resuming monitoring for orphaned OOF fold on GPU {gpu_id}")
                    threading.Thread(target=self._monitor_orphaned_oof_container, args=(exp_name, cn, gpu_id), daemon=True).start()
                    monitored += 1
                    continue
                exp_dir = self.rt.cfg.experiments_dir / exp_name
                if not self._is_claude_active_in_container(cn) and self._has_result_event_any(exp_dir, exp_dir / "output"):
                    db.add_log(exp_name, "Recovery: container found but claude is already inactive; finalizing immediately")
                    self._finalize_experiment(exp_name, best_score)
                    recovered += 1
                    continue
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

    def _monitor_orphaned_oof_container(self, exp_name, container_name, gpu_id):
        db.add_log(exp_name, f"Resumed OOF monitoring (container {container_name})")
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
            time.sleep(10)

        direction = self.rt.cfg.best_score_sort_key()
        best_score = db.get_stats(direction).get("best_score", 0) or 0
        self._finalize_experiment(exp_name, best_score)
        with self.rt.lock:
            self.rt.remove_experiment_from_gpu(gpu_id, exp_name)
