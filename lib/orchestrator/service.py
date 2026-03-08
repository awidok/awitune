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
            prev_exps = db.get_all_experiments(limit=20, status="completed")

            if is_analysis:
                ws = prepare_analyst_workspace(cfg, exp_dir, prompt)
            else:
                effective_prompt = prompt
                if is_stacking:
                    effective_prompt = self._build_stacking_prompt(
                        prompt,
                        stack_sources=(task_payload or {}).get("stack_sources") or [],
                    )
                ws = prepare_workspace(
                    cfg,
                    base_solution or str(cfg.solutions_dir / "baseline"),
                    exp_dir,
                    effective_prompt,
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
                    if agent_done_at is None and has_result_event(events_file):
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

        rows = self._load_oof_registry()
        rows = [row for row in rows if Path(row.get("path", "")).exists()]
        existing = {(row.get("experiment"), row.get("path")) for row in rows}

        for fp in found:
            exp = db.get_experiment(exp_name) or {}
            row = {
                "experiment": exp_name,
                "path": str(fp.resolve()),
                "score": test_score,
                "updated_at": datetime.now().isoformat(),
                "splitter": self.rt.cfg.stacking_oof_splitter,
                "folds": int(self.rt.cfg.stacking_oof_folds),
                "family": self._infer_family({"name": exp_name, "task_type": exp.get("task_type", "")}),
            }
            key = (row["experiment"], row["path"])
            if key not in existing:
                rows.append(row)
                existing.add(key)

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
        available = [r for r in rows if Path(r.get("path", "")).exists()]
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
            "## STACKING MODE (orchestrator override)\n"
            "This run is explicitly for stacking/blending. You may ignore generic 'NO stacking' rules.\n"
            f"Use {self.rt.cfg.stacking_oof_splitter} with {self.rt.cfg.stacking_oof_folds} folds for any new OOF generation.\n"
            "Reuse OOF predictions from previous experiments whenever possible.\n"
            "Prioritize blending diverse families (tree/transformer/dcn/mlp), not variants of one family.\n"
            "Save OOF table to `/app/output/oof_predictions.parquet` (customer_id + predict_* columns).\n"
            "Save blender report in `/app/output/report.md` including which OOF sources were used.\n\n"
            "### Registered OOF predictions\n"
            f"{oof_block}\n"
        )
        if user_prompt:
            return f"{stacking_guidance}\n\n## Stacking task details\n{user_prompt}"
        return stacking_guidance

    def _extract_stacking_sources(self, item: dict) -> list[str]:
        direct = item.get("stack_sources")
        if isinstance(direct, list):
            sources = [str(x).strip() for x in direct if str(x).strip()]
            if sources:
                return sources

        base_experiment = str(item.get("base_experiment") or "").strip()
        if "," in base_experiment:
            sources = [x.strip() for x in base_experiment.split(",") if x.strip()]
            if sources:
                return sources

        prompt = str(item.get("prompt") or "")
        if not prompt:
            return []
        all_names = {e.get("name", "") for e in db.get_all_experiments(limit=2000)}
        detected = [name for name in all_names if name and name in prompt]
        return sorted(set(detected))

    def _has_ready_oof_for_experiment(self, exp_name: str) -> bool:
        rows = self._load_oof_registry()
        for row in rows:
            if str(row.get("experiment") or "") != exp_name:
                continue
            if Path(str(row.get("path") or "")).exists():
                return True
        return False

    def _build_oof_runner_script(self) -> str:
        return """import argparse
import subprocess
from pathlib import Path

import numpy as np
import polars as pl


def get_splitter(y, n_splits: int):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return splitter.split(np.zeros((len(y), 1)), y)
    except Exception:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return splitter.split(np.arange(len(y)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--n-folds", type=int, required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "local_train.parquet"
    df = pl.read_parquet(str(train_path))
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if not target_cols:
        raise RuntimeError("No target_* columns found in local_train.parquet")

    y = df.select(target_cols).to_numpy()
    splits = list(get_splitter(y, args.n_folds))
    if args.fold_idx < 0 or args.fold_idx >= len(splits):
        raise RuntimeError(f"Invalid fold index {args.fold_idx} for {len(splits)} splits")

    train_idx, val_idx = splits[args.fold_idx]
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


if __name__ == "__main__":
    main()
"""

    def _enqueue_external_oof_folds_for_source(self, parent_exp_name: str, requester: str = ""):
        parent = db.get_experiment(parent_exp_name) or {}
        if not parent.get("workspace_dir"):
            if requester:
                db.add_log(requester, f"Cannot queue OOF for {parent_exp_name}: missing workspace", level="warning")
            return
        existing = db.get_all_experiments(limit=3000)
        existing_for_parent_active = [
            e for e in existing if e.get("task_type") == "oof_fold" and e.get("parent_experiment") == parent_exp_name
            and e.get("status") in ("queued", "running")
        ]
        if existing_for_parent_active:
            return

        parent_dir = self.rt.cfg.experiments_dir / parent_exp_name
        oof_runner_path = parent_dir / "oof_runner.py"
        oof_runner_path.write_text(self._build_oof_runner_script())

        n_folds = int(self.rt.cfg.stacking_oof_folds)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for fold_idx in range(n_folds):
            eid = f"oof_{parent_exp_name}_f{fold_idx}_{ts}"
            db.create_experiment(
                eid,
                prompt=f"External OOF fold {fold_idx}/{n_folds - 1} for {parent_exp_name}",
                base_solution=str(parent.get("workspace_dir", "")),
                parent_experiment=parent_exp_name,
                task_type="oof_fold",
            )
            db.add_log(
                eid,
                f"Queued external OOF fold {fold_idx + 1}/{n_folds} for {parent_exp_name}"
                + (f" (requested by {requester})" if requester else ""),
            )
            item = {
                "id": eid,
                "prompt": "",
                "base_solution": str(parent.get("workspace_dir", "")),
                "task_type": "oof_fold",
                "auto": True,
                "parent_experiment": parent_exp_name,
                "fold_idx": fold_idx,
                "n_folds": n_folds,
                "oof_runner_path": str(oof_runner_path),
            }
            with self.rt.lock:
                self.rt.auto_queue.append(item)
        self.orchestrator_log(f"Queued {n_folds} external OOF fold task(s) for {parent_exp_name}")

    def _ensure_stacking_dependencies(self, item: dict) -> bool:
        if item.get("task_type") != "stacking":
            return True
        sources = self._extract_stacking_sources(item)
        if not sources:
            db.add_log(item["id"], "Stacking has no explicit sources; running without OOF dependencies", level="warning")
            return True
        item["stack_sources"] = sources

        missing = []
        for source_exp in sources:
            if self._has_ready_oof_for_experiment(source_exp):
                continue
            missing.append(source_exp)
            self._enqueue_external_oof_folds_for_source(source_exp, requester=item["id"])

        if missing:
            db.add_log(
                item["id"],
                f"Waiting for OOF dependencies: {', '.join(missing)}",
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
        if len(fold_exps) < n_folds:
            return

        parts = []
        fold_cv_scores = []
        for e in fold_exps:
            fp = self.rt.cfg.experiments_dir / e["name"] / "output" / "oof_fold_predictions.parquet"
            if fp.exists():
                parts.append(pl.read_parquet(str(fp)))
            if isinstance(e.get("cv_score"), (int, float)):
                fold_cv_scores.append(float(e["cv_score"]))
        if len(parts) < n_folds:
            return

        merged = pl.concat(parts, how="vertical")
        if "customer_id" in merged.columns:
            merged = merged.unique(subset=["customer_id"], keep="first")

        parent_out = self.rt.cfg.experiments_dir / parent_exp_name / "output"
        parent_out.mkdir(parents=True, exist_ok=True)
        final_oof = parent_out / "oof_predictions.parquet"
        merged.write_parquet(str(final_oof))
        parent = db.get_experiment(parent_exp_name) or {}
        self._register_oof_predictions(parent_exp_name, parent_out, parent.get("test_score"))
        if fold_cv_scores:
            cv_score = float(sum(fold_cv_scores) / len(fold_cv_scores))
            db.update_experiment(parent_exp_name, cv_score=cv_score)
            db.add_log(parent_exp_name, f"Updated CV score from OOF folds: {cv_score:.6f}")
        db.add_log(parent_exp_name, f"External OOF merged from {len(parts)} fold(s): {final_oof}")

    def _run_oof_fold_in_thread(self, exp_name: str, gpu_id: int, task_payload: dict):
        cfg = self.rt.cfg
        exp_dir = cfg.experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        out = exp_dir / "output"
        out.mkdir(parents=True, exist_ok=True)

        parent_exp = str(task_payload.get("parent_experiment") or "")
        fold_idx = int(task_payload.get("fold_idx", 0))
        n_folds = int(task_payload.get("n_folds", max(2, int(cfg.stacking_oof_folds))))
        runner_path = Path(str(task_payload.get("oof_runner_path") or ""))
        parent = db.get_experiment(parent_exp) or {}
        parent_ws = Path(str(parent.get("workspace_dir") or ""))

        db.update_experiment(exp_name, status="running", gpu_id=gpu_id, started_at=datetime.now().isoformat(), exp_dir=str(exp_dir))
        db.add_log(exp_name, f"Starting external OOF fold {fold_idx + 1}/{n_folds} on GPU {gpu_id}")
        t0 = time.time()
        exit_code = -1
        try:
            if not parent_ws.exists() or not (parent_ws / "run.py").exists():
                raise RuntimeError(f"Parent workspace/run.py not found for {parent_exp}")
            if not runner_path.exists():
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
            ]

            log_path = exp_dir / "oof_fold.log"
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                deadline = time.time() + cfg.timeout_minutes * 60
                while True:
                    ret = proc.poll()
                    if ret is not None:
                        exit_code = ret
                        break
                    if time.time() > deadline:
                        subprocess.run(self.docker_cmd + ["stop", "-t", "30", cn], capture_output=True, timeout=60)
                        proc.wait(timeout=60)
                        exit_code = -1
                        break
                    time.sleep(5)

            oof_pred = out / "oof_fold_predictions.parquet"
            status = "completed" if exit_code == 0 and oof_pred.exists() else "failed"
            notes = f"external OOF fold {fold_idx + 1}/{n_folds}"
            if status == "failed" and not oof_pred.exists():
                notes = "failed: missing oof_fold_predictions.parquet"
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
            db.add_log(exp_name, f"OOF fold finished with status={status}, exit={exit_code}")
            if status == "completed" and parent_exp:
                self._aggregate_parent_oof(parent_exp)
        except Exception as exc:
            db.update_experiment(
                exp_name,
                status="failed",
                finished_at=datetime.now().isoformat(),
                exit_code=-2,
                elapsed_min=round((time.time() - t0) / 60, 2),
                notes=f"OOF ERROR: {exc}",
            )
            db.add_log(exp_name, f"OOF ERROR: {exc}", level="error")
        finally:
            with self.rt.lock:
                self.rt.remove_experiment_from_gpu(gpu_id, exp_name)

    def worker_loop(self):
        last_sync_at = 0.0
        while self.rt.worker_running:
            try:
                now = time.time()
                if now - last_sync_at >= 15:
                    self.rt.sync_running_from_docker()
                    last_sync_at = now

                started_any = False
                blocked_in_cycle = 0
                while True:
                    item = None
                    with self.rt.lock:
                        if self.rt.manual_queue:
                            item = self.rt.manual_queue.popleft()
                        elif self.rt.auto_queue:
                            item = self.rt.auto_queue.popleft()
                    if item is None:
                        break

                    if item.get("task_type") == "stacking":
                        if not self._ensure_stacking_dependencies(item):
                            blocked_in_cycle += 1
                            with self.rt.lock:
                                if item.get("auto"):
                                    self.rt.auto_queue.append(item)
                                else:
                                    self.rt.manual_queue.append(item)
                            with self.rt.lock:
                                total_queued = len(self.rt.manual_queue) + len(self.rt.auto_queue)
                            if total_queued <= blocked_in_cycle:
                                break
                            continue

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
                            "task_payload": item,
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

                    self.orchestrator_log(
                        f"Getting ideas (used: {len(used_idea_names)}, need {needed}, stacking_mode={self.rt.cfg.enable_stacking_mode})..."
                    )
                    unused = feeder.get_unused_prompts(used_idea_names, limit=needed)
                    unused = self._filter_and_prioritize_auto_ideas(unused, needed)
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
        post_result_grace = 180
        events_file = self.rt.cfg.experiments_dir / exp_name / "output" / "events.jsonl"
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

            if not agent_done_logged and has_result_event(events_file):
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
