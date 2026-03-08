"""Queue and auto-feed helpers for orchestration runtime."""

from datetime import datetime

from . import db


def collect_used_idea_names(rt) -> set:
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

    return used_idea_names


def queue_idea(rt, cfg, idea: dict, idx: int, resolve_base_solution, orchestrator_log) -> bool:
    try:
        if not isinstance(idea, dict):
            orchestrator_log(f"Skipping invalid idea at index {idx}: not a dict")
            return False

        idea_name = str(idea.get("name") or "").strip()
        idea_prompt = str(idea.get("prompt") or "").strip()
        if not idea_name or not idea_prompt:
            orchestrator_log(f"Skipping invalid idea at index {idx}: missing name or prompt")
            return False

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        eid = f"auto_{idea_name}_{ts}_{idx}"
        base_exp = str(idea.get("base_experiment", "default") or "default")
        base_sol = resolve_base_solution(base_exp)
        parent = base_exp if base_exp and base_exp != "default" else ""
        idea_task_type = str(idea.get("task_type", "experiment") or "experiment")
        idea_reasoning = str(idea.get("reasoning", "") or "")

        db.create_experiment(
            eid,
            prompt=idea_prompt[:500],
            base_solution=base_sol,
            parent_experiment=parent,
            task_type=idea_task_type,
        )
        db.add_log(eid, f"Auto-queued {idea_task_type}: {idea_name}")
        orchestrator_log(f"Queued: {eid} ({idea_task_type}) - {idea_reasoning[:100]}")
        item = {
            "id": eid,
            "prompt": idea_prompt,
            "base_solution": base_sol,
            "idea_name": idea_name,
            "auto": True,
            "reference_code": idea.get("reference_code"),
            "task_type": idea_task_type,
        }
        with rt.lock:
            rt.auto_queue.append(item)
        return True
    except Exception as e:
        orchestrator_log(f"Failed to queue idea at index {idx}: {e}")
        return False
