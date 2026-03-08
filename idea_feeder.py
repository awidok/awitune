"""
Experiment idea feeder for the dashboard auto-queue.

All ideas are generated on-demand via LLM (generate_ideas.py).
No static prompts - everything is dynamic based on experiment history.
"""

import traceback

from .config import ProjectConfig


def configure(cfg: ProjectConfig):
    """Configure the idea feeder with project config."""
    global _cfg
    _cfg = cfg


_cfg = None


def _try_llm_generation(count: int = 3) -> list:
    """Generate new ideas via LLM."""
    try:
        if _cfg is None:
            print("[idea_feeder] No config set, cannot generate ideas")
            return []
        from .generate_ideas import generate_ideas
        return generate_ideas(count=count)
    except Exception as e:
        print(f"[idea_feeder] LLM generation failed: {e}")
        traceback.print_exc()
        return []


def get_all_prompts():
    """Return all available experiment prompts (generated on-demand via LLM)."""
    ideas = _try_llm_generation(count=3)
    return ideas


def get_unused_prompts(used_names: set, limit: int = 3) -> list:
    """Return unused prompts - always generate fresh via LLM."""
    new_ideas = _try_llm_generation(count=limit)
    if new_ideas:
        unused_new = [i for i in new_ideas if i["name"] not in used_names]
        if unused_new:
            return unused_new[:limit]
    return []
