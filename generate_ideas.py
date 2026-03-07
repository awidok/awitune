"""
LLM-powered meta-agent for generating experiment ideas.

Generic version: the system prompt is task-agnostic.
Project-specific context (README, evaluation details) comes from config.

Usage (via CLI):
    python -m awitune.generate_ideas --project ./my_project --count 3
"""

import argparse
import difflib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests
import urllib3
from dotenv import load_dotenv

from . import db
from .config import ProjectConfig, load_config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

_cfg: ProjectConfig | None = None

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


SYSTEM_PROMPT = """\
You are an ML research lead managing experiment optimization for a machine learning task.
Your job is to analyze the current state of experiments and decide what to try next.

You receive:
- The task description
- A condensed experiment history with metric analysis
- An automated "what worked / what didn't" summary
- Source code of the best solution + diffs from other top solutions
- Reports from recent experiments
- Analyst reports with data insights (if any)
- Currently running and queued experiments (to avoid duplication)

You can propose TWO types of tasks:

### Type 1: analysis (PREFERRED when stuck or starting)
A data exploration run. The agent explores the dataset and writes a report.
ALWAYS prioritize analysis when:
- No analyst reports exist yet
- Score has plateaued and you need fresh insights
- You suspect there are unexplored patterns in the data
- Recent experiments failed and you need to understand why

Analysis should investigate:
- Feature distributions, correlations, and importance
- Target patterns and imbalances
- NaN patterns and missing data structure
- Potential data quality issues
- Feature interactions and engineering opportunities
- Model error analysis (which samples are misclassified and why)

### Type 2: experiment
A model training run. The agent gets a workspace with run.py and trains models.
Use when you have clear hypotheses from analysis or want to test specific ideas.

**CRITICAL: Diversify your solution portfolio!**
The goal is to build MULTIPLE diverse solutions that can later be stacked. You should:

1. **Maintain solution diversity** — track different "families" of approaches:
   - DCNv2-based (current baseline)
   - Transformer-based (attention over features)
   - Tree-based (LightGBM, XGBoost, CatBoost)
   - TabNet / TabTransformer
   - Simple MLP with different encodings
   - Linear models with feature engineering

2. **Improve EACH solution independently** — don't just improve the best one:
   - If DCNv2 is best, still improve the Transformer solution
   - If tree-based is weak, try to make it competitive
   - Each solution family should have its own improvement trajectory

3. **Propose ALTERNATIVE approaches** — not just incremental improvements:
   - If current best is DCNv2, propose a Transformer from scratch
   - If all solutions are neural, propose a tree-based approach
   - If all solutions use same features, propose different feature engineering

4. **Track solution families** — when proposing, consider:
   - How many solutions exist in each family?
   - Which families are underexplored?
   - Which families showed promise but need more work?

For experiments you can:
1. Start from the best solution and modify it (incremental)
2. Start from a different solution that showed promise (alternative trajectory)
3. Create a NEW solution from scratch with different architecture (diversification)
4. Combine ideas from multiple solutions (cross-pollination)

IMPORTANT RULES:
- ANALYSIS FIRST: If no recent analyst reports exist, propose analysis before experiments
- DIVERSIFY: Build a portfolio of DIFFERENT solution families (DCNv2, Transformer, Tree-based, etc.)
- IMPROVE ALL FAMILIES: Don't just improve the best — make each family competitive
- PROPOSE ALTERNATIVES: If all solutions are similar, propose a fundamentally different architecture
- Be specific and actionable — give concrete code changes, not vague suggestions
- Learn from failures — if something was tried and didn't work, don't repeat it
- NEVER propose something already running or queued
- NEVER propose a minor variation of a failed approach
- Focus on approaches that are FUNDAMENTALLY different from what's been tried
- If the score has plateaued, run analysis to find new directions OR try a different solution family
- Each experiment must be a single self-contained run.py — train and predict in one script
- Propose at most 2 analysis tasks per batch
- NO STACKING/BLENDING: Do NOT propose stacking, blending, or ensemble averaging experiments
  - The baseline already has an ensemble. Focus on improving individual models.
  - Stacking/blending will be handled by a separate orchestration tool.
  - Focus on: architecture improvements, loss functions, feature engineering, regularization, training tricks

Respond with a JSON array of task objects. Each object must have:
{
  "name": "snake_case_short_name (max 30 chars)",
                "type": "object",
                "properties": {},
  "base_experiment": "name of experiment to use as starting workspace, or 'default'",
  "prompt": "Detailed instructions with code changes.",
  "reference_code": null or {
    "experiment": "name of another experiment whose code contains useful ideas",
    "what_to_take": "What specifically to incorporate"
  }
}

Return ONLY valid JSON — no markdown fences, no commentary outside the array.
"""


def configure(cfg: ProjectConfig):
    global _cfg
    _cfg = cfg


def collect_context() -> dict:
    db.init_db()

    readme = ""
    readme_path = _cfg.project_dir / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(errors="replace")

    direction = _cfg.best_score_sort_key()
    all_completed = db.get_all_experiments(limit=200, status="completed")
    best_exp = db.get_best_experiment(direction)
    best_score = best_exp.get("test_score", 0) if best_exp else 0

    running = db.get_all_experiments(limit=50, status="running")
    queued = db.get_all_experiments(limit=50, status="queued")

    scored = sorted(
        [e for e in all_completed if e.get("test_score") and e["test_score"] > 0],
        key=lambda e: e["test_score"],
        reverse=(_cfg.metric_direction == "maximize"),
    )

    reports = {}
    for e in (all_completed[:5] + scored[:5]):
        name = e["name"]
        if name in reports:
            continue
        output_dir = e.get("output_dir")
        if not output_dir:
            continue
        report_path = Path(output_dir) / "report.md"
        if report_path.exists():
            reports[name] = report_path.read_text(errors="replace")

    code_snippets = {}
    for exp in scored[:5]:
        ws_dir = exp.get("workspace_dir")
        if not ws_dir:
            continue
        run_py = Path(ws_dir) / "run.py"
        if run_py.exists():
            code_snippets[exp["name"]] = {
                "code": run_py.read_text(errors="replace"),
                "score": exp["test_score"],
            }
    if not code_snippets:
        baseline_run = _cfg.solutions_dir / "baseline" / "run.py"
        if baseline_run.exists():
            code_snippets["baseline (default)"] = {
                "code": baseline_run.read_text(errors="replace"),
                "score": best_score,
            }

    return {
        "readme": readme,
        "experiments": all_completed,
        "scored": scored,
        "best_score": best_score,
        "best_experiment": best_exp,
        "running": running,
        "queued": queued,
        "reports": reports,
        "code_snippets": code_snippets,
    }


def _build_analysis_summary(scored: list) -> str:
    if len(scored) < 2:
        return ""
    lines = ["## Analysis: What Worked and What Didn't\n"]
    improved = [e for e in scored if e.get("improved")]
    not_improved = [e for e in scored if not e.get("improved")]
    if improved:
        lines.append("### Approaches that IMPROVED the score:")
        for e in improved[:10]:
            prompt_short = (e.get("prompt") or "")[:80].replace("\n", " ")
            lines.append(f"- **{e['name']}** ({e['test_score']:.6f}): {prompt_short}")
        lines.append("")
    if not_improved:
        lines.append("### Approaches that DID NOT improve:")
        for e in not_improved[:15]:
            prompt_short = (e.get("prompt") or "")[:80].replace("\n", " ")
            lines.append(f"- **{e['name']}** ({e.get('test_score', 'N/A')}): {prompt_short}")
        lines.append("")
    if len(scored) >= 3:
        lines.append("### Score progression (chronological):")
        by_time = sorted(scored, key=lambda e: e.get("created_at", ""))
        for e in by_time[-15:]:
            imp = " ★" if e.get("improved") else ""
            lines.append(f"  {e['test_score']:.6f}{imp}  {e['name']}")
        lines.append("")
    return "\n".join(lines)


def _build_code_section(code_snippets: dict) -> str:
    if not code_snippets:
        return ""
    parts = ["## Source Code\n"]
    names = list(code_snippets.keys())
    best_name = names[0]
    best_info = code_snippets[best_name]
    parts.append(f"### Best solution: {best_name} (score: {best_info['score']:.6f})\n")
    parts.append(f"```python\n{best_info['code']}\n```\n")
    if len(names) > 1:
        best_lines = best_info["code"].splitlines(keepends=True)
        for other_name in names[1:]:
            other_info = code_snippets[other_name]
            other_lines = other_info["code"].splitlines(keepends=True)
            diff = list(difflib.unified_diff(best_lines, other_lines,
                                              fromfile=f"{best_name}/run.py",
                                              tofile=f"{other_name}/run.py", n=3))
            if diff:
                diff_text = "".join(diff)
                if len(diff_text) > 5000:
                    diff_text = diff_text[:5000] + "\n... (diff truncated) ...\n"
                parts.append(f"### Diff: {other_name} (score: {other_info['score']:.6f}) vs best\n")
                parts.append(f"```diff\n{diff_text}```\n")
    return "\n".join(parts)


def build_user_prompt(ctx: dict, count: int) -> str:
    parts = []
    parts.append("## Task Description\n")
    parts.append(ctx["readme"] if ctx["readme"] else "No task description available.")
    parts.append("")
    parts.append(f"## Current Best Score: {ctx['best_score']:.6f}")
    parts.append(f"## Metric: {_cfg.test_metric_key} ({_cfg.metric_direction})\n")

    running = ctx.get("running", [])
    queued = ctx.get("queued", [])
    if running or queued:
        parts.append("## Currently In Progress (DO NOT duplicate)\n")
        for e in running:
            parts.append(f"- [running] {e['name']}: {(e.get('prompt') or '')[:100]}")
        for e in queued:
            parts.append(f"- [queued] {e['name']}: {(e.get('prompt') or '')[:100]}")
        parts.append("")

    experiments = ctx["experiments"]
    if experiments:
        parts.append("## Experiment History\n")
        parts.append("| # | Name | Score | Improved? | Approach |")
        parts.append("|---|------|-------|-----------|----------|")
        for i, exp in enumerate(experiments):
            score = f"{exp['test_score']:.6f}" if exp.get("test_score") else "N/A"
            imp = "YES" if exp.get("improved") else "no"
            prompt_short = (exp.get("prompt") or "")[:80].replace("|", "/").replace("\n", " ")
            parts.append(f"| {i+1} | {exp['name']} | {score} | {imp} | {prompt_short} |")
        parts.append("")

    analysis = _build_analysis_summary(ctx["scored"])
    if analysis:
        parts.append(analysis)

    code_section = _build_code_section(ctx["code_snippets"])
    if code_section:
        parts.append(code_section)

    if ctx["reports"]:
        parts.append("## Experiment Reports\n")
        for name, report in list(ctx["reports"].items())[:5]:
            if len(report) > 3000:
                report = report[:3000] + "\n... (truncated) ...\n"
            parts.append(f"### {name}\n{report}\n")

    parts.append(f"\n## Your Task\nPropose {count} task(s). Return JSON array with {count} object(s).\n"
                 f"Check 'Currently In Progress' — do NOT duplicate those.")
    return "\n".join(parts)


def call_openai_api(system: str, user: str, temperature: float = 0.7) -> str:
    url = f"{OPENAI_API_BASE}/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 8000,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, timeout=300, verify=False)
    response.raise_for_status()
    data = response.json()
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    return message.get("content", "") or message.get("reasoning_content", "")


def parse_ideas(response_text: str) -> list[dict]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1:
        text = text[bracket_start:bracket_end + 1]
    ideas = json.loads(text)
    if not isinstance(ideas, list):
        ideas = [ideas]
    validated = []
    for idea in ideas:
        validated.append({
            "name": str(idea.get("name", "unnamed")),
            "task_type": str(idea.get("task_type", "experiment")),
            "reasoning": str(idea.get("reasoning", "")),
            "base_experiment": str(idea.get("base_experiment", "default")),
            "prompt": str(idea.get("prompt", "")),
            "reference_code": idea.get("reference_code"),
        })
    return validated


def generate_ideas(count: int = 1) -> list[dict]:
    """Generate experiment ideas via LLM and return them (no caching)."""
    print(f"Generating {count} experiment idea(s) via LLM...")
    ctx = collect_context()
    user_prompt = build_user_prompt(ctx, count)
    print(f"  Context: {len(ctx['experiments'])} completed, prompt: {len(user_prompt)} chars")
    try:
        response_text = call_openai_api(SYSTEM_PROMPT, user_prompt)
        print(f"  LLM response: {len(response_text)} chars")
        if not response_text:
            print("  WARNING: Empty response from LLM")
            return []
        ideas = parse_ideas(response_text)
        print(f"  Parsed {len(ideas)} idea(s)")
        return ideas
    except Exception as e:
        print(f"  ERROR in generate_ideas: {e}")
        traceback.print_exc()
        return []
