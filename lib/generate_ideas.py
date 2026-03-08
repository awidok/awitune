"""
LLM-powered meta-agent for generating experiment ideas.

Uses function calling to fetch detailed information on-demand,
reducing context size from ~70K to ~5K characters.

Usage (via CLI):
    python -m awitune.generate_ideas --project ./my_project --count 3
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import requests
import urllib3
from dotenv import load_dotenv

from . import db
from .config import ProjectConfig, load_config
from .orchestrator.tools import TOOLS, dispatch_tool_call, configure as configure_tools

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

_cfg: ProjectConfig | None = None

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


SYSTEM_PROMPT = """\
You are an ML research lead managing experiment optimization for a machine learning task.
Your job is to analyze the current state of experiments and decide what to try next.

You have access to TOOLS to fetch detailed information on-demand. Use them wisely:
- First, review the experiment summary table to understand what's been tried
- Then, use tools to get details about promising or failed experiments
- Read analyst reports to understand data patterns
- Check reference code for proven techniques

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

4. **Include small tuning experiments too** — not every run should be a big refactor:
   - Also propose lightweight experiments focused on hyperparameters/training recipe
   - Examples: LR/scheduler, weight decay, dropout, batch size, warmup, EMA/SWA, loss coefficients
   - These should be low-risk, fast-to-validate, and easy to compare

IMPORTANT RULES:
- ANALYSIS FIRST: If no recent analyst reports exist, propose analysis before experiments
- DIVERSIFY: Build a portfolio of DIFFERENT solution families
- USE TOOLS: Fetch details on-demand instead of guessing
- Learn from failures — if something was tried and didn't work, don't repeat it
- NEVER propose something already running or queued
- Focus on approaches that are FUNDAMENTALLY different from what's been tried
- Keep a MIX of proposal scope: some architecture-level ideas + some small hyperparameter tuning ideas
- NO STACKING/BLENDING: Focus on improving individual models

Respond with a JSON array of task objects. Each object must have:
{
  "name": "snake_case_short_name (max 30 chars)",
  "task_type": "experiment" or "analysis",
  "reasoning": "1-3 sentences explaining why this should work",
  "base_experiment": "name of experiment to use as starting workspace, or 'default'",
  "prompt": "Detailed instructions with code changes.",
  "reference_code": null or {
    "experiment": "name of another experiment whose code contains useful ideas",
    "what_to_take": "What specifically to incorporate"
  },
  "stack_sources": ["exp_A", "exp_B", "exp_C"]
}
`stack_sources` is optional for non-stacking tasks, but required for `task_type="stacking"`.

Return ONLY valid JSON — no markdown fences, no commentary outside the array.
"""


def configure(cfg: ProjectConfig):
    """Configure the idea generator with project config."""
    global _cfg
    _cfg = cfg
    configure_tools(cfg)


def build_system_prompt() -> str:
    prompt = SYSTEM_PROMPT
    if not _cfg or not _cfg.enable_stacking_mode:
        return prompt

    prompt = prompt.replace(
        '- NO STACKING/BLENDING: Focus on improving individual models',
        (
            "- STACKING MODE ENABLED: you may propose limited stacking/blending ideas when useful.\n"
            "- Keep base-model work dominant: at least half of proposed tasks should be non-stacking.\n"
            "- If proposing stacking, reuse existing OOF predictions from prior experiments where possible."
        ),
    )
    prompt = prompt.replace(
        '"task_type": "experiment" or "analysis",',
        '"task_type": "experiment", "analysis", or "stacking",',
    )
    prompt += (
        "\nIf task_type is 'stacking', always include stack_sources with 2-5 experiment names "
        "that should be stacked together."
    )
    return prompt


def build_compact_context() -> dict:
    """Build a compact context with just summaries - agent can fetch details via tools."""
    db.init_db()

    readme = ""
    readme_path = _cfg.project_dir / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(errors="replace")
        # Truncate to first 2000 chars
        if len(readme) > 2000:
            readme = readme[:2000] + "\n... (truncated, full README available in project)"

    direction = _cfg.best_score_sort_key()
    all_completed = db.get_all_experiments(limit=200, status="completed")
    best_exp = db.get_best_experiment(direction)
    best_score = best_exp.get("test_score", 0) if best_exp else 0

    running = db.get_all_experiments(limit=50, status="running")
    queued = db.get_all_experiments(limit=50, status="queued")

    # Build compact experiment list (just names and scores)
    experiments = []
    scored = []
    cv_scored = []
    for e in all_completed:
        exp_info = {
            "name": e.get("name", ""),
            "test_score": e.get("test_score"),
            "val_score": e.get("val_score"),
            "cv_score": e.get("cv_score"),
            "improved": e.get("improved", False),
            "status": e.get("status"),
            "created_at": e.get("created_at", ""),
        }
        experiments.append(exp_info)
        if e.get("test_score"):
            scored.append(exp_info)
        if e.get("cv_score") is not None:
            cv_scored.append(exp_info)

    # Sort by score
    if _cfg.metric_direction == "maximize":
        scored.sort(key=lambda x: x.get("test_score", 0), reverse=True)
        cv_scored.sort(key=lambda x: x.get("cv_score", 0), reverse=True)
    else:
        scored.sort(key=lambda x: x.get("test_score", float("inf")))
        cv_scored.sort(key=lambda x: x.get("cv_score", float("inf")))

    return {
        "readme": readme,
        "best_score": best_score,
        "best_experiment": best_exp.get("name") if best_exp else None,
        "running": [{"name": e.get("name"), "prompt": (e.get("prompt") or "")[:100]} for e in running],
        "queued": [{"name": e.get("name"), "prompt": (e.get("prompt") or "")[:100]} for e in queued],
        "experiments": experiments[-50:],  # Last 50 experiments
        "top_experiments": [e["name"] for e in scored[:5]],  # Top 5 names
        "top_cv_experiments": [e["name"] for e in cv_scored[:5]],
    }


def build_user_prompt(ctx: dict, count: int) -> str:
    """Build a compact user prompt - agent can fetch details via tools."""
    parts = []
    
    # Task description (truncated)
    parts.append("## Task Description\n")
    parts.append(ctx["readme"])
    parts.append("")
    
    # Current state
    parts.append(f"## Current Best Score: {ctx['best_score']:.6f}")
    if ctx.get("best_experiment"):
        parts.append(f"Best experiment: {ctx['best_experiment']}")
    parts.append(f"## Metric: {_cfg.test_metric_key} ({_cfg.metric_direction})\n")

    # Running/queued (to avoid duplication)
    running = ctx.get("running", [])
    queued = ctx.get("queued", [])
    if running or queued:
        parts.append("## Currently In Progress (DO NOT duplicate)\n")
        for e in running:
            parts.append(f"- [running] {e['name']}: {e['prompt']}")
        for e in queued:
            parts.append(f"- [queued] {e['name']}: {e['prompt']}")
        parts.append("")

    # Compact experiment table
    experiments = ctx["experiments"]
    if experiments:
        parts.append("## Experiment Summary\n")
        parts.append("Use `get_experiment_summary(name)` for details, `get_experiment_code(name)` for code.\n")
        parts.append("| Name | Test | Val | CV | Improved? |")
        parts.append("|------|------|-----|----|-----------|")
        for exp in experiments[-30:]:  # Last 30
            test_score = f"{exp['test_score']:.6f}" if exp.get("test_score") is not None else "N/A"
            val_score = f"{exp['val_score']:.6f}" if exp.get("val_score") is not None else "N/A"
            cv_score = f"{exp['cv_score']:.6f}" if exp.get("cv_score") is not None else "N/A"
            imp = "★" if exp.get("improved") else ""
            parts.append(f"| {exp['name']} | {test_score} | {val_score} | {cv_score} | {imp} |")
        parts.append("")

    # Top experiments hint
    if ctx.get("top_experiments"):
        parts.append("## Top Experiments (fetch code with get_experiment_code)\n")
        for name in ctx["top_experiments"]:
            parts.append(f"- {name}")
        parts.append("")
    if ctx.get("top_cv_experiments"):
        parts.append("## Top CV Experiments (OOF quality signal)\n")
        for name in ctx["top_cv_experiments"]:
            parts.append(f"- {name}")
        parts.append("")

    # Tool usage hint
    parts.append("## Available Tools\n")
    parts.append("- `get_experiment_summary(name)` — brief info about an experiment\n")
    parts.append("- `get_experiment_code(name)` — full source code\n")
    parts.append("- `get_experiment_report(name)` — training report with observations\n")
    parts.append("- `get_best_solution_code()` — code of the best solution\n")
    parts.append("- `get_reference_code(filename)` — reference solutions (winner code)\n")
    parts.append("- `list_analyst_reports()` — available data analysis reports\n")
    parts.append("- `get_analyst_report(name)` — specific analysis report\n")
    parts.append("")

    parts.append(f"## Your Task\nPropose {count} task(s). Return JSON array with {count} object(s).")
    return "\n".join(parts)


def call_openai_with_tools(system: str, user: str, temperature: float = 0.7, max_retries: int = 3) -> str:
    """Call OpenAI API with function calling support with retry logic."""
    import time
    
    url = f"{OPENAI_API_BASE}/chat/completions"
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        payload = {
            "model": "default",
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 8000,
            "temperature": temperature,
        }
        
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        
        # Retry logic for transient errors
        last_error = None
        for retry_attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=300, verify=False)
                response.raise_for_status()
                data = response.json()
                break  # Success, exit retry loop
            except requests.exceptions.Timeout as e:
                last_error = e
                if retry_attempt < max_retries - 1:
                    wait_time = 2 ** retry_attempt
                    print(f"  Timeout, retrying in {wait_time}s (attempt {retry_attempt + 1}/{max_retries})...", flush=True)
                    time.sleep(wait_time)
                    continue
                print(f"  ERROR: Timeout after {max_retries} retries", flush=True)
                raise
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 0
                # Retry on rate limit and server errors
                if status_code in (429, 502, 503, 504) and retry_attempt < max_retries - 1:
                    wait_time = 2 ** retry_attempt
                    print(f"  HTTP {status_code}, retrying in {wait_time}s (attempt {retry_attempt + 1}/{max_retries})...", flush=True)
                    time.sleep(wait_time)
                    continue
                print(f"  ERROR: HTTP {status_code} after {retry_attempt + 1} attempt(s)", flush=True)
                raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if retry_attempt < max_retries - 1:
                    wait_time = 2 ** retry_attempt
                    print(f"  Connection error, retrying in {wait_time}s (attempt {retry_attempt + 1}/{max_retries})...", flush=True)
                    time.sleep(wait_time)
                    continue
                print(f"  ERROR: Connection error after {max_retries} retries", flush=True)
                raise
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        # Check if model wants to call tools
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            # No more tool calls, return the final response
            content = message.get("content", "") or message.get("reasoning_content", "")
            return content
        
        # Process tool calls
        messages.append(message)
        
        for tool_call in tool_calls:
            tool_id = tool_call.get("id", "")
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            
            print(f"  Tool call: {tool_name}({arguments})", flush=True)
            
            # Dispatch tool call
            result = dispatch_tool_call(tool_name, arguments)
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False)[:10000],  # Limit response size
            })
    
    return ""


def parse_ideas(response_text: str) -> list[dict]:
    """Parse ideas from LLM response."""
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
        raw_sources = idea.get("stack_sources")
        if not isinstance(raw_sources, list):
            raw_sources = []
        validated.append({
            "name": str(idea.get("name", "unnamed")),
            "task_type": str(idea.get("task_type", "experiment")),
            "reasoning": str(idea.get("reasoning", "")),
            "base_experiment": str(idea.get("base_experiment", "default")),
            "prompt": str(idea.get("prompt", "")),
            "reference_code": idea.get("reference_code"),
            "stack_sources": [str(x) for x in raw_sources if str(x).strip()],
        })
    return validated


def generate_ideas(count: int = 1) -> list[dict]:
    """Generate experiment ideas via LLM with tool calling."""
    print(f"Generating {count} experiment idea(s) via LLM (with tools)...")
    ctx = build_compact_context()
    user_prompt = build_user_prompt(ctx, count)
    print(f"  Context: {len(ctx['experiments'])} experiments, prompt: {len(user_prompt)} chars")
    
    try:
        response_text = call_openai_with_tools(build_system_prompt(), user_prompt)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument("--count", type=int, default=3, help="Number of ideas to generate")
    args = parser.parse_args()

    cfg = load_config(args.project)
    configure(cfg)
    
    ideas = generate_ideas(args.count)
    print(json.dumps(ideas, indent=2, ensure_ascii=False))
