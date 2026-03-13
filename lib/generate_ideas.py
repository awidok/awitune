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
You are the strategic brain of an automated ML experiment system. You analyze experiment \
history, identify what works, and decide the highest-ROI next experiments.

The agents executing your tasks are skilled coders but NOT strategists — they implement \
exactly what you tell them. Your prompts must be **specific and detailed**: include exact \
hyperparameter values, code patterns, architecture descriptions, and expected outcomes.

## Your Decision Framework

### Step 1: Diagnose the current state
Use TOOLS to answer these questions before proposing anything:
- What is the current best score? How far from the theoretical ceiling?
- What was tried in the last 5-10 experiments? What improved vs. what didn't?
- Are there analyst reports with unused insights?
- Which solution families exist? (DCNv2, tree, transformer, MLP, etc.)
- What's the score gap between the best and 2nd-best solution family?
- Are there subsets, groups, or failure regions where different experiments win for different reasons?
- Which completed models are actually diverse enough to stack? Use `get_diversity_candidates()` instead of picking top-N by score only.

### Step 2: Identify the highest-ROI action
Apply this priority ladder (top = highest priority):

1. **Fix broken patterns** — if recent experiments keep failing or regressing, diagnose why \
   before launching more. Propose an analysis task to investigate.
2. **Exploit proven wins** — if experiment X improved score, apply the same technique to \
   other solution families, or combine it with other proven wins.
3. **Strengthen weak families** — if one solution family (e.g., tree-based) is far behind, \
   a targeted improvement there adds more stacking value than +0.001 on the best family.
4. **Explore new territory** — propose a fundamentally different approach only when existing \
   families are well-optimized.
5. **Micro-tune the best** — small hyperparameter tweaks on the best solution (low risk, low reward).

### Step 3: Write detailed task prompts
Your `prompt` field is the ONLY instruction the agent sees. It must contain:
- **What to change**: Exact code modifications, not vague directions
- **Why it should work**: Reference to evidence (experiment scores, analyst findings)
- **How to implement**: Specific hyperparameters, architecture details, code patterns
- **What to watch for**: Expected training behavior, potential failure modes

BAD prompt: "Try a transformer architecture"
GOOD prompt: "Replace the DCNv2 backbone with a 4-layer FT-Transformer: embed categoricals \
with dim=32, use 8 attention heads, FFN ratio=4/3, dropout=0.1. Use the same PLE encoding \
for numerics. Keep focal loss and EMA. Expected: slower training (~4h) but potentially \
better feature interactions. Watch for: attention collapse (all heads attending to same \
features) — if val AUC doesn't improve after epoch 3, reduce LR to 5e-4."

## Task Types

### analysis
Data exploration. Use when:
- No analyst reports exist yet (ALWAYS start with analysis)
- Score plateaued for 5+ experiments
- Recent experiments show unexpected patterns (e.g., val AUC drops on certain target groups)
- You need error analysis on specific targets

Your analysis prompt should specify EXACTLY what to investigate and what output format you expect.

### experiment
Model training. Use when you have a clear, evidence-based hypothesis.
The agent will modify `run.py` and run training. Be specific about:
- Which solution family / base experiment to fork from
- Exact changes to make (architecture, loss, features, training recipe)
- Expected outcome and how to validate

## Solution Portfolio Strategy
Build diverse solutions for stacking. Track families:
- **Neural**: DCNv2, Transformer, MLP, TabNet
- **Tree**: LightGBM, XGBoost, CatBoost
- **Linear**: Ridge, LogisticRegression with engineered features

Each family should have its own improvement trajectory. Don't just optimize the best one.

## Portfolio Strategy Beyond Global Score
The orchestrator must reason at two levels:
1. **Global score** — overall leaderboard progress.
2. **Subpopulation / specialist opportunities** — where different experiments may win on different slices, target groups, or failure modes.

Do NOT hardcode any project-specific target names or assumptions. Instead:
- infer weak regions from experiment metrics, reports, and recent failures,
- look for evidence that one experiment dominates a subset while another dominates elsewhere,
- use `get_diversity_candidates()` to choose stacking sources from different families and avoid redundant ensembles of near-identical models.

## Rules
- NEVER propose something already running or queued (check the "In Progress" section)
- NEVER repeat a failed approach without a clear reason why it will work this time
- USE TOOLS to fetch experiment details, code, reports before deciding
- Include a MIX: some high-risk/high-reward + some safe incremental improvements
- NO STACKING/BLENDING in experiment tasks (handled separately)
- Keep experiment names short and descriptive (max 30 chars, snake_case)
- NEVER propose tasks to compute OOF (out-of-fold) predictions — OOF generation is handled \
  AUTOMATICALLY by the orchestrator when stacking tasks need them. Your job is to propose \
  model training experiments and analysis tasks, NOT OOF computation. If you want OOF \
  predictions for stacking, just propose a stacking task with stack_sources — the system \
  will automatically generate OOF folds for each source experiment.
- NEVER instruct agents to do cross-validation inside run.py — experiments train on train, \
  validate on val, that's it. Internal CV wastes hours of GPU time. The orchestrator handles \
  CV/OOF automatically when needed for stacking.

## Output Format
JSON array of task objects:
```
[
  {
    "name": "snake_case_name",
    "task_type": "experiment" | "analysis",
    "reasoning": "1-3 sentences: evidence → hypothesis → expected outcome",
    "base_experiment": "experiment_name or 'default'",
    "prompt": "Detailed, specific instructions for the agent executor",
    "reference_code": null | {"experiment": "name", "what_to_take": "description"},
    "stack_sources": []
  }
]
```
`stack_sources` is optional for non-stacking tasks, required for `task_type="stacking"`.

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

    # Enable stacking mode in the prompt
    prompt = prompt.replace(
        '- NO STACKING/BLENDING in experiment tasks (handled separately)',
        (
            "- STACKING MODE ENABLED: you may propose stacking/blending tasks.\n"
            "- Keep base-model work dominant: at least half of proposed tasks should be non-stacking.\n"
            "- If proposing stacking, reuse existing OOF predictions from prior experiments.\n"
            "- Use `get_oof_registry()` tool to check which experiments have OOF predictions available.\n"
            "- **CRITICAL**: Any task involving ensembling, blending, stacking, weighted averaging, \n"
            "  or meta-models MUST use task_type=\"stacking\". Using task_type=\"experiment\" for such \n"
            "  tasks will cause them to run WITHOUT stacking data, wasting GPU time."
        ),
    )
    prompt = prompt.replace(
        '"task_type": "experiment" | "analysis",',
        '"task_type": "experiment" | "analysis" | "stacking",',
    )
    prompt += (
        "\n\n## IMPORTANT: task_type Rules for Stacking\n"
        "- Any task that combines predictions from multiple models (ensemble, blend, stack, "
        "weighted average, meta-model, rank averaging) MUST have task_type=\"stacking\".\n"
        "- task_type=\"stacking\" triggers special workspace setup: OOF predictions from "
        "stack_sources are joined and mounted as training data.\n"
        "- task_type=\"experiment\" does NOT have access to OOF data — it only gets the base "
        "solution code and raw features.\n"
        "- If task_type is \"stacking\", you MUST include stack_sources with 2-5 experiment names "
        "that should be stacked together. Pick diverse families for best stacking gains.\n"
        "- If you are unsure whether a task is stacking, ask yourself: \"Does this task need "
        "predictions from other models as input?\" If yes → stacking. If no → experiment.\n"
        "\n## IMPORTANT: OOF Predictions Are Automatic\n"
        "- NEVER propose tasks whose goal is to compute OOF (out-of-fold) predictions.\n"
        "- OOF prediction generation is handled AUTOMATICALLY by the orchestrator.\n"
        "- When you propose a stacking task with stack_sources, the system automatically "
        "generates OOF folds for each source experiment that doesn't have them yet.\n"
        "- Your job: propose model training experiments (task_type=\"experiment\") to build "
        "diverse base models, and stacking tasks (task_type=\"stacking\") to combine them.\n"
        "- The OOF pipeline is: you propose experiment → it trains → you propose stacking "
        "with that experiment in stack_sources → orchestrator auto-generates OOF → stacking runs.\n"
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
    parts.append("- `get_experiment_metrics(name)` — detailed per-target metrics from eval/metrics JSON\n")
    parts.append("- `get_best_solution_code()` — code of the best solution\n")
    parts.append("- `get_reference_code(filename)` — reference solutions (winner code)\n")
    parts.append("- `list_reference_files()` — list available reference files before fetching\n")
    parts.append("- `search_experiments(status, task_type, name_contains, min_score)` — filter experiments\n")
    parts.append("- `get_diff_between_experiments(a, b)` — code diff between two experiments\n")
    parts.append("- `list_analyst_reports()` — available data analysis reports\n")
    parts.append("- `get_analyst_report(name)` — specific analysis report\n")
    parts.append("- `get_training_logs(name)` — epoch-by-epoch training metrics\n")
    parts.append("- `get_oof_registry()` — OOF predictions registry for stacking planning\n")
    parts.append("- `get_targetwise_portfolio()` — hard targets, per-target winners, specialist opportunities\n")
    parts.append("- `get_diversity_candidates(limit)` — diversity-aware stacking candidates grouped by family\n")
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
        stack_sources = [str(x) for x in raw_sources if str(x).strip()]
        task_type = str(idea.get("task_type", "experiment"))
        idea_name = str(idea.get("name", "unnamed"))
        idea_prompt = str(idea.get("prompt", ""))

        # Auto-correct task_type when stacking indicators are present
        if task_type != "stacking" and _cfg and _cfg.enable_stacking_mode:
            if stack_sources:
                task_type = "stacking"
            else:
                _lower = (idea_name + " " + idea_prompt).lower()
                _stacking_keywords = ("stacking", "ensemble", "blending", "blend", "meta_model", "meta-model")
                if any(kw in _lower for kw in _stacking_keywords):
                    task_type = "stacking"

        validated.append({
            "name": idea_name,
            "task_type": task_type,
            "reasoning": str(idea.get("reasoning", "")),
            "base_experiment": str(idea.get("base_experiment", "default")),
            "prompt": idea_prompt,
            "reference_code": idea.get("reference_code"),
            "stack_sources": stack_sources,
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


SMART_LAUNCH_SYSTEM_PROMPT = """\
You convert a user's free-text experiment instruction into a structured task definition \
for an ML agent executor.

You have TOOLS to look up existing experiments, their code, scores, and analyst reports. \
Use them to pick the right base_experiment and write a detailed prompt.

The agent is an executor, not a strategist. Your `prompt` must be **specific and detailed**: \
include exact hyperparameter values, code patterns, architecture descriptions. \
The agent will implement exactly what you describe.

Rules:
- stacking/blending/ensembling/weighted averaging/meta-model/rank averaging → task_type="stacking", fill stack_sources
- data analysis → task_type="analysis"
- everything else → task_type="experiment", pick best base_experiment
- NEVER propose tasks to compute OOF predictions — OOF generation is automatic when stacking needs it
- Expand brief instructions into specific technical guidance with code hints
- Use tools to find the right experiment names and understand current code

CRITICAL task_type rules:
- task_type="stacking" triggers special workspace: OOF predictions from stack_sources are \
  joined and mounted as training data. Without this, the agent has NO access to other models' predictions.
- task_type="experiment" only gets base solution code and raw features — NO OOF data.
- If the task needs predictions from other models as input → MUST be task_type="stacking".
- If the user mentions ensemble, blend, stack, weighted average, meta-model → task_type="stacking".

Return EXACTLY ONE JSON object:
{
  "name": "snake_case_name (max 30 chars)",
  "task_type": "experiment" | "analysis" | "stacking",
  "reasoning": "why this makes sense",
  "base_experiment": "experiment_name or 'default'",
  "prompt": "Detailed, specific instructions for the agent executor",
  "reference_code": null | {"experiment": "name", "what_to_take": "description"},
  "stack_sources": ["exp_A", "exp_B"]
}

Return ONLY valid JSON — no markdown fences, no commentary.
"""


def generate_smart_idea(user_instruction: str) -> dict | None:
    """Generate a single experiment idea from a free-text user instruction.

    Uses the existing LLM infrastructure (tools, context) but with a
    specialised system prompt that converts the user's brief instruction
    into a fully structured experiment definition.
    """
    if not _cfg:
        raise RuntimeError("generate_ideas module not configured — call configure(cfg) first")

    print(f"[smart_launch] Generating idea from: {user_instruction!r}")
    ctx = build_compact_context()
    user_prompt = build_user_prompt(ctx, 1)
    user_prompt += f"\n\n## USER INSTRUCTION (this is what you must convert into an experiment)\n{user_instruction}\n"

    try:
        response_text = call_openai_with_tools(SMART_LAUNCH_SYSTEM_PROMPT, user_prompt)
        print(f"[smart_launch] LLM response: {len(response_text)} chars")
        if not response_text:
            print("[smart_launch] WARNING: Empty response from LLM")
            return None

        # parse_ideas expects an array; smart launch returns a single object
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try parsing as single object first, then as array
        bracket_start = text.find("{")
        bracket_end = text.rfind("}")
        if bracket_start != -1 and bracket_end != -1:
            candidate = text[bracket_start:bracket_end + 1]
            try:
                idea = json.loads(candidate)
                if isinstance(idea, dict):
                    raw_sources = idea.get("stack_sources")
                    if not isinstance(raw_sources, list):
                        raw_sources = []
                    stack_sources = [str(x) for x in raw_sources if str(x).strip()]
                    task_type = str(idea.get("task_type", "experiment"))
                    idea_name = str(idea.get("name", "smart_launch"))
                    idea_prompt = str(idea.get("prompt", user_instruction))

                    # Auto-correct task_type when stacking indicators are present
                    if task_type != "stacking" and _cfg and _cfg.enable_stacking_mode:
                        if stack_sources:
                            task_type = "stacking"
                        else:
                            _lower = (idea_name + " " + idea_prompt).lower()
                            _stacking_keywords = ("stacking", "ensemble", "blending", "blend", "meta_model", "meta-model")
                            if any(kw in _lower for kw in _stacking_keywords):
                                task_type = "stacking"

                    return {
                        "name": idea_name,
                        "task_type": task_type,
                        "reasoning": str(idea.get("reasoning", "")),
                        "base_experiment": str(idea.get("base_experiment", "default")),
                        "prompt": idea_prompt,
                        "reference_code": idea.get("reference_code"),
                        "stack_sources": stack_sources,
                    }
            except json.JSONDecodeError:
                pass

        # Fallback: try parse_ideas (handles arrays)
        ideas = parse_ideas(response_text)
        if ideas:
            return ideas[0]
        return None
    except Exception as e:
        print(f"[smart_launch] ERROR: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Path to project directory")
    parser.add_argument("--count", type=int, default=3, help="Number of ideas to generate")
    args = parser.parse_args()

    cfg = load_config(args.project)
    configure(cfg)
    
    ideas = generate_ideas(args.count)
    print(json.dumps(ideas, indent=2, ensure_ascii=False))
