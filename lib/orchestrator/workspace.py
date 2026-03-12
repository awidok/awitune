"""Workspace and prompt preparation helpers for orchestration runs."""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .. import db


def resolve_base_solution(cfg, base_experiment: str) -> str:
    if not base_experiment or base_experiment == "default":
        return str(cfg.solutions_dir / "baseline")
    exp = db.get_experiment(base_experiment)
    if exp and exp.get("workspace_dir"):
        ws_path = Path(exp["workspace_dir"])
        if ws_path.is_dir() and (ws_path / "run.py").exists():
            return str(ws_path)
    return str(cfg.solutions_dir / "baseline")


def build_reference_code_section(cfg, reference_code: dict) -> str:
    if not reference_code:
        return ""
    ref_exp_name = reference_code.get("experiment", "")
    what_to_take = reference_code.get("what_to_take", "")
    if not ref_exp_name:
        return ""

    if cfg.reference_dir:
        ref_dir = cfg.reference_dir / ref_exp_name
        if ref_dir.is_dir():
            files = reference_code.get("files")
            if not files:
                files = [f.name for f in sorted(ref_dir.glob("*.py"))]
            section = f"\n\n## Reference Code: {ref_exp_name}\n"
            if what_to_take:
                section += f"**Specifically**: {what_to_take}\n"
            for fname in files:
                fpath = ref_dir / fname
                if fpath.exists():
                    code = fpath.read_text(errors="replace")
                    if len(code) > 20000:
                        code = code[:20000] + "\n# ... (truncated) ...\n"
                    section += f"\n### {fname}\n```python\n{code}\n```\n"
            return section

    ref_exp = db.get_experiment(ref_exp_name)
    if not ref_exp or not ref_exp.get("workspace_dir"):
        return ""
    ref_run_py = Path(ref_exp["workspace_dir"]) / "run.py"
    if not ref_run_py.exists():
        return ""
    ref_code = ref_run_py.read_text(errors="replace")
    ref_score = ref_exp.get("test_score", "?")
    section = f"\n\n## Reference Code (from experiment {ref_exp_name}, score {ref_score})\n"
    if what_to_take:
        section += f"**Specifically**: {what_to_take}\n"
    section += f"\n```python\n{ref_code}\n```\n"
    return section


def analyst_reports_dir(cfg):
    return cfg.data_dir / "analyst_reports"


def get_analyst_reports_summary(cfg) -> str:
    d = analyst_reports_dir(cfg)
    if not d.exists():
        return "No previous analysis has been done yet."
    reports = sorted(d.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not reports:
        return "No previous analysis has been done yet."
    parts = []
    for r in reports[:5]:
        content = r.read_text(errors="replace")
        if len(content) > 3000:
            content = content[:3000] + "\n... (truncated) ...\n"
        parts.append(f"### {r.stem}\n{content}\n")
    return "\n".join(parts)


def copy_analyst_reports_to_workspace(cfg, ws: Path):
    d = analyst_reports_dir(cfg)
    if not d.exists():
        return
    reports = sorted(d.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not reports:
        return
    dest = ws / "analyst_reports"
    dest.mkdir(exist_ok=True)
    for r in reports[:10]:
        shutil.copy2(r, dest / r.name)
    for j in d.glob("*.json"):
        shutil.copy2(j, dest / j.name)
    os.chmod(dest, 0o777)
    for f in dest.iterdir():
        os.chmod(f, 0o666)


def prepare_analyst_workspace(cfg, exp_dir, analysis_focus):
    ws = exp_dir / "workspace"
    if ws.exists():
        subprocess.run(["chmod", "-R", "u+rwX", str(ws)], capture_output=True, timeout=30)
    ws.mkdir(parents=True, exist_ok=True)
    os.chmod(ws, 0o777)
    copy_analyst_reports_to_workspace(cfg, ws)

    if cfg.analyst_prompt and cfg.analyst_prompt.exists():
        tpl = cfg.analyst_prompt.read_text()
        prompt = tpl.replace("{ANALYSIS_FOCUS}", analysis_focus or "General dataset exploration")
        prompt = prompt.replace("{PREVIOUS_ANALYSIS}", get_analyst_reports_summary(cfg))
    else:
        prompt = f"# Analysis Task\n\n{analysis_focus}\n"

    (exp_dir / "CLAUDE.md").write_text(prompt)
    od = exp_dir / "output"
    od.mkdir(exist_ok=True)
    os.chmod(od, 0o777)
    return ws


def prepare_workspace(cfg, base_path, exp_dir, custom_prompt, best_score, prev_exps, reference_code=None):
    ws = exp_dir / "workspace"
    if ws.exists():
        subprocess.run(["chmod", "-R", "u+rwX", str(ws)], capture_output=True, timeout=30)
    ws.mkdir(parents=True, exist_ok=True)
    base = Path(base_path)
    if base.is_dir():
        for f in base.iterdir():
            if f.is_file():
                shutil.copy2(f, ws / f.name)

    project_readme = cfg.project_dir / "README.md"
    if project_readme.exists():
        shutil.copy2(project_readme, ws / "README.md")

    os.chmod(ws, 0o777)
    for f in ws.iterdir():
        os.chmod(f, 0o666)

    exp_info = f"# Experiment: {exp_dir.name}\n\n"
    exp_info += f"**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    exp_info += f"**Base solution**: {base_path}\n"
    exp_info += f"**Best score at start**: {best_score:.6f}\n\n"
    if custom_prompt:
        exp_info += f"## Custom Prompt\n\n{custom_prompt}\n\n"
    (exp_dir / "EXPERIMENT_INFO.md").write_text(exp_info)

    prev_md = f"# Previous Experiments\n\nBest score: **{best_score:.6f}**\n\n"
    if prev_exps:
        prev_md += "| # | Name | Score | Notes |\n|---|------|-------|-------|\n"
        for i, e in enumerate(prev_exps[-20:]):
            score = e.get("test_score") or "?"
            prev_md += f"| {i + 1} | {e.get('name','')} | {score} | {e.get('notes','')} |\n"
    (ws / "prev_experiments.md").write_text(prev_md)

    copy_analyst_reports_to_workspace(cfg, ws)

    tpl = cfg.agent_prompt.read_text()
    prompt = tpl.replace("{BEST_SCORE}", f"{best_score:.6f}")
    if custom_prompt:
        prompt += f"\n\n## SPECIFIC TASK FOR THIS RUN\n{custom_prompt}\n"
    ref_section = build_reference_code_section(cfg, reference_code)
    if ref_section:
        prompt += ref_section
    if reference_code and reference_code.get("experiment") and cfg.reference_dir:
        ref_dir = cfg.reference_dir / reference_code["experiment"]
        if ref_dir.is_dir():
            ref_dest = ws / "reference"
            ref_dest.mkdir(exist_ok=True)
            for f in ref_dir.glob("*.py"):
                shutil.copy2(f, ref_dest / f.name)
            os.chmod(ref_dest, 0o777)
            for f in ref_dest.iterdir():
                os.chmod(f, 0o666)
    (exp_dir / "CLAUDE.md").write_text(prompt)

    od = exp_dir / "output"
    od.mkdir(exist_ok=True)
    os.chmod(od, 0o777)
    return ws


def _load_oof_registry(cfg) -> list[dict]:
    """Load OOF predictions registry."""
    fp = cfg.data_dir / "stacking_oof_registry.json"
    if not fp.exists():
        return []
    try:
        data = json.loads(fp.read_text(errors="replace"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    except Exception:
        return []
    return []


def prepare_stacking_workspace(cfg, exp_dir, custom_prompt, best_score, prev_exps, stack_sources):
    """Prepare workspace for stacking experiment.

    Stacking uses the SAME format as regular experiments:
    - Data is mounted at /app/data/ (read-only) — same as regular experiments
    - run.py uses --train /app/data/local_train etc. — same CLI interface
    - Output goes to /app/output/ — same output contract

    This enables stacking-on-stacking: a stacking experiment's output can be
    used as input to another stacking experiment.

    Data preparation:
    1. Load original data splits
    2. Join OOF predictions to train split (for meta-model training)
    3. Join val/test/submit predictions from base models
    4. Strip target columns from test split (prevent leakage)
    5. Save to exp_dir/stacking_data/ (mounted as /app/data/ in container)
    """
    import polars as pl

    ws = exp_dir / "workspace"
    if ws.exists():
        subprocess.run(["chmod", "-R", "u+rwX", str(ws)], capture_output=True, timeout=30)
    ws.mkdir(parents=True, exist_ok=True)

    # Copy stacking solution template
    stacking_template = cfg.solutions_dir / "stacking"
    if stacking_template.is_dir():
        for f in stacking_template.iterdir():
            if f.is_file():
                shutil.copy2(f, ws / f.name)
    else:
        raise RuntimeError(f"Stacking solution template not found: {stacking_template}")

    # Create stacking data directory (will be mounted as /app/data/ in container)
    stacking_data_dir = exp_dir / "stacking_data"
    stacking_data_dir.mkdir(exist_ok=True)

    # Load original data splits
    train_df = pl.read_parquet(str(cfg.data_dir / "local_train.parquet"))
    val_df = pl.read_parquet(str(cfg.data_dir / "local_val.parquet"))
    test_df = pl.read_parquet(str(cfg.data_dir / "local_test.parquet"))
    submit_df = pl.read_parquet(str(cfg.data_dir / "contest_test.parquet"))

    # Load OOF registry to find OOF predictions for train split
    oof_registry = _load_oof_registry(cfg)
    oof_by_experiment = {}
    for row in oof_registry:
        exp_name = row.get("experiment", "")
        oof_path = row.get("path", "")
        if exp_name and oof_path and Path(oof_path).exists():
            oof_by_experiment[exp_name] = oof_path

    # Track which columns came from which experiment
    column_sources = {}  # column_name -> experiment_name

    # Load and join predictions from base models
    valid_sources = []
    sources_with_oof = []
    sources_without_oof = []

    for source_name in stack_sources:
        source_exp = db.get_experiment(source_name)
        if not source_exp:
            continue
        # Skip analysis and oof_fold experiments — they don't produce predictions
        source_task_type = str(source_exp.get("task_type") or "").lower()
        if source_task_type in ("analysis", "oof_fold"):
            print(f"[stacking] Skipping source {source_name}: task_type={source_task_type}")
            continue

        # Find the output directory with predictions
        source_output = None
        if source_exp.get("exp_dir"):
            candidate = Path(source_exp["exp_dir"]) / "output"
            if candidate.exists():
                source_output = candidate
        elif source_exp.get("workspace_dir"):
            candidate = Path(source_exp["workspace_dir"]).parent / "output"
            if candidate.exists():
                source_output = candidate

        if not source_output:
            continue

        # Load predictions for val/test/submit splits
        try:
            val_pred = pl.read_parquet(str(source_output / "val_predictions.parquet"))
            test_pred = pl.read_parquet(str(source_output / "test_predictions.parquet"))
            submit_pred = pl.read_parquet(str(source_output / "submission.parquet"))
        except Exception:
            continue

        # Get prediction columns (predict_*)
        pred_cols = [c for c in val_pred.columns if c.startswith("predict_")]
        if not pred_cols:
            continue

        # Rename prediction columns to include source prefix
        rename_map = {}
        for col in pred_cols:
            new_col = f"predict_{source_name}_{col[len('predict_'):]}"
            rename_map[col] = new_col
            column_sources[new_col] = source_name

        val_pred = val_pred.rename(rename_map)
        test_pred = test_pred.rename(rename_map)
        submit_pred = submit_pred.rename(rename_map)

        # Join predictions with val/test/submit
        val_df = val_df.join(val_pred, on="customer_id", how="left")
        test_df = test_df.join(test_pred, on="customer_id", how="left")
        submit_df = submit_df.join(submit_pred, on="customer_id", how="left")

        # Load OOF predictions for train split (if available)
        oof_path = oof_by_experiment.get(source_name)
        if oof_path:
            try:
                oof_pred = pl.read_parquet(oof_path)
                oof_pred_cols = [c for c in oof_pred.columns if c.startswith("predict_")]
                if oof_pred_cols:
                    oof_pred = oof_pred.rename(rename_map)
                    train_df = train_df.join(oof_pred, on="customer_id", how="left")
                    # Check OOF coverage — warn if many NaN
                    renamed_cols = list(rename_map.values())
                    if renamed_cols:
                        null_count = train_df.select(renamed_cols[0]).null_count().item()
                        coverage_pct = 100.0 * (1 - null_count / len(train_df))
                        if null_count > 0:
                            print(
                                f"[stacking] WARNING: OOF for {source_name} has {null_count} NaN "
                                f"out of {len(train_df)} train rows ({coverage_pct:.1f}% coverage). "
                                f"NaN will be filled with 0.5."
                            )
                    sources_with_oof.append(source_name)
            except Exception:
                sources_without_oof.append(source_name)
        else:
            sources_without_oof.append(source_name)

        valid_sources.append(source_name)

    if len(valid_sources) < 2:
        raise ValueError(f"Need at least 2 valid prediction sources for stacking, got {len(valid_sources)} from {stack_sources}")

    # Fill NaN in prediction columns with 0.5 (neutral probability)
    # This handles cases where OOF predictions don't cover all training rows
    pred_columns = [c for c in train_df.columns if c.startswith("predict_")]
    if pred_columns:
        train_df = train_df.with_columns([
            pl.col(c).fill_null(0.5) for c in pred_columns
        ])
    pred_columns_val = [c for c in val_df.columns if c.startswith("predict_")]
    if pred_columns_val:
        val_df = val_df.with_columns([
            pl.col(c).fill_null(0.5) for c in pred_columns_val
        ])
    pred_columns_test = [c for c in test_df.columns if c.startswith("predict_")]
    if pred_columns_test:
        test_df = test_df.with_columns([
            pl.col(c).fill_null(0.5) for c in pred_columns_test
        ])
    pred_columns_submit = [c for c in submit_df.columns if c.startswith("predict_")]
    if pred_columns_submit:
        submit_df = submit_df.with_columns([
            pl.col(c).fill_null(0.5) for c in pred_columns_submit
        ])

    # Strip target columns from test split to prevent data leakage
    target_cols = [c for c in test_df.columns if c.startswith("target_")]
    if target_cols:
        test_df = test_df.drop(target_cols)

    # Save joined data splits (same filenames as regular experiments)
    train_df.write_parquet(str(stacking_data_dir / "local_train.parquet"))
    val_df.write_parquet(str(stacking_data_dir / "local_val.parquet"))
    test_df.write_parquet(str(stacking_data_dir / "local_test.parquet"))
    submit_df.write_parquet(str(stacking_data_dir / "contest_test.parquet"))

    # Copy any metadata files from original data dir
    for meta_file in cfg.data_dir.glob("*.json"):
        shutil.copy2(meta_file, stacking_data_dir / meta_file.name)

    # Set permissions on stacking data directory
    os.chmod(stacking_data_dir, 0o777)
    for f in stacking_data_dir.iterdir():
        os.chmod(f, 0o666)

    # Write column sources mapping to workspace
    column_sources_file = ws / "column_sources.json"
    column_sources_file.write_text(json.dumps(column_sources, indent=2))

    project_readme = cfg.project_dir / "README.md"
    if project_readme.exists():
        shutil.copy2(project_readme, ws / "README.md")

    os.chmod(ws, 0o777)
    for f in ws.iterdir():
        if f.is_dir():
            os.chmod(f, 0o777)
            for subf in f.iterdir():
                os.chmod(subf, 0o666)
        else:
            os.chmod(f, 0o666)

    # Check OOF coverage per source for experiment info
    oof_coverage_info = {}
    for src in sources_with_oof:
        src_pred_cols = [c for c in train_df.columns if c.startswith(f"predict_{src}_")]
        if src_pred_cols:
            # After fill_null, check how many were originally null (already filled)
            # We can't check anymore since we filled, so just note it was joined
            oof_coverage_info[src] = "full coverage (NaN filled with 0.5 if any gaps)"

    exp_info = f"# Stacking Experiment: {exp_dir.name}\n\n"
    exp_info += f"**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    exp_info += f"**Best score at start**: {best_score:.6f}\n"
    exp_info += f"**Base models**: {len(valid_sources)}\n"
    exp_info += f"**Sources with OOF (train predictions)**: {len(sources_with_oof)}\n"
    exp_info += f"**Sources without OOF**: {len(sources_without_oof)}\n"
    exp_info += f"**NaN handling**: prediction NaN values filled with 0.5 (neutral probability)\n\n"
    for i, src in enumerate(valid_sources):
        oof_status = "✓ OOF" if src in sources_with_oof else "✗ no OOF"
        exp_info += f"{i+1}. {src} ({oof_status})\n"
    if custom_prompt:
        exp_info += f"\n## Custom Prompt\n\n{custom_prompt}\n\n"
    (exp_dir / "EXPERIMENT_INFO.md").write_text(exp_info)

    prev_md = f"# Previous Experiments\n\nBest score: **{best_score:.6f}**\n\n"
    if prev_exps:
        prev_md += "| # | Name | Score | Notes |\n|---|------|-------|-------|\n"
        for i, e in enumerate(prev_exps[-20:]):
            score = e.get("test_score") or "?"
            prev_md += f"| {i + 1} | {e.get('name','')} | {score} | {e.get('notes','')} |\n"
    (ws / "prev_experiments.md").write_text(prev_md)

    # Build stacking-specific prompt
    prompt = _build_stacking_agent_prompt(
        cfg, best_score, valid_sources, column_sources, custom_prompt,
        sources_with_oof=sources_with_oof, sources_without_oof=sources_without_oof,
    )
    (exp_dir / "CLAUDE.md").write_text(prompt)

    od = exp_dir / "output"
    od.mkdir(exist_ok=True)
    os.chmod(od, 0o777)
    return ws


def _build_stacking_agent_prompt(cfg, best_score, valid_sources, column_sources, custom_prompt,
                                  sources_with_oof=None, sources_without_oof=None):
    """Build the agent prompt for stacking experiments."""
    sources_with_oof = sources_with_oof or []
    sources_without_oof = sources_without_oof or []

    # Group columns by source experiment
    source_columns = {}
    for col, src in column_sources.items():
        if src not in source_columns:
            source_columns[src] = []
        source_columns[src].append(col)

    source_table = ""
    for src in valid_sources:
        cols = source_columns.get(src, [])
        cols_str = ", ".join(sorted(cols)[:5])
        if len(cols) > 5:
            cols_str += f", ... ({len(cols)} total)"
        oof_marker = "✓" if src in sources_with_oof else "✗"
        source_table += f"| {src} | {oof_marker} | {cols_str} |\n"

    oof_note = ""
    if sources_without_oof:
        oof_note = (
            f"\n**Note**: {len(sources_without_oof)} source(s) have NO OOF predictions for the train split. "
            f"The train split will have NaN values for their prediction columns. "
            f"Handle NaN appropriately (e.g., fill with 0.5 or column mean).\n"
        )

    prompt = f"""# Stacking Meta-Model Executor

You are executing a stacking/blending experiment combining {len(valid_sources)} base models.
The orchestrator has decided what to stack — your job is to implement the meta-model, run it, and report results.

**Current best score**: {best_score:.6f} (Macro ROC-AUC)

## Data
`/app/data/` (READ-ONLY) — same location and format as regular experiments.
Each split already contains base model predictions as extra columns:
- `local_train.parquet` — train split with targets + OOF prediction columns (fit meta-model here)
- `local_val.parquet` — val split with targets + prediction columns (validate here)
- `local_test.parquet` — test split WITHOUT targets + prediction columns (predict here)
- `contest_test.parquet` — submit split, no targets (predict here)

**IMPORTANT**: Train the meta-model on TRAIN data, validate on VAL. Do NOT train on val.
{oof_note}
## Prediction Columns (pre-joined)
| Source Experiment | OOF | Columns |
|-------------------|-----|---------|
{source_table}
Column format: `predict_<experiment_name>_<target_id>` (e.g., `predict_exp_123_1_1` → target_1_1 from exp_123)
OOF column: ✓ = train split has OOF predictions, ✗ = train split has NaN for this source

## Workspace
- `run.py` — stacking template (modify this)
- `column_sources.json` — column → experiment mapping
- `prev_experiments.md` — experiment history

## Execution Steps

### Step 1: Read Context
1. Read `run.py` — understand the stacking template
2. Read `prev_experiments.md` — understand what was tried and what scores were achieved
3. Read `column_sources.json` — understand which prediction columns come from which experiment

### Step 2: Implement the Task
The orchestrator's task is in the **SPECIFIC TASK FOR THIS RUN** section below.
If no specific task: improve the default LogisticRegression stacker with better techniques.
- **ALL CODE MUST BE IN `run.py`** — no separate files
- Modify the existing `run.py`, don't rewrite from scratch
- `run.py` must accept `--train`, `--val`, `--test`, `--submit` prefix args and output dir positional arg
- `load_split(prefix)` loads `{{prefix}}.parquet`
- Do NOT hardcode data paths

### Step 3: Run Training (SINGLE RUN ONLY)
```bash
python run.py \\
  --train /app/data/local_train \\
  --val /app/data/local_val \\
  --test /app/data/local_test \\
  --submit /app/data/contest_test \\
  /app/output
```

**CRITICAL: You get exactly ONE successful training run. This is non-negotiable.**
- Run `python run.py ...` once.
- If it crashes → fix the bug → retry. You may retry up to 3 times total.
- The FIRST run that completes successfully (produces output files) is FINAL.
- After a successful run: verify outputs → write report → STOP IMMEDIATELY.
- **FORBIDDEN after successful training:**
  - Re-running with "improved" parameters
  - Running "one more time" to see if results improve
  - Running ablation studies or comparisons
  - Re-running because you think you can do better
  - Any additional `python run.py` execution
- If you violate this rule, the experiment is considered failed.
- Never leave the experiment without output files.

### Step 4: Verify Outputs
```bash
ls -la /app/output/val_predictions.parquet /app/output/test_predictions.parquet /app/output/submission.parquet /app/output/metrics.json /app/output/training_logs.json
```
ALL files must exist. If any is missing, the experiment is wasted.

### Step 5: Write Report
Write `/app/output/report.md` manually (NOT generated by run.py). Include:
1. **What was changed** — list specific code modifications to run.py
2. **Per-target AUC** — table for all targets with AUC
3. **Which sources contributed most** — feature importance or weight analysis
4. **Observations** — what worked, what didn't, anomalies noticed

### Step 6: Stop
After writing the report — **stop**. Do not run additional experiments, ablations, or "quick tests".

## Output Contract
**ALL prediction files MUST contain predictions for ALL targets present in the data.**
Use the exact same prediction-column schema as the regular experiment format.
Do not invent alternative column names, nested formats, or partial outputs.

| File | Required | Content |
|------|----------|---------|
| `val_predictions.parquet` | **YES** | `customer_id` + ALL `predict_*` columns for local_val |
| `test_predictions.parquet` | **YES** | `customer_id` + ALL `predict_*` columns for local_test |
| `submission.parquet` | **YES** | `customer_id` + ALL `predict_*` columns for contest_test |
| `metrics.json` | **YES** | `{{"val_macro_roc_auc": <float>}}` at minimum |
| `training_logs.json` | **YES** | structured training logs |
| `tb_logs/` | optional | TensorBoard logs (nice to have, not required) |

## REQUIRED OUTPUT FORMAT DETAILS
- `val_predictions.parquet`, `test_predictions.parquet`, `submission.parquet` MUST be flat parquet tables.
- First column: `customer_id`.
- Remaining columns: prediction columns named exactly `predict_<target_id>` for every target.
- Example columns: `predict_1_1`, `predict_1_2`, ..., `predict_10_1`.
- Values in every `predict_*` column must be probabilities in `[0, 1]`.
- Do NOT output target columns, long format tables, JSON blobs, arrays-in-cells, or renamed prediction columns.
- Even if you only change part of the stacker, still output the FULL prediction table for ALL targets.
- `metrics.json` must be valid JSON object, not markdown, and must contain numeric key `val_macro_roc_auc`.
- `training_logs.json` must be valid JSON object, not markdown or plain text.

Return ONLY files in this exact format. Format violations make the experiment unusable for OOF/stacking.

## Hard Rules

### Single-Run Policy (STRICTLY ENFORCED)
**You get exactly ONE successful training run. This is non-negotiable.**

- Run `python run.py ...` once.
- If it crashes → fix the bug → retry. You may retry up to 3 times total.
- The FIRST run that completes successfully (produces output files) is FINAL.
- After a successful run: verify outputs → write report → STOP IMMEDIATELY.
- **FORBIDDEN after successful training:**
  - Re-running with "improved" parameters
  - Running "one more time" to see if results improve
  - Running ablation studies or comparisons
  - Re-running because you think you can do better
  - Any additional `python run.py` execution
- If you violate this rule, the experiment is considered failed.

### Code Rules
- **ALL CODE MUST BE IN `run.py`** — no separate files, no helper modules
- Modify the existing `run.py`, don't rewrite from scratch
- `run.py` must accept `--train`, `--val`, `--test`, `--submit` prefix args
- Do NOT hardcode data paths
- Do NOT retrain base models — use pre-computed predictions only
- Train meta-model on TRAIN, validate on VAL — never train on val
- **NO cross-validation inside run.py** — train on train, validate on val, that's it

"""

    if custom_prompt:
        prompt += f"\n## SPECIFIC TASK FOR THIS RUN\n{custom_prompt}\n"

    return prompt
