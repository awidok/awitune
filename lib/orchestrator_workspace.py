"""Workspace and prompt preparation helpers for orchestration runs."""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from . import db


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
