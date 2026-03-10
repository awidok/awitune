"""
Tools for the orchestrator agent to fetch detailed information on-demand.

Instead of sending all context upfront, the agent can call these tools
to get specific details when needed.
"""

import json
from pathlib import Path
from typing import Optional

try:
    from .. import db
    from ..config import ProjectConfig
except ImportError:
    # Allow running as script
    import db
    from config import ProjectConfig

_cfg: Optional[ProjectConfig] = None


def configure(cfg: ProjectConfig):
    """Configure tools with project config."""
    global _cfg
    _cfg = cfg


# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_experiment_summary",
            "description": "Get a brief summary of an experiment (score, status, approach). Use this first to understand what experiments exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the experiment (e.g., 'auto_mixture_of_experts_mlp_20260307_130557')"
                    }
                },
                "required": ["experiment_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_code",
            "description": "Get the full source code of an experiment's run.py. Use when you need to understand implementation details or copy code patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the experiment"
                    }
                },
                "required": ["experiment_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_report",
            "description": "Get the detailed report from an experiment (training logs, per-target AUC, observations). Use to understand what worked and what didn't.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the experiment"
                    }
                },
                "required": ["experiment_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_analyst_report",
            "description": "Get a data analysis report (feature importance, error patterns, correlations). Use to get insights about the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_name": {
                        "type": "string",
                        "description": "Name of the analyst report (e.g., 'auto_feature_importance_analysis_20260307_035459')"
                    }
                },
                "required": ["report_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_best_solution_code",
            "description": "Get the source code of the best performing solution. Use as reference for improvements.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_reference_code",
            "description": "Get code from reference solutions (winner solutions, baseline improvements). Use to incorporate proven techniques.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename to read (e.g., '01_feature_engineering.py', '02_train_nn.py')"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_diff_between_experiments",
            "description": "Get the diff between two experiments' code. Use to understand what changed between approaches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_a": {
                        "type": "string",
                        "description": "First experiment name"
                    },
                    "experiment_b": {
                        "type": "string",
                        "description": "Second experiment name"
                    }
                },
                "required": ["experiment_a", "experiment_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_analyst_reports",
            "description": "List all available analyst reports with brief descriptions. Use to find relevant data analysis.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_training_logs",
            "description": "Get detailed training logs from an experiment (epoch-by-epoch metrics). Use to analyze training dynamics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the experiment"
                    }
                },
                "required": ["experiment_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_reference_files",
            "description": "List available reference solution files (winner code, baselines). Use before get_reference_code to know what files exist.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_experiments",
            "description": "Search and filter experiments by status, task_type, score range, or name pattern. Returns a compact list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status: completed, running, failed, queued (optional)"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Filter by task type: experiment, analysis, stacking, oof_fold (optional)"
                    },
                    "name_contains": {
                        "type": "string",
                        "description": "Filter by name substring (optional)"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum test_score (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_metrics",
            "description": "Get detailed evaluation metrics (per-target AUC, confusion matrices, etc.) from eval_results.json or metrics.json.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the experiment"
                    }
                },
                "required": ["experiment_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_oof_registry",
            "description": "Get the OOF predictions registry — which experiments have OOF files, their scores, and families. Essential for planning stacking experiments.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def get_experiment_summary(experiment_name: str) -> dict:
    """Get a brief summary of an experiment."""
    exp = db.get_experiment(experiment_name)
    if not exp:
        return {"error": f"Experiment '{experiment_name}' not found"}
    
    return {
        "name": exp.get("name"),
        "status": exp.get("status"),
        "test_score": exp.get("test_score"),
        "val_score": exp.get("val_score"),
        "improved": exp.get("improved", False),
        "prompt": (exp.get("prompt") or "")[:500],
        "notes": exp.get("notes"),
        "elapsed_min": exp.get("elapsed_min"),
        "created_at": exp.get("created_at"),
    }


def get_experiment_code(experiment_name: str) -> dict:
    """Get the full source code of an experiment's run.py."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    
    exp_dir = _cfg.experiments_dir / experiment_name
    run_py = exp_dir / "workspace" / "run.py"
    
    if not run_py.exists():
        return {"error": f"run.py not found for experiment '{experiment_name}'"}
    
    code = run_py.read_text(errors="replace")
    return {
        "experiment_name": experiment_name,
        "code": code,
        "lines": len(code.splitlines()),
    }


def get_experiment_report(experiment_name: str) -> dict:
    """Get the detailed report from an experiment."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    
    exp_dir = _cfg.experiments_dir / experiment_name
    report_path = exp_dir / "output" / "report.md"
    
    if not report_path.exists():
        return {"error": f"Report not found for experiment '{experiment_name}'"}
    
    report = report_path.read_text(errors="replace")
    return {
        "experiment_name": experiment_name,
        "report": report,
    }


def get_analyst_report(report_name: str) -> dict:
    """Get a data analysis report."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    
    # Try both .md and .json
    reports_dir = _cfg.data_dir / "analyst_reports"
    
    for ext in [".md", ".json"]:
        path = reports_dir / f"{report_name}{ext}"
        if path.exists():
            content = path.read_text(errors="replace")
            return {
                "report_name": report_name,
                "format": ext[1:],
                "content": content,
            }
    
    return {"error": f"Analyst report '{report_name}' not found"}


def get_best_solution_code() -> dict:
    """Get the source code of the best performing solution."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    
    direction = _cfg.best_score_sort_key()
    best_exp = db.get_best_experiment(direction)
    
    if not best_exp:
        # Fall back to baseline
        baseline_path = _cfg.solutions_dir / "baseline" / "run.py"
        if baseline_path.exists():
            return {
                "source": "baseline",
                "code": baseline_path.read_text(errors="replace"),
            }
        return {"error": "No best solution found"}
    
    return get_experiment_code(best_exp["name"])


def get_reference_code(filename: str) -> dict:
    """Get code from reference solutions."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    
    if _cfg.reference_dir is None:
        return {"error": "No reference directory configured"}
    
    # Try winner_1st_place first
    ref_path = _cfg.reference_dir / "winner_1st_place" / filename
    if not ref_path.exists():
        ref_path = _cfg.reference_dir / filename
    
    if not ref_path.exists():
        return {"error": f"Reference file '{filename}' not found"}
    
    return {
        "filename": filename,
        "code": ref_path.read_text(errors="replace"),
    }


def get_diff_between_experiments(experiment_a: str, experiment_b: str) -> dict:
    """Get the diff between two experiments' code."""
    import difflib
    
    code_a = get_experiment_code(experiment_a)
    code_b = get_experiment_code(experiment_b)
    
    if "error" in code_a:
        return {"error": code_a["error"]}
    if "error" in code_b:
        return {"error": code_b["error"]}
    
    lines_a = code_a["code"].splitlines(keepends=True)
    lines_b = code_b["code"].splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(
        lines_a, lines_b,
        fromfile=f"{experiment_a}/run.py",
        tofile=f"{experiment_b}/run.py",
        n=3
    ))
    
    diff_text = "".join(diff)
    if len(diff_text) > 10000:
        diff_text = diff_text[:10000] + "\n... (diff truncated) ...\n"
    
    return {
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "diff": diff_text,
    }


def list_analyst_reports() -> dict:
    """List all available analyst reports."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    
    reports_dir = _cfg.data_dir / "analyst_reports"
    if not reports_dir.exists():
        return {"reports": []}
    
    reports = []
    for f in sorted(reports_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
        # Read first few lines for description
        content = f.read_text(errors="replace")
        first_lines = "\n".join(content.split("\n")[:5])
        reports.append({
            "name": f.stem,
            "file": str(f.name),
            "preview": first_lines[:300],
        })
    
    return {"reports": reports[:20]}  # Limit to 20 most recent


def get_training_logs(experiment_name: str) -> dict:
    """Get detailed training logs from an experiment."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    
    exp_dir = _cfg.experiments_dir / experiment_name
    logs_path = exp_dir / "output" / "training_logs.json"
    
    if not logs_path.exists():
        return {"error": f"Training logs not found for experiment '{experiment_name}'"}
    
    try:
        logs = json.loads(logs_path.read_text())
        return {
            "experiment_name": experiment_name,
            "logs": logs,
        }
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse training logs: {e}"}


def list_reference_files() -> dict:
    """List available reference solution files."""
    if _cfg is None:
        return {"error": "Tools not configured"}
    if _cfg.reference_dir is None:
        return {"files": [], "message": "No reference directory configured"}

    files = []
    for subdir in sorted(_cfg.reference_dir.iterdir()):
        if subdir.is_dir():
            for f in sorted(subdir.glob("*")):
                if f.is_file():
                    files.append({"dir": subdir.name, "file": f.name, "size": f.stat().st_size})
        elif subdir.is_file():
            files.append({"dir": "", "file": subdir.name, "size": subdir.stat().st_size})
    return {"files": files}


def search_experiments(
    status: str = None,
    task_type: str = None,
    name_contains: str = None,
    min_score: float = None,
    limit: int = 20,
) -> dict:
    """Search and filter experiments."""
    all_exps = db.get_all_experiments(limit=2000)
    results = []
    for e in all_exps:
        if status and e.get("status") != status:
            continue
        if task_type and e.get("task_type") != task_type:
            continue
        if name_contains and name_contains.lower() not in (e.get("name") or "").lower():
            continue
        if min_score is not None:
            score = e.get("test_score")
            if score is None or float(score) < min_score:
                continue
        results.append({
            "name": e.get("name"),
            "status": e.get("status"),
            "task_type": e.get("task_type"),
            "test_score": e.get("test_score"),
            "val_score": e.get("val_score"),
            "cv_score": e.get("cv_score"),
            "improved": e.get("improved", False),
            "parent_experiment": e.get("parent_experiment"),
            "created_at": e.get("created_at"),
        })
        if len(results) >= limit:
            break
    return {"experiments": results, "total_matched": len(results)}


def get_experiment_metrics(experiment_name: str) -> dict:
    """Get detailed evaluation metrics from an experiment."""
    if _cfg is None:
        return {"error": "Tools not configured"}

    exp_dir = _cfg.experiments_dir / experiment_name
    for fname in ["eval_results.json", "metrics.json"]:
        fp = exp_dir / "output" / fname
        if fp.exists():
            try:
                data = json.loads(fp.read_text(errors="replace"))
                return {"experiment_name": experiment_name, "source": fname, "metrics": data}
            except json.JSONDecodeError as exc:
                return {"error": f"Failed to parse {fname}: {exc}"}
    return {"error": f"No metrics files found for experiment '{experiment_name}'"}


def get_oof_registry() -> dict:
    """Get the OOF predictions registry for stacking planning."""
    if _cfg is None:
        return {"error": "Tools not configured"}

    registry_path = _cfg.data_dir / "stacking_oof_registry.json"
    if not registry_path.exists():
        return {"entries": [], "message": "No OOF registry yet. Run OOF/CV on experiments first."}

    try:
        data = json.loads(registry_path.read_text(errors="replace"))
        if not isinstance(data, list):
            return {"entries": [], "message": "Invalid registry format"}
        entries = []
        for row in data:
            if not isinstance(row, dict):
                continue
            entries.append({
                "experiment": row.get("experiment"),
                "score": row.get("score"),
                "family": row.get("family"),
                "path_exists": Path(row.get("path", "")).exists(),
                "folds": row.get("folds"),
                "splitter": row.get("splitter"),
            })
        return {"entries": entries}
    except Exception as exc:
        return {"error": f"Failed to read OOF registry: {exc}"}


# Dispatch function for tool calls
def dispatch_tool_call(tool_name: str, arguments: dict) -> dict:
    """Dispatch a tool call to the appropriate function."""
    handlers = {
        "get_experiment_summary": get_experiment_summary,
        "get_experiment_code": get_experiment_code,
        "get_experiment_report": get_experiment_report,
        "get_analyst_report": get_analyst_report,
        "get_best_solution_code": get_best_solution_code,
        "get_reference_code": get_reference_code,
        "get_diff_between_experiments": get_diff_between_experiments,
        "list_analyst_reports": list_analyst_reports,
        "get_training_logs": get_training_logs,
        "list_reference_files": list_reference_files,
        "search_experiments": search_experiments,
        "get_experiment_metrics": get_experiment_metrics,
        "get_oof_registry": get_oof_registry,
    }
    
    handler = handlers.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}
    
    return handler(**arguments)
