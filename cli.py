"""
awitune CLI entry point.

Usage:
    python -m awitune run ./projects/my_project/
    python -m awitune run ./projects/my_project/ --port 8090 --no-worker
"""

import argparse
import sys
from pathlib import Path

from .config import load_config


def cmd_reset(args):
    """Reset experiment DB for a project."""
    import shutil
    cfg = load_config(args.project_dir)

    db_path = cfg.experiments_dir / "experiments.db"
    removed = []

    if db_path.exists():
        db_path.unlink()
        removed.append(str(db_path))

    if args.experiments and cfg.experiments_dir.exists():
        for d in cfg.experiments_dir.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))

    if removed:
        print(f"Removed {len(removed)} item(s):")
        for r in removed:
            print(f"  - {r}")
    else:
        print("Nothing to reset.")


def cmd_run(args):
    """Start the dashboard for a project."""
    cfg = load_config(args.project_dir)

    if args.gpus:
        cfg.gpus = [int(g) for g in args.gpus.split(",")]

    from . import db, idea_feeder, generate_ideas
    from .dashboard import app, init_app, start_proxy, start_worker, rt, recover_orphaned_experiments

    init_app(cfg)

    idea_feeder.configure(cfg)
    generate_ideas.configure(cfg)

    if not args.no_proxy:
        print("Auto-starting proxy...")
        start_proxy()

    if not args.no_worker:
        print("Auto-starting worker...")
        start_worker()

    rt.sync_running_from_docker()
    print(f"Running GPUs: {dict(rt.running_gpus)}")

    print("Checking for orphaned experiments...")
    recover_orphaned_experiments()

    queued_exps = db.get_all_experiments(limit=200, status="queued")
    if queued_exps:
        queued_exps.sort(key=lambda e: e.get("created_at", ""))
        manual_count = auto_count = 0
        for exp in queued_exps:
            item = {
                "id": exp["name"],
                "prompt": exp.get("prompt", ""),
                "base_solution": exp.get("base_solution", "") or str(cfg.solutions_dir / "baseline"),
            }
            is_auto = exp["name"].startswith("auto_")
            if is_auto:
                item["auto"] = True
            with rt.lock:
                if is_auto:
                    rt.auto_queue.append(item)
                    auto_count += 1
                else:
                    rt.manual_queue.append(item)
                    manual_count += 1
        print(f"  Re-queued {manual_count} manual + {auto_count} auto experiment(s)")

    port = args.port or 8090
    host = args.host or "::"
    print(f"\n  awitune dashboard: http://[::]:{port}")
    print(f"  Project: {cfg.name} ({cfg.project_dir})\n")
    app.run(host=host, port=port, debug=False, threaded=True)


def main():
    parser = argparse.ArgumentParser(prog="awitune", description="ML experiment orchestrator")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Start dashboard for a project")
    run_p.add_argument("project_dir", type=str, help="Path to project directory")
    run_p.add_argument("--port", type=int, default=8090)
    run_p.add_argument("--host", type=str, default="::")
    run_p.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g. 0,1,2)")
    run_p.add_argument("--no-proxy", action="store_true")
    run_p.add_argument("--no-worker", action="store_true")

    reset_p = sub.add_parser("reset", help="Reset experiment DB and idea cache")
    reset_p.add_argument("project_dir", type=str, help="Path to project directory")
    reset_p.add_argument("--experiments", action="store_true",
                         help="Also delete experiment directories (workspaces, outputs, logs)")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "reset":
        cmd_reset(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
