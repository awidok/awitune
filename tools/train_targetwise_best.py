#!/usr/bin/env python3
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description="Select best model per target by val AUC.")
    p.add_argument("--project-dir", required=True, help="Path to project dir, e.g. projects/data_fusion_2026")
    p.add_argument("--solutions-file", default="", help="Optional JSON list[str] with experiment names")
    p.add_argument("--max-candidates", type=int, default=0, help="Optional cap for candidate count by test_score")
    p.add_argument("--output-dir", default="", help="Output dir (default: experiments/manual_targetwise_best_<ts>)")
    return p.parse_args()


def load_candidates(project_dir: Path, solutions_file: str, max_candidates: int) -> list[str]:
    exp_root = project_dir / "experiments"
    if solutions_file:
        names = json.loads(Path(solutions_file).read_text())
        return [str(x).strip() for x in names if str(x).strip()]

    conn = sqlite3.connect(exp_root / "experiments.db")
    cur = conn.cursor()
    rows = cur.execute(
        """
        select name
        from experiments
        where status='completed' and task_type='experiment'
        order by coalesce(test_score, 0) desc
        """
    ).fetchall()
    names = [r[0] for r in rows]
    if max_candidates > 0:
        names = names[:max_candidates]
    return names


def sorted_df(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort("customer_id")


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    exp_root = project_dir / "experiments"
    data_dir = project_dir / "data"

    y_df = sorted_df(data_dir / "local_val_target.parquet")
    target_cols = [c for c in y_df.columns if c.startswith("target_")]
    pred_cols = [f"predict_{c[len('target_'):]}" for c in target_cols]

    candidates = load_candidates(project_dir, args.solutions_file, args.max_candidates)
    if len(candidates) < 2:
        raise RuntimeError("Need at least 2 candidate solutions")

    n_targets = len(target_cols)
    n_val = y_df.height
    y = y_df.select(target_cols).to_numpy()
    best_auc = np.full(n_targets, -1.0, dtype=np.float64)
    best_model = [""] * n_targets
    best_val = np.zeros((n_val, n_targets), dtype=np.float32)
    best_test = None
    best_submit = None
    used_candidates = []

    for name in candidates:
        out = exp_root / name / "output"
        val_fp = out / "val_predictions.parquet"
        test_fp = out / "test_predictions.parquet"
        sub_fp = out / "submission.parquet"
        if not (val_fp.exists() and test_fp.exists() and sub_fp.exists()):
            continue

        val_df = sorted_df(val_fp).select(["customer_id"] + pred_cols)
        test_df = sorted_df(test_fp).select(["customer_id"] + pred_cols)
        sub_df = sorted_df(sub_fp).select(["customer_id"] + pred_cols)
        if not val_df["customer_id"].equals(y_df["customer_id"]):
            continue

        val_np = val_df.select(pred_cols).to_numpy()
        test_np = test_df.select(pred_cols).to_numpy()
        sub_np = sub_df.select(pred_cols).to_numpy()

        if best_test is None:
            best_test = np.zeros_like(test_np, dtype=np.float32)
        if best_submit is None:
            best_submit = np.zeros_like(sub_np, dtype=np.float32)

        used_candidates.append(name)
        for j in range(n_targets):
            try:
                auc = roc_auc_score(y[:, j], val_np[:, j])
            except ValueError:
                auc = 0.5
            if auc > best_auc[j]:
                best_auc[j] = float(auc)
                best_model[j] = name
                best_val[:, j] = val_np[:, j]
                best_test[:, j] = test_np[:, j]
                best_submit[:, j] = sub_np[:, j]

    if not used_candidates:
        raise RuntimeError("No valid candidate experiments found")

    val_macro = float(np.mean(best_auc))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (exp_root / f"manual_targetwise_best_{ts}")
    out_data = out_dir / "output"
    out_data.mkdir(parents=True, exist_ok=True)

    val_out = pl.DataFrame({"customer_id": y_df["customer_id"]})
    test_out = pl.DataFrame({"customer_id": sorted_df(exp_root / used_candidates[0] / "output" / "test_predictions.parquet")["customer_id"]})
    sub_out = pl.DataFrame({"customer_id": sorted_df(exp_root / used_candidates[0] / "output" / "submission.parquet")["customer_id"]})
    for j, p_col in enumerate(pred_cols):
        val_out = val_out.with_columns(pl.Series(p_col, best_val[:, j]))
        test_out = test_out.with_columns(pl.Series(p_col, best_test[:, j]))
        sub_out = sub_out.with_columns(pl.Series(p_col, best_submit[:, j]))

    val_out.write_parquet(out_data / "val_predictions.parquet")
    test_out.write_parquet(out_data / "test_predictions.parquet")
    sub_out.write_parquet(out_data / "submission.parquet")

    selected_by_target = {}
    for t_col, auc, model in zip(target_cols, best_auc.tolist(), best_model):
        selected_by_target[t_col] = {"model": model, "val_auc": round(float(auc), 6)}

    manifest = {
        "created_at": datetime.now().isoformat(),
        "project_dir": str(project_dir),
        "output_dir": str(out_dir),
        "selector": "best_per_target_by_val_auc",
        "num_candidates_requested": len(candidates),
        "num_candidates_used": len(used_candidates),
        "candidates_used": used_candidates,
        "val_macro_roc_auc": round(val_macro, 6),
        "selected_by_target": selected_by_target,
    }
    (out_dir / "targetwise_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Candidates used: {len(used_candidates)}")
    print(f"Val macro AUC (target-wise best): {val_macro:.6f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
