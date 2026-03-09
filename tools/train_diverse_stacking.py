#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description="Train stacking model from an explicit solutions list.")
    p.add_argument("--project-dir", required=True, help="Path to project dir, e.g. projects/data_fusion_2026")
    p.add_argument("--solutions-file", default="", help="JSON file with list[str] of experiment names")
    p.add_argument("--solutions", nargs="*", default=None, help="Explicit experiment names")
    p.add_argument("--output-dir", default="", help="Output experiment dir (default: experiments/manual_stacking_<ts>)")
    p.add_argument("--max-iter", type=int, default=300)
    return p.parse_args()


def load_solution_names(args):
    if args.solutions:
        names = [x.strip() for x in args.solutions if str(x).strip()]
    elif args.solutions_file:
        names = json.loads(Path(args.solutions_file).read_text())
        names = [str(x).strip() for x in names if str(x).strip()]
    else:
        raise ValueError("Provide --solutions or --solutions-file")
    if len(names) < 2:
        raise ValueError("Need at least 2 solutions for stacking")
    return names


def _sorted_by_id(df: pl.DataFrame) -> pl.DataFrame:
    if "customer_id" not in df.columns:
        raise ValueError("Missing customer_id")
    return df.sort("customer_id")


def _validate_prediction_columns(df: pl.DataFrame):
    pred_cols = [c for c in df.columns if c.startswith("predict_")]
    if not pred_cols:
        raise ValueError("No predict_* columns found")
    return ["customer_id"] + pred_cols


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    exp_root = project_dir / "experiments"
    data_dir = project_dir / "data"
    y_path = data_dir / "local_val_target.parquet"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing validation targets: {y_path}")

    names = load_solution_names(args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (exp_root / f"manual_stacking_{ts}")
    out_data = out_dir / "output"
    out_data.mkdir(parents=True, exist_ok=True)

    y_df = _sorted_by_id(pl.read_parquet(y_path))
    target_cols = [c for c in y_df.columns if c.startswith("target_")]
    if not target_cols:
        raise ValueError("No target_* columns in local_val_target.parquet")
    pred_cols = [f"predict_{c[len('target_'):]}" for c in target_cols]

    val_frames = []
    test_frames = []
    submit_frames = []
    used = []
    for name in names:
        val_fp = exp_root / name / "output" / "val_predictions.parquet"
        test_fp = exp_root / name / "output" / "test_predictions.parquet"
        submit_fp = exp_root / name / "output" / "submission.parquet"
        if not val_fp.exists() or not test_fp.exists() or not submit_fp.exists():
            continue
        val_df = _sorted_by_id(pl.read_parquet(val_fp))
        test_df = _sorted_by_id(pl.read_parquet(test_fp))
        submit_df = _sorted_by_id(pl.read_parquet(submit_fp))
        _validate_prediction_columns(val_df)
        _validate_prediction_columns(test_df)
        missing_val = [c for c in pred_cols if c not in val_df.columns]
        missing_test = [c for c in pred_cols if c not in test_df.columns]
        missing_submit = [c for c in pred_cols if c not in submit_df.columns]
        if missing_val or missing_test or missing_submit:
            continue
        if not val_df["customer_id"].equals(y_df["customer_id"]):
            continue
        val_frames.append(val_df.select(["customer_id"] + pred_cols))
        test_frames.append(test_df.select(["customer_id"] + pred_cols))
        submit_frames.append(submit_df.select(["customer_id"] + pred_cols))
        used.append(name)

    if len(used) < 2:
        raise RuntimeError("Not enough valid solutions after filtering")

    n_val = y_df.height
    n_test = test_frames[0].height
    n_submit = submit_frames[0].height
    y = y_df.select(target_cols).to_numpy()
    val_pred = np.zeros((n_val, len(target_cols)), dtype=np.float64)
    test_pred = np.zeros((n_test, len(target_cols)), dtype=np.float64)
    submit_pred = np.zeros((n_submit, len(target_cols)), dtype=np.float64)
    per_target_auc = {}

    for j, (t_col, p_col) in enumerate(zip(target_cols, pred_cols)):
        x_val = np.column_stack([df[p_col].to_numpy() for df in val_frames])
        x_test = np.column_stack([df[p_col].to_numpy() for df in test_frames])
        x_submit = np.column_stack([df[p_col].to_numpy() for df in submit_frames])
        y_col = y[:, j].astype(int)

        if np.unique(y_col).shape[0] < 2:
            val_pred[:, j] = np.clip(x_val.mean(axis=1), 0.0, 1.0)
            test_pred[:, j] = np.clip(x_test.mean(axis=1), 0.0, 1.0)
            submit_pred[:, j] = np.clip(x_submit.mean(axis=1), 0.0, 1.0)
            per_target_auc[t_col] = None
            continue

        clf = LogisticRegression(
            solver="liblinear",
            max_iter=args.max_iter,
        )
        clf.fit(x_val, y_col)
        val_pred[:, j] = clf.predict_proba(x_val)[:, 1]
        test_pred[:, j] = clf.predict_proba(x_test)[:, 1]
        submit_pred[:, j] = clf.predict_proba(x_submit)[:, 1]
        per_target_auc[t_col] = float(roc_auc_score(y_col, val_pred[:, j]))

    macro_auc = float(roc_auc_score(y, val_pred, average="macro"))

    val_out = pl.DataFrame({"customer_id": y_df["customer_id"]})
    test_out = pl.DataFrame({"customer_id": test_frames[0]["customer_id"]})
    sub_out = pl.DataFrame({"customer_id": submit_frames[0]["customer_id"]})
    for j, p_col in enumerate(pred_cols):
        val_out = val_out.with_columns(pl.Series(p_col, val_pred[:, j]))
        test_out = test_out.with_columns(pl.Series(p_col, test_pred[:, j]))
        sub_out = sub_out.with_columns(pl.Series(p_col, submit_pred[:, j]))

    val_out.write_parquet(out_data / "val_predictions.parquet")
    test_out.write_parquet(out_data / "test_predictions.parquet")
    sub_out.write_parquet(out_data / "submission.parquet")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "project_dir": str(project_dir),
        "output_dir": str(out_dir),
        "metric": "val_macro_roc_auc",
        "val_macro_roc_auc": macro_auc,
        "num_solutions_requested": len(names),
        "num_solutions_used": len(used),
        "used_solutions": used,
        "requested_solutions": names,
        "per_target_auc": per_target_auc,
    }
    (out_dir / "stacking_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "used_solutions.json").write_text(json.dumps(used, indent=2))

    print(f"Used {len(used)} solutions")
    print(f"Val macro AUC: {macro_auc:.6f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
