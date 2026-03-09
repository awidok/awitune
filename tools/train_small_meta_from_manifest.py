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
    p = argparse.ArgumentParser(description="Train small per-target meta model from selected_solutions manifest")
    p.add_argument("--project-dir", required=True, help="Path to project dir, e.g. projects/data_fusion_2026")
    p.add_argument("--manifest", required=True, help="Path to stacking_manifest.json with selected_solutions")
    p.add_argument("--output-dir", default="", help="Output dir, default experiments/manual_small_meta_<ts>")
    p.add_argument("--c", type=float, default=0.1, help="Inverse regularization strength for LogisticRegression")
    p.add_argument("--max-iter", type=int, default=400)
    return p.parse_args()


def macro_auc(y_true, y_pred):
    aucs = []
    for j in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, j], y_pred[:, j])
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    return float(np.mean(aucs))


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    selected = [str(x) for x in manifest.get("selected_solutions", [])]
    if len(selected) < 2:
        raise RuntimeError("Need at least 2 selected_solutions in manifest")

    exp_root = project_dir / "experiments"
    data_dir = project_dir / "data"
    val_y_df = pl.read_parquet(data_dir / "local_val_target.parquet").sort("customer_id")
    test_y_df = pl.read_parquet(data_dir / "local_test_target.parquet").sort("customer_id")
    target_cols = [c for c in val_y_df.columns if c.startswith("target_")]
    pred_cols = [c.replace("target_", "predict_") for c in target_cols]

    val_parts = []
    test_parts = []
    submit_parts = []
    submit_customer_id = None
    used = []

    for name in selected:
        out = exp_root / name / "output"
        val_fp = out / "val_predictions.parquet"
        test_fp = out / "test_predictions.parquet"
        sub_fp = out / "submission.parquet"
        if not (val_fp.exists() and test_fp.exists() and sub_fp.exists()):
            continue
        val_df = pl.read_parquet(val_fp).sort("customer_id")
        test_df = pl.read_parquet(test_fp).sort("customer_id")
        sub_df = pl.read_parquet(sub_fp).sort("customer_id")
        if not all(c in val_df.columns for c in pred_cols):
            continue
        if not val_df["customer_id"].equals(val_y_df["customer_id"]):
            continue
        val_parts.append(val_df.select(pred_cols).to_numpy())
        test_parts.append(test_df.select(pred_cols).to_numpy())
        submit_parts.append(sub_df.select(pred_cols).to_numpy())
        if submit_customer_id is None:
            submit_customer_id = sub_df["customer_id"]
        used.append(name)

    if len(used) < 2:
        raise RuntimeError("Not enough valid solutions after filtering")

    # Shapes: [M, N, T]
    val_stack = np.stack(val_parts, axis=0)
    test_stack = np.stack(test_parts, axis=0)
    sub_stack = np.stack(submit_parts, axis=0)
    y_val = val_y_df.select(target_cols).to_numpy()
    y_test = test_y_df.select(target_cols).to_numpy()

    n_models, n_val, n_targets = val_stack.shape
    n_test = test_stack.shape[1]
    n_sub = sub_stack.shape[1]

    val_pred = np.zeros((n_val, n_targets), dtype=np.float64)
    test_pred = np.zeros((n_test, n_targets), dtype=np.float64)
    sub_pred = np.zeros((n_sub, n_targets), dtype=np.float64)
    model_info = {}

    for j, t_col in enumerate(target_cols):
        x_val = val_stack[:, :, j].T  # [N, M]
        x_test = test_stack[:, :, j].T
        x_sub = sub_stack[:, :, j].T
        y = y_val[:, j].astype(int)
        if np.unique(y).shape[0] < 2:
            val_pred[:, j] = np.clip(x_val.mean(axis=1), 0.0, 1.0)
            test_pred[:, j] = np.clip(x_test.mean(axis=1), 0.0, 1.0)
            sub_pred[:, j] = np.clip(x_sub.mean(axis=1), 0.0, 1.0)
            model_info[t_col] = {"mode": "mean_fallback"}
            continue

        clf = LogisticRegression(
            solver="liblinear",
            C=args.c,
            max_iter=args.max_iter,
        )
        clf.fit(x_val, y)
        val_pred[:, j] = clf.predict_proba(x_val)[:, 1]
        test_pred[:, j] = clf.predict_proba(x_test)[:, 1]
        sub_pred[:, j] = clf.predict_proba(x_sub)[:, 1]
        model_info[t_col] = {
            "mode": "logreg",
            "intercept": float(clf.intercept_[0]),
            "coef": [float(x) for x in clf.coef_[0].tolist()],
        }

    val_auc = macro_auc(y_val, val_pred)
    test_auc = macro_auc(y_test, test_pred)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (exp_root / f"manual_small_meta_{ts}")
    out_data = out_dir / "output"
    out_data.mkdir(parents=True, exist_ok=True)

    val_out = pl.DataFrame({"customer_id": val_y_df["customer_id"]})
    test_out = pl.DataFrame({"customer_id": test_y_df["customer_id"]})
    sub_out = pl.DataFrame({"customer_id": submit_customer_id})
    for j, c in enumerate(pred_cols):
        val_out = val_out.with_columns(pl.Series(c, val_pred[:, j]))
        test_out = test_out.with_columns(pl.Series(c, test_pred[:, j]))
        sub_out = sub_out.with_columns(pl.Series(c, sub_pred[:, j]))

    val_out.write_parquet(out_data / "val_predictions.parquet")
    test_out.write_parquet(out_data / "test_predictions.parquet")
    sub_out.write_parquet(out_data / "submission.parquet")

    (out_data / "eval_results.json").write_text(json.dumps({
        "val_macro_roc_auc": round(val_auc, 6),
        "test_macro_roc_auc": round(test_auc, 6),
    }, indent=2))
    (out_dir / "small_meta_manifest.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "project_dir": str(project_dir),
        "source_manifest": str(Path(args.manifest).resolve()),
        "num_models_used": len(used),
        "used_solutions": used,
        "c": args.c,
        "max_iter": args.max_iter,
        "val_macro_roc_auc": val_auc,
        "test_macro_roc_auc": test_auc,
        "per_target_model_info": model_info,
    }, indent=2))

    print(f"Used models: {len(used)}")
    print(f"Val macro AUC:  {val_auc:.6f}")
    print(f"Test macro AUC: {test_auc:.6f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
