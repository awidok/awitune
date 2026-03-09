#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description="Train tiny global ridge blender from selected_solutions manifest.")
    p.add_argument("--project-dir", required=True, help="Path to project dir, e.g. projects/data_fusion_2026")
    p.add_argument("--manifest", required=True, help="Path to stacking_manifest.json")
    p.add_argument("--lambda-ridge", type=float, default=100.0, help="L2 regularization")
    p.add_argument("--output-dir", default="", help="Output dir, default experiments/manual_small_ridge_<ts>")
    return p.parse_args()


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
        raise RuntimeError("Need at least 2 selected_solutions")

    exp_root = project_dir / "experiments"
    data_dir = project_dir / "data"

    val_y_df = pl.read_parquet(data_dir / "local_val_target.parquet").sort("customer_id")
    test_y_df = pl.read_parquet(data_dir / "local_test_target.parquet").sort("customer_id")
    target_cols = [c for c in val_y_df.columns if c.startswith("target_")]
    pred_cols = [c.replace("target_", "predict_") for c in target_cols]

    val_parts = []
    test_parts = []
    sub_parts = []
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
        sub_parts.append(sub_df.select(pred_cols).to_numpy())
        if submit_customer_id is None:
            submit_customer_id = sub_df["customer_id"]
        used.append(name)

    if len(used) < 2:
        raise RuntimeError("Not enough valid models after filtering")

    # [N, T, M]
    val_stack = np.stack(val_parts, axis=2)
    test_stack = np.stack(test_parts, axis=2)
    sub_stack = np.stack(sub_parts, axis=2)
    y_val = val_y_df.select(target_cols).to_numpy()
    y_test = test_y_df.select(target_cols).to_numpy()

    n_models = val_stack.shape[2]
    x = val_stack.reshape(-1, n_models)
    y = y_val.reshape(-1)
    lam = float(args.lambda_ridge)

    a = x.T @ x + lam * np.eye(n_models)
    b = x.T @ y
    w = np.linalg.solve(a, b)
    w = np.clip(w, 0.0, None)
    if float(w.sum()) > 0:
        w = w / float(w.sum())
    else:
        w = np.full(n_models, 1.0 / n_models)

    val_pred = np.tensordot(val_stack, w, axes=([2], [0]))
    test_pred = np.tensordot(test_stack, w, axes=([2], [0]))
    sub_pred = np.tensordot(sub_stack, w, axes=([2], [0]))

    val_auc = macro_auc(y_val, val_pred)
    test_auc = macro_auc(y_test, test_pred)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (exp_root / f"manual_small_ridge_{ts}")
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

    manifest_out = {
        "created_at": datetime.now().isoformat(),
        "method": "global_ridge_weighted_average",
        "project_dir": str(project_dir),
        "source_manifest": str(Path(args.manifest).resolve()),
        "lambda_ridge": lam,
        "used_solutions": used,
        "weights": [float(x) for x in w.tolist()],
        "val_macro_roc_auc": val_auc,
        "test_macro_roc_auc": test_auc,
    }
    (out_dir / "small_ridge_manifest.json").write_text(json.dumps(manifest_out, indent=2))

    print(f"Used models: {len(used)}")
    print(f"Val macro AUC:  {val_auc:.6f}")
    print(f"Test macro AUC: {test_auc:.6f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
