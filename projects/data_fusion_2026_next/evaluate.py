"""
Evaluate agent output against local_test labels.

Usage:
    python evaluate.py <output_dir> [--data-dir <data_dir>]

Reads:
    <output_dir>/val_predictions.parquet
    <output_dir>/test_predictions.parquet
    <data_dir>/local_val_target.parquet
    <data_dir>/local_test_target.parquet

Outputs:
    <output_dir>/eval_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent
DATA = PROJECT_DIR / "data"


def compute_macro_auc(predictions_path: Path, labels_path: Path, split_name: str) -> dict:
    """Compute macro ROC-AUC for a split."""
    preds_df = pl.read_parquet(predictions_path)
    labels_df = pl.read_parquet(labels_path)

    # Join on customer_id to align
    target_cols = sorted([c for c in labels_df.columns if c.startswith("target_")])
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]

    # Ensure alignment by customer_id
    merged = labels_df.join(preds_df, on="customer_id", how="inner")

    if len(merged) != len(labels_df):
        print(f"  WARNING: {split_name} — merged {len(merged)} rows, expected {len(labels_df)}")

    y_true = merged.select(target_cols).to_numpy()
    y_pred = merged.select(predict_cols).to_numpy()

    per_target = {}
    aucs = []
    for j, col in enumerate(target_cols):
        try:
            auc = roc_auc_score(y_true[:, j], y_pred[:, j])
        except ValueError:
            auc = 0.5
        per_target[col] = round(auc, 6)
        aucs.append(auc)

    macro_auc = float(np.mean(aucs))
    return {
        f"{split_name}_macro_roc_auc": round(macro_auc, 6),
        f"{split_name}_per_target_auc": per_target,
        f"{split_name}_n_rows": len(merged),
    }


def main():
    global DATA
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <output_dir> [--data-dir <data_dir>]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv):
            DATA = Path(sys.argv[idx + 1])

    results = {}

    # Evaluate val predictions
    val_pred_path = output_dir / "val_predictions.parquet"
    val_label_path = DATA / "local_val_target.parquet"
    if val_pred_path.exists() and val_label_path.exists():
        print(f"Evaluating val predictions …")
        val_results = compute_macro_auc(val_pred_path, val_label_path, "val")
        results.update(val_results)
        print(f"  Val Macro ROC-AUC: {val_results['val_macro_roc_auc']:.6f}")
    else:
        print(f"  Skipping val: pred={val_pred_path.exists()}, labels={val_label_path.exists()}")

    # Evaluate test predictions
    test_pred_path = output_dir / "test_predictions.parquet"
    test_label_path = DATA / "local_test_target.parquet"
    if test_pred_path.exists() and test_label_path.exists():
        print(f"Evaluating test predictions …")
        test_results = compute_macro_auc(test_pred_path, test_label_path, "test")
        results.update(test_results)
        print(f"  Test Macro ROC-AUC: {test_results['test_macro_roc_auc']:.6f}")
    else:
        print(f"  Skipping test: pred={test_pred_path.exists()}, labels={test_label_path.exists()}")

    # Check report exists
    report_path = output_dir / "report.md"
    results["report_exists"] = report_path.exists()
    if report_path.exists():
        report_size = report_path.stat().st_size
        print(f"  Report: {report_size} bytes")
    else:
        print(f"  WARNING: report.md not found — agent should generate an experiment report")

    # Check submission exists
    sub_path = output_dir / "submission.parquet"
    results["submission_exists"] = sub_path.exists()
    if sub_path.exists():
        sub_df = pl.read_parquet(sub_path)
        results["submission_rows"] = len(sub_df)
        results["submission_cols"] = len(sub_df.columns)
        print(f"  Submission: {len(sub_df)} rows, {len(sub_df.columns)} cols")

    # Save results
    eval_path = output_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_path}")

    # Print summary
    print(f"\n{'='*60}")
    if "val_macro_roc_auc" in results:
        print(f"  Val  Macro ROC-AUC: {results['val_macro_roc_auc']:.6f}")
    if "test_macro_roc_auc" in results:
        print(f"  Test Macro ROC-AUC: {results['test_macro_roc_auc']:.6f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
