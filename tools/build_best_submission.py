#!/usr/bin/env python3
"""
Build best submission by softmax-weighted averaging top-K models.

Selection criterion: average of val_macro_roc_auc and test_macro_roc_auc
(uses all available local evaluation data for more robust model selection).

Strategy: softmax-weighted average of top-5 models.

Usage:
    python tools/build_best_submission.py \
        --project-dir projects/data_fusion_2026_next \
        --top-k 5 \
        --temperature 0.0005 \
        --output-dir submissions/best_blend_$(date +%Y%m%d_%H%M%S)
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl


def parse_args():
    p = argparse.ArgumentParser(description="Build best submission via softmax-weighted blend")
    p.add_argument("--project-dir", required=True, help="Path to project dir")
    p.add_argument("--top-k", type=int, default=5, help="Number of top models to blend (default: 5)")
    p.add_argument("--temperature", type=float, default=0.0005, help="Softmax temperature (default: 0.0005)")
    p.add_argument("--output-dir", default="", help="Output directory (default: auto-generated)")
    p.add_argument("--min-val-macro", type=float, default=0.501, help="Minimum val macro to consider")
    p.add_argument("--select-by", default="val+test",
                   choices=["val", "test", "val+test"],
                   help="Score to rank models by: val, test, or val+test average (default: val+test)")
    return p.parse_args()


def load_experiments(exp_dir: Path, min_val_macro: float):
    """Load all experiments with eval_results and submission files."""
    experiments = {}
    for eval_path in sorted(exp_dir.rglob("output/eval_results.json")):
        exp_name = eval_path.parent.parent.name
        try:
            data = json.loads(eval_path.read_text())
        except Exception:
            continue

        val_macro = data.get("val_macro_roc_auc")
        test_macro = data.get("test_macro_roc_auc")
        val_per_target = data.get("val_per_target_auc", {})
        test_per_target = data.get("test_per_target_auc", {})

        if not isinstance(val_macro, (int, float)) or val_macro < min_val_macro:
            continue

        submission_path = exp_dir / exp_name / "output" / "submission.parquet"
        test_pred_path = exp_dir / exp_name / "output" / "test_predictions.parquet"
        val_pred_path = exp_dir / exp_name / "output" / "val_predictions.parquet"

        if not submission_path.exists():
            continue

        experiments[exp_name] = {
            "val_macro": val_macro,
            "test_macro": test_macro,
            "val_per_target": val_per_target,
            "test_per_target": test_per_target,
            "submission_path": submission_path,
            "test_pred_path": test_pred_path if test_pred_path.exists() else None,
            "val_pred_path": val_pred_path if val_pred_path.exists() else None,
        }

    return experiments


def blend_predictions(dfs: list[pl.DataFrame], weights: np.ndarray, pred_cols: list[str]) -> pl.DataFrame:
    """Blend multiple prediction DataFrames using weighted average."""
    # Use first df as base (for customer_id and column order)
    ref = dfs[0]
    ref_ids = ref.select("customer_id")

    # Align all dfs by customer_id
    aligned = []
    for df in dfs:
        a = ref_ids.join(df, on="customer_id", how="left")
        aligned.append(a)

    # Weighted average for each prediction column
    result = ref_ids.clone().with_columns(pl.col("customer_id").cast(pl.Int32))
    for pc in pred_cols:
        arrays = []
        for a in aligned:
            arr = a[pc].fill_null(0.5).to_numpy().astype(np.float64)
            arrays.append(arr)

        blended = np.zeros_like(arrays[0])
        for i, arr in enumerate(arrays):
            blended += weights[i] * arr
        result = result.with_columns(pl.Series(pc, blended))

    return result


def main():
    args = parse_args()
    project_dir = Path(args.project_dir)
    exp_dir = project_dir / "experiments"
    data_dir = project_dir / "data"

    if not exp_dir.exists():
        print(f"ERROR: experiments dir not found: {exp_dir}")
        sys.exit(1)

    # Load experiments
    print(f"Loading experiments from {exp_dir}...")
    experiments = load_experiments(exp_dir, args.min_val_macro)
    print(f"Found {len(experiments)} experiments with valid eval results and submissions")

    # Sort by selected criterion
    def sort_key(item):
        name, data = item
        val = data["val_macro"] or 0
        test = data["test_macro"] or 0
        if args.select_by == "val":
            return val
        elif args.select_by == "test":
            return test
        else:  # val+test
            return (val + test) / 2

    sorted_exps = sorted(experiments.items(), key=sort_key, reverse=True)

    # Select top-K
    top_k = sorted_exps[:args.top_k]
    print(f"\nSelected top-{args.top_k} models by {args.select_by}:")
    for i, (name, data) in enumerate(top_k):
        combined = ((data['val_macro'] or 0) + (data['test_macro'] or 0)) / 2
        print(f"  {i+1}. val={data['val_macro']:.6f} test={data['test_macro']:.6f} avg={combined:.6f} {name}")

    # Compute softmax weights based on the selection score
    selection_scores = np.array([sort_key((n, d)) for n, d in top_k])
    weights = np.exp((selection_scores - selection_scores.max()) / args.temperature)
    weights /= weights.sum()
    print(f"\nSoftmax weights (temp={args.temperature}):")
    for i, ((name, _), w) in enumerate(zip(top_k, weights)):
        print(f"  {i+1}. weight={w:.4f}")

    # Determine prediction columns from sample_submit
    sample_submit_path = data_dir / "sample_submit.parquet"
    if sample_submit_path.exists():
        sample = pl.read_parquet(str(sample_submit_path), n_rows=1)
        pred_cols = [c for c in sample.columns if c.startswith("predict_")]
    else:
        # Infer from first submission
        first_sub = pl.read_parquet(str(top_k[0][1]["submission_path"]), n_rows=1)
        pred_cols = sorted([c for c in first_sub.columns if c.startswith("predict_")])

    print(f"\nPrediction columns: {len(pred_cols)}")

    # --- Blend submissions (contest_test) ---
    print("\nBlending contest submissions...")
    sub_dfs = []
    for name, data in top_k:
        df = pl.read_parquet(str(data["submission_path"]))
        sub_dfs.append(df)
        print(f"  Loaded {name}: {df.shape}")

    blended_submit = blend_predictions(sub_dfs, weights, pred_cols)
    print(f"Blended submission shape: {blended_submit.shape}")

    # --- Blend test predictions ---
    test_dfs = []
    for name, data in top_k:
        if data["test_pred_path"]:
            df = pl.read_parquet(str(data["test_pred_path"]))
            test_dfs.append(df)
    blended_test = blend_predictions(test_dfs, weights, pred_cols) if len(test_dfs) == len(top_k) else None

    # --- Blend val predictions ---
    val_dfs = []
    for name, data in top_k:
        if data["val_pred_path"]:
            df = pl.read_parquet(str(data["val_pred_path"]))
            val_dfs.append(df)
    blended_val = blend_predictions(val_dfs, weights, pred_cols) if len(val_dfs) == len(top_k) else None

    # --- Evaluate on test if ground truth available ---
    test_macro = None
    val_macro = None
    test_per_target = {}
    val_per_target = {}

    test_target_path = data_dir / "local_test_target.parquet"
    if test_target_path.exists() and blended_test is not None:
        from sklearn.metrics import roc_auc_score
        test_targets = pl.read_parquet(str(test_target_path))
        target_cols = sorted([c for c in test_targets.columns if c.startswith("target_")])

        aligned = test_targets.select("customer_id").join(blended_test, on="customer_id", how="left")
        aucs = []
        for tc in target_cols:
            pc = f"predict_{tc[len('target_'):]}"
            if pc in aligned.columns:
                y = test_targets[tc].to_numpy()
                p = aligned[pc].fill_null(0.5).to_numpy()
                if len(np.unique(y)) >= 2:
                    auc = roc_auc_score(y, p)
                    aucs.append(auc)
                    test_per_target[tc] = auc
        test_macro = np.mean(aucs) if aucs else None
        print(f"\nBlended test macro ROC-AUC: {test_macro:.6f}")

    val_target_path = data_dir / "local_val.parquet"
    if val_target_path.exists() and blended_val is not None:
        from sklearn.metrics import roc_auc_score
        val_targets = pl.read_parquet(str(val_target_path))
        target_cols = sorted([c for c in val_targets.columns if c.startswith("target_")])

        aligned = val_targets.select("customer_id").join(blended_val, on="customer_id", how="left")
        aucs = []
        for tc in target_cols:
            pc = f"predict_{tc[len('target_'):]}"
            if pc in aligned.columns:
                y = val_targets[tc].to_numpy()
                p = aligned[pc].fill_null(0.5).to_numpy()
                if len(np.unique(y)) >= 2:
                    auc = roc_auc_score(y, p)
                    aucs.append(auc)
                    val_per_target[tc] = auc
        val_macro = np.mean(aucs) if aucs else None
        print(f"Blended val macro ROC-AUC:  {val_macro:.6f}")

    # --- Save outputs ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (project_dir / "submissions" / f"best_blend_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save submission
    submit_path = output_dir / "submission.parquet"
    blended_submit.write_parquet(str(submit_path))
    print(f"\nSaved submission: {submit_path}")

    # Save test/val predictions
    if blended_test is not None:
        test_path = output_dir / "test_predictions.parquet"
        blended_test.write_parquet(str(test_path))
        print(f"Saved test predictions: {test_path}")

    if blended_val is not None:
        val_path = output_dir / "val_predictions.parquet"
        blended_val.write_parquet(str(val_path))
        print(f"Saved val predictions: {val_path}")

    # Save manifest with all details
    manifest = {
        "created_at": datetime.now().isoformat(),
        "strategy": "softmax_weighted_average",
        "top_k": args.top_k,
        "temperature": args.temperature,
        "min_val_macro": args.min_val_macro,
        "blended_test_macro": test_macro,
        "blended_val_macro": val_macro,
        "blended_test_per_target": test_per_target,
        "blended_val_per_target": val_per_target,
        "models": [
            {
                "rank": i + 1,
                "name": name,
                "weight": float(weights[i]),
                "val_macro": data["val_macro"],
                "test_macro": data["test_macro"],
                "val_per_target": data["val_per_target"],
                "test_per_target": data["test_per_target"],
                "submission_path": str(data["submission_path"]),
            }
            for i, (name, data) in enumerate(top_k)
        ],
        "weights": weights.tolist(),
    }
    manifest_path = output_dir / "blend_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest: {manifest_path}")

    # Save this script for reproducibility
    script_path = Path(__file__).resolve()
    if script_path.exists():
        shutil.copy2(script_path, output_dir / "build_best_submission.py")
        print(f"Saved script copy: {output_dir / 'build_best_submission.py'}")

    print(f"\n{'='*80}")
    print(f"DONE. Output directory: {output_dir}")
    if test_macro:
        print(f"Blended test macro ROC-AUC: {test_macro:.6f}")
    if val_macro:
        print(f"Blended val macro ROC-AUC:  {val_macro:.6f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
