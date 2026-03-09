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
    p = argparse.ArgumentParser(description="Train rank-blend and small CatBoost meta using val-only optimization.")
    p.add_argument("--project-dir", required=True, help="Path to project dir, e.g. projects/data_fusion_2026")
    p.add_argument("--top-k", type=int, default=12, help="Number of candidate base solutions by val_score")
    p.add_argument("--rank-trials", type=int, default=220, help="Random weight trials for rank blending")
    p.add_argument("--cat-iterations", type=int, default=80, help="CatBoost iterations per target")
    p.add_argument("--cat-depth", type=int, default=4, help="CatBoost depth")
    p.add_argument("--cat-thread-count", type=int, default=4, help="CatBoost thread_count")
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


def rank_norm_2d(a: np.ndarray) -> np.ndarray:
    # Per-column rank normalization to [0,1].
    n, m = a.shape
    out = np.empty_like(a, dtype=np.float32)
    for j in range(m):
        order = np.argsort(a[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(n, dtype=np.float32)
        if n > 1:
            ranks /= (n - 1)
        out[:, j] = ranks
    return out


def load_candidates(project_dir: Path, top_k: int):
    exp_root = project_dir / "experiments"
    conn = sqlite3.connect(exp_root / "experiments.db")
    cur = conn.cursor()
    rows = cur.execute(
        """
        select name, coalesce(val_score, 0) as val_score, coalesce(test_score, 0) as test_score
        from experiments
        where status='completed' and task_type='experiment'
        order by val_score desc
        """
    ).fetchall()
    return rows[: max(top_k * 3, top_k)]


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    exp_root = project_dir / "experiments"
    data_dir = project_dir / "data"

    val_y_df = pl.read_parquet(data_dir / "local_val_target.parquet").sort("customer_id")
    test_y_df = pl.read_parquet(data_dir / "local_test_target.parquet").sort("customer_id")
    target_cols = [c for c in val_y_df.columns if c.startswith("target_")]
    pred_cols = [c.replace("target_", "predict_") for c in target_cols]
    y_val = val_y_df.select(target_cols).to_numpy()
    y_test = test_y_df.select(target_cols).to_numpy()

    rows = load_candidates(project_dir, args.top_k)
    candidates = []
    for name, val_score, test_score in rows:
        out = exp_root / name / "output"
        val_fp = out / "val_predictions.parquet"
        test_fp = out / "test_predictions.parquet"
        sub_fp = out / "submission.parquet"
        if not (val_fp.exists() and test_fp.exists() and sub_fp.exists()):
            continue
        v_df = pl.read_parquet(val_fp).sort("customer_id")
        t_df = pl.read_parquet(test_fp).sort("customer_id")
        s_df = pl.read_parquet(sub_fp).sort("customer_id")
        if not all(c in v_df.columns for c in pred_cols):
            continue
        if not v_df["customer_id"].equals(val_y_df["customer_id"]):
            continue
        candidates.append({
            "name": name,
            "val_score": float(val_score),
            "test_score": float(test_score),
            "val": v_df.select(pred_cols).to_numpy(),
            "test": t_df.select(pred_cols).to_numpy(),
            "submit": s_df.select(pred_cols).to_numpy(),
            "submit_customer_id": s_df["customer_id"],
        })
        if len(candidates) >= args.top_k:
            break

    if len(candidates) < 3:
        raise RuntimeError("Not enough valid candidate experiments")

    names = [c["name"] for c in candidates]
    V = np.stack([c["val"] for c in candidates], axis=0)      # [M,N,T]
    T = np.stack([c["test"] for c in candidates], axis=0)     # [M,Nt,T]
    S = np.stack([c["submit"] for c in candidates], axis=0)   # [M,Ns,T]
    submit_customer_id = candidates[0]["submit_customer_id"]

    m = V.shape[0]
    # Rank-normalized tensors.
    Vr = np.stack([rank_norm_2d(V[i]) for i in range(m)], axis=0)
    Tr = np.stack([rank_norm_2d(T[i]) for i in range(m)], axis=0)
    Sr = np.stack([rank_norm_2d(S[i]) for i in range(m)], axis=0)

    # Baseline and val-only optimized rank-blend.
    w_best = np.full(m, 1.0 / m, dtype=np.float64)
    pred_best_val = np.tensordot(w_best, Vr, axes=(0, 0))
    best_val_auc = macro_auc(y_val, pred_best_val)

    rng = np.random.default_rng(20260309)
    for _ in range(args.rank_trials):
        w = rng.dirichlet(np.ones(m))
        pred_val = np.tensordot(w, Vr, axes=(0, 0))
        auc = macro_auc(y_val, pred_val)
        if auc > best_val_auc:
            best_val_auc = auc
            w_best = w
            pred_best_val = pred_val

    pred_rank_test = np.tensordot(w_best, Tr, axes=(0, 0))
    pred_rank_submit = np.tensordot(w_best, Sr, axes=(0, 0))
    rank_val_auc = macro_auc(y_val, pred_best_val)
    rank_test_auc = macro_auc(y_test, pred_rank_test)

    # Small CatBoost meta (val-trained).
    from catboost import CatBoostClassifier

    n_val = V.shape[1]
    n_test = T.shape[1]
    n_sub = S.shape[1]
    n_targets = V.shape[2]
    cat_val = np.zeros((n_val, n_targets), dtype=np.float64)
    cat_test = np.zeros((n_test, n_targets), dtype=np.float64)
    cat_sub = np.zeros((n_sub, n_targets), dtype=np.float64)

    for j in range(n_targets):
        x_val = V[:, :, j].T
        x_test = T[:, :, j].T
        x_sub = S[:, :, j].T
        y = y_val[:, j].astype(int)
        if np.unique(y).shape[0] < 2:
            cat_val[:, j] = x_val.mean(axis=1)
            cat_test[:, j] = x_test.mean(axis=1)
            cat_sub[:, j] = x_sub.mean(axis=1)
            continue
        model = CatBoostClassifier(
            iterations=args.cat_iterations,
            depth=args.cat_depth,
            learning_rate=0.05,
            l2_leaf_reg=6.0,
            loss_function="Logloss",
            verbose=False,
            thread_count=args.cat_thread_count,
            random_seed=20260309 + j,
        )
        model.fit(x_val, y)
        cat_val[:, j] = model.predict_proba(x_val)[:, 1]
        cat_test[:, j] = model.predict_proba(x_test)[:, 1]
        cat_sub[:, j] = model.predict_proba(x_sub)[:, 1]

    cat_val_auc = macro_auc(y_val, cat_val)
    cat_test_auc = macro_auc(y_test, cat_test)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rank_dir = exp_root / f"manual_rank_blend_{ts}"
    cat_dir = exp_root / f"manual_small_catboost_{ts}"
    for out_dir, pv, pt, ps in [
        (rank_dir, pred_best_val, pred_rank_test, pred_rank_submit),
        (cat_dir, cat_val, cat_test, cat_sub),
    ]:
        out = out_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        val_out = pl.DataFrame({"customer_id": val_y_df["customer_id"]})
        test_out = pl.DataFrame({"customer_id": test_y_df["customer_id"]})
        sub_out = pl.DataFrame({"customer_id": submit_customer_id})
        for j, c in enumerate(pred_cols):
            val_out = val_out.with_columns(pl.Series(c, pv[:, j]))
            test_out = test_out.with_columns(pl.Series(c, pt[:, j]))
            sub_out = sub_out.with_columns(pl.Series(c, ps[:, j]))
        val_out.write_parquet(out / "val_predictions.parquet")
        test_out.write_parquet(out / "test_predictions.parquet")
        sub_out.write_parquet(out / "submission.parquet")

    (rank_dir / "rank_blend_manifest.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "method": "rank_blend_val_optimized",
        "top_k": len(names),
        "rank_trials": args.rank_trials,
        "candidate_solutions": names,
        "weights": [float(x) for x in w_best.tolist()],
        "val_macro_roc_auc": rank_val_auc,
        "test_macro_roc_auc": rank_test_auc,
    }, indent=2))
    (rank_dir / "output" / "eval_results.json").write_text(json.dumps({
        "val_macro_roc_auc": round(rank_val_auc, 6),
        "test_macro_roc_auc": round(rank_test_auc, 6),
    }, indent=2))

    (cat_dir / "small_catboost_manifest.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "method": "small_catboost_meta_val_trained",
        "top_k": len(names),
        "cat_iterations": args.cat_iterations,
        "cat_depth": args.cat_depth,
        "candidate_solutions": names,
        "val_macro_roc_auc": cat_val_auc,
        "test_macro_roc_auc": cat_test_auc,
    }, indent=2))
    (cat_dir / "output" / "eval_results.json").write_text(json.dumps({
        "val_macro_roc_auc": round(cat_val_auc, 6),
        "test_macro_roc_auc": round(cat_test_auc, 6),
    }, indent=2))

    print(f"Candidates used: {len(names)}")
    print(f"RankBlend  val={rank_val_auc:.6f} test={rank_test_auc:.6f} dir={rank_dir}")
    print(f"CatBoost   val={cat_val_auc:.6f} test={cat_test_auc:.6f} dir={cat_dir}")


if __name__ == "__main__":
    main()
