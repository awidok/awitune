#!/usr/bin/env python3
"""
Generate train/val/test splits and OOF fold indices.

Scheme:
- Full train.parquet (750k samples) is split into 5 folds using MultilabelStratifiedKFold
- Fold 4: split into val and test using MultilabelStratifiedKFold (2 splits)
  - First half: val set (for early stopping in baseline)
  - Second half: test set (for final evaluation)

For baseline experiments:
- train = folds 0-3 (600k samples)
- val = fold4_val (75k samples)
- test = fold4_test (75k samples)

For OOF experiments (stacking):
- All 5 folds are used for OOF (full 750k samples)
- Each fold: train on 4/5, validate on 1/5
- This gives OOF predictions for ALL training data

Output: data/split_indices.json with all indices saved
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

SEED = 42
N_FOLDS = 5


def main():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    
    # Load full training data
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"train.parquet not found at {train_path}")
    
    print(f"Loading {train_path}...")
    df = pl.read_parquet(str(train_path))
    n_total = len(df)
    print(f"  Total samples: {n_total}")
    
    # Get target columns for stratification
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    print(f"  Target columns: {len(target_cols)}")
    
    y = df.select(target_cols).to_numpy()
    
    # Generate 5-fold split
    print(f"\nGenerating {N_FOLDS}-fold MultilabelStratifiedKFold split...")
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = list(mskf.split(np.zeros((n_total, 1)), y))
    except ImportError:
        print("WARNING: iterstrat not installed, falling back to KFold")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = list(kf.split(np.arange(n_total)))
    
    # Extract fold indices
    fold_indices = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_indices.append({
            "fold": fold_idx,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        print(f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
    
    # Fold 4 is reserved for val/test split
    # Split fold 4's val_idx into val and test using MultilabelStratifiedKFold
    fold4_val_idx = np.array(fold_indices[4]["val_idx"])
    fold4_y = y[fold4_val_idx]
    
    print(f"\nSplitting fold 4 into val/test...")
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        mskf2 = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        val_test_splits = list(mskf2.split(np.zeros((len(fold4_val_idx), 1)), fold4_y))
    except ImportError:
        from sklearn.model_selection import KFold
        kf2 = KFold(n_splits=2, shuffle=True, random_state=SEED)
        val_test_splits = list(kf2.split(np.arange(len(fold4_val_idx))))
    
    # 2-fold split: first half and second half
    # val_test_splits[0] = (first_half_idx, second_half_idx)
    val_sub_idx, test_sub_idx = val_test_splits[0]
    
    # Map back to original indices
    val_idx = fold4_val_idx[val_sub_idx].tolist()
    test_idx = fold4_val_idx[test_sub_idx].tolist()
    
    print(f"  Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Train indices = folds 0-3 train (all samples except fold 4 val)
    # Actually for baseline: train = all samples NOT in fold 4's val_idx
    fold4_all_val = set(fold_indices[4]["val_idx"])
    train_idx = [i for i in range(n_total) if i not in fold4_all_val]
    print(f"  Train (folds 0-3): {len(train_idx)}")
    
    # Verify no overlap
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    assert len(train_set & val_set) == 0, "Train/val overlap!"
    assert len(train_set & test_set) == 0, "Train/test overlap!"
    assert len(val_set & test_set) == 0, "Val/test overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == n_total, "Missing samples!"
    print("  ✓ No overlap between train/val/test")
    
    # Build output
    # For OOF fold 4: train = folds 0-3 (same as baseline), val = entire fold 4
    # This allows reusing baseline predictions for OOF fold 4
    oof_fold_4_train_idx = train_idx  # folds 0-3
    oof_fold_4_val_idx = fold_indices[4]["val_idx"]  # entire fold 4
    
    result = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_total": n_total,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "oof_folds": [
            {
                "fold": f["fold"],
                "train_idx": f["train_idx"],
                "val_idx": f["val_idx"],
            }
            for f in fold_indices[:4]  # Folds 0-3 use standard CV split
        ] + [
            {
                "fold": 4,
                "train_idx": oof_fold_4_train_idx,  # folds 0-3 (same as baseline)
                "val_idx": oof_fold_4_val_idx,      # entire fold 4
            }
        ],
        "holdout_fold": {
            "fold": 4,
            "val_idx": val_idx,
            "test_idx": test_idx,
        },
    }
    
    # Save
    output_path = data_dir / "split_indices.json"
    with open(output_path, "w") as f:
        json.dump(result, f)
    print(f"\nSaved to {output_path}")
    
    # Also create the split parquet files
    print("\nCreating split parquet files...")
    
    # Load all data once
    all_indices = np.arange(n_total)
    
    # Train split (folds 0-3)
    train_df = df.filter(pl.col("customer_id").is_in(
        df.select("customer_id").to_series()[train_idx].to_list()
    ))
    # Actually, simpler: use row indices directly
    train_df = df[train_idx]
    train_out = data_dir / "local_train.parquet"
    train_df.write_parquet(str(train_out))
    print(f"  {train_out}: {len(train_df)} rows")
    
    # Val split
    val_df = df[val_idx]
    val_out = data_dir / "local_val.parquet"
    val_df.write_parquet(str(val_out))
    print(f"  {val_out}: {len(val_df)} rows")
    
    # Test split
    test_df = df[test_idx]
    test_out = data_dir / "local_test.parquet"
    test_df.write_parquet(str(test_out))
    print(f"  {test_out}: {len(test_df)} rows")
    
    # Save targets separately for convenience
    target_cols_with_id = ["customer_id"] + target_cols
    
    train_target = train_df.select(target_cols_with_id)
    train_target.write_parquet(str(data_dir / "local_train_target.parquet"))
    
    val_target = val_df.select(target_cols_with_id)
    val_target.write_parquet(str(data_dir / "local_val_target.parquet"))
    
    test_target = test_df.select(target_cols_with_id)
    test_target.write_parquet(str(data_dir / "local_test_target.parquet"))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
