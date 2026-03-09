# Task: Improve ML Solution for Data Fusion 2026 Contest

## Context
You are an expert ML engineer competing in Data Fusion 2026 — Task 2 "Киберполка".
This is a **multi-label binary classification** problem: predict probability of opening each of 41 banking products for 750k customers.

**Metric**: Macro Averaged ROC-AUC across 41 targets.
**Current best val score**: {BEST_SCORE}

## Data Layout
Data is in `/app/data/` (READ-ONLY). **One parquet per split** (features + targets in one file):

- **Train**: `/app/data/local_train.parquet` — 600k rows (customer_id + 67 cat + 2373 num + 41 target_*)
- **Val**: `/app/data/local_val.parquet` — 75k rows (same schema)
- **Test**: `/app/data/local_test.parquet` — 75k rows (same schema)
- **Submit**: `/app/data/contest_test.parquet` — 250k rows (features only, no target_* columns)

Features: `cat_feature_1`..`cat_feature_67` (int32), `num_feature_1`..`num_feature_2373` (float64). Many NaN values.
Targets: 41 columns `target_1_1`..`target_10_1`. Strong class imbalance. Correlation within groups.

## Your Workspace
Your code is in `/app/workspace/`. It contains:
- `run.py` — the current best solution (MODIFY THIS)
- `prev_experiments.md` — summary of previous experiments and what worked/didn't
- `analyst_reports/` — data analysis reports from analyst agents (if any). READ THESE for insights on features, null patterns, correlations, etc.

## What You Must Do
1. **Read and understand** the current `run.py` thoroughly.
2. **Read `prev_experiments.md`** to understand what has been tried.
3. **Read `analyst_reports/`** if it exists — use data insights to guide your approach.
4. **Think about what improvement to try**. Good ideas:
   - Feature engineering (interactions, aggregations, target encoding, null patterns)
   - Architecture tweaks (attention, gating, different tower sizes)
   - Loss function improvements (focal loss, asymmetric loss, label smoothing)
   - Training tricks (SWA, mixup, learning rate schedules, warmup)
   - Better handling of NaN patterns
   - Feature selection / dimensionality reduction
5. **Implement the improvement** by modifying `run.py`.
6. **Run the training**:
   ```bash
   python run.py \
     --train /app/data/local_train \
     --val /app/data/local_val \
     --test /app/data/local_test \
     --submit /app/data/contest_test \
     /app/output
   ```
7. **Check the results** in `/app/output/metrics.json` and `/app/output/training_logs.json`.
8. **Verify ALL output files exist** before finishing (see Output Contract).

**IMPORTANT**: `run.py` must accept `--train`, `--val`, `--test`, `--submit` (data prefix paths).
The helper `load_split(prefix)` loads `{prefix}.parquet` and splits into features vs target_* columns. Contest test has no target columns.
Do NOT hardcode data paths — use the prefixes passed via arguments.

## Output Contract (MANDATORY — experiment is wasted without these)
Your `run.py` MUST produce ALL of these files in the output directory (passed as output_dir):
- `val_predictions.parquet` — predictions on local_val (columns: customer_id + predict_1_1..predict_10_1)
- `test_predictions.parquet` — predictions on local_test (same format)
- `submission.parquet` — predictions on contest test set (same format as sample_submit)
- `metrics.json` — must contain at least `{"val_macro_roc_auc": <float>}`
- `training_logs.json` — detailed training logs for analysis (see Training Logs Format below)
- `tb_logs/` — TensorBoard logs with loss/AUC curves

**If ANY file is missing, the experiment is a total waste.** Always verify all files exist after training.

## Training Logs Format
Your `run.py` must save `training_logs.json` with the following structure:
```json
{
  "model_a": {
    "name": "Model name",
    "logs": [{"epoch": 1, "train_loss": 0.15, "val_auc": 0.78, "lr": 1e-3, "time": 120}, ...],
    "final_auc": 0.82,
    "hparams": {...}
  },
  "model_b": {...},
  "ensemble": {
    "final_auc": 0.85,
    "per_target_auc": {"target_1_1": 0.83, ...},
    "best_weights": [[0.6, 0.4], ...],
    "weight_stats": {"mean_a": 0.55, "mean_b": 0.45, "std_a": 0.1, "std_b": 0.1}
  },
  "target_cols": ["target_1_1", ...],
  "n_features": 2440,
  "n_targets": 41,
  "train_samples": 600000,
  "val_samples": 75000,
  "elapsed_min": 150.5
}
```

## Report Guidelines
**YOU (the agent) must write `report.md`** by analyzing `metrics.json`, `training_logs.json`, and TensorBoard logs.
Do NOT generate report.md programmatically in run.py. Use **text tables** (not images). The report must include:

1. **Changes Made** — what you changed vs the baseline and why
2. **Training Log** — epoch-by-epoch table with columns: Epoch, Train Loss, Val Loss, Val AUC, LR, Time
3. **Per-Target AUC** — table with columns: Target, AUC, Support (num samples), Pos Rate (% positive class)
4. **Feature Importance (top-20)** — if applicable (gradient-based, permutation, or model-native importance)
5. **Observations** — what worked, what didn't, what you'd try next, any anomalies noticed

Example format:
```
# Experiment Report

## Changes Made
- Replaced BCE with Focal Loss (gamma=2.0, alpha=0.25)

## Training Log
| Epoch | Train Loss | Val Loss | Val AUC | LR     | Time   |
|-------|-----------|---------|---------|--------|--------|
| 1     | 0.1523    | 0.1498  | 0.7891  | 1.0e-3 | 2m 15s |
| 2     | 0.1401    | 0.1389  | 0.7945  | 9.5e-4 | 2m 12s |

## Per-Target AUC (val)
| Target     | AUC    | Support | Pos Rate |
|-----------|--------|---------|----------|
| target_1_1 | 0.8234 | 75000   | 12.3%    |
| target_1_2 | 0.7891 | 75000   | 3.1%     |

## Feature Importance (top-20)
| Rank | Feature        | Importance |
|------|---------------|-----------|
| 1    | num_feature_42 | 0.0234    |

## Observations
- Focal loss improved rare targets but slightly hurt common ones
- No overfitting observed
```

## Critical Rules

### Single-Run Policy (MANDATORY)
- In one orchestrator experiment, you must complete **at most one full training run** that reaches normal end and writes outputs.
- You may fix code/runtime errors and rerun **only until the first successful full run**.
- After the first successful full run:
  1. verify output files,
  2. write `report.md`,
  3. stop.
- Do **not** run additional "try another config", "one more training", "ablation", or "quick comparison" loops in the same experiment.
- The decision to start another experiment belongs to the orchestrator, not to this agent run.

### Time & Resource Constraints
- Training the baseline takes **~2-3 hours**. Do not make the architecture significantly heavier.
- You have **1 GPU**. Avoid dramatically increasing model dimensions or batch sizes.
- The data is READ-ONLY in `/app/data/`. Write outputs to `/app/output/`.

### GPU Utilization & Efficiency (MANDATORY)
- Prefer changes that improve **quality per GPU-hour**, not just absolute quality.
- Keep GPU busy during training: avoid obvious input pipeline stalls, unnecessary CPU bottlenecks, and long idle waits.
- **Do not rely on artifact reuse/caching shortcuts as the main strategy**. Improve runtime by writing more efficient code paths.
- Favor compute-efficient techniques that preserve or improve quality:
  - mixed precision (when stable),
  - efficient dataloading / preprocessing implementation,
  - batch sizing should prioritize quality/stability (small batches are acceptable when they improve generalization),
  - minimizing Python overhead in hot loops,
  - vectorized tensor/dataframe operations instead of slow row-wise logic,
  - avoiding redundant forward passes and repeated full-dataset evaluations.
- Optimize for **system throughput** on shared infrastructure:
  - avoid expensive CPU-bound preprocessing inside every epoch,
  - avoid unnecessary data copies/materialization,
  - keep memory footprint controlled to reduce contention.
- If two options are similar in expected quality, choose the one with lower runtime / memory footprint.

### Code Change Rules
- **MODIFY the existing `run.py`** — do NOT rewrite it from scratch.
- Keep the existing model architecture (DCNv2 backbone). Add to it, don't replace it.
- Focus on **one clear, incremental improvement**. Don't try to change everything at once.
- Keep your changes to ~50-150 lines of new/modified code. The less you change, the less can break.
- **NO stacking or blending** — do NOT implement stacking, blending, or ensemble averaging of multiple models.
  - The baseline already has an ensemble (2 models with per-target weights). Improve the models, not the ensembling.
  - Stacking/blending will be handled by a separate orchestration tool in the future.
  - Focus on: architecture improvements, loss functions, feature engineering, regularization, training tricks.

### Solution Diversity Strategy
The orchestrator is building a **portfolio of diverse solutions** for future stacking:
- Different solution families: DCNv2, Transformer, Tree-based (LightGBM/XGBoost), TabNet, MLP
- Each family should be improved independently to become competitive
- Your experiment should either:
  1. Improve the current solution family (incremental)
  2. Create a new solution with different architecture (diversification)
- The goal is to have 3-5 strong but DIFFERENT solutions that can be stacked later

### Output is Mandatory
- **Your code MUST produce ALL output files** (see Output Contract below).
- Before finishing, **verify files exist**: `ls -la /app/output/*.parquet /app/output/metrics.json /app/output/report.md`
- If training fails or you run out of time, **run the original unmodified `run.py`** as fallback to ensure output files are produced.

## Evaluation
After you finish, the orchestrator will:
1. Read `test_predictions.parquet` and compute Macro ROC-AUC against local_test labels.
2. If your score > {BEST_SCORE}, your solution becomes the new baseline.
3. `submission.parquet` will be kept for potential leaderboard submission.

Good luck! Focus on what will actually improve the score.
