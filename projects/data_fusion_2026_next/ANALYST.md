# Task: Analyze Dataset for Data Fusion 2026 Contest

## Context
You are a data scientist analyzing the dataset for a multi-label classification contest.
Your job is NOT to train models — it is to discover useful patterns, anomalies,
and insights that will help ML engineers build better solutions.

**Contest**: Data Fusion 2026 — Task 2 "Киберполка"
**Problem**: Multi-label binary classification — predict probability of opening each of 41 banking products for 750k customers.
**Metric**: Macro Averaged ROC-AUC across 41 targets.

## Data Layout
Data is in `/app/data/` (READ-ONLY):
- `local_train_main.parquet` — 600k × 200 (67 cat + 132 num + customer_id)
- `local_train_extra.parquet` — 600k × 2242 (2241 num + customer_id)
- `local_train_target.parquet` — 600k × 42 (41 targets + customer_id)
- `local_val_main.parquet` — 75k (same schema as train)
- `local_val_extra.parquet`
- `local_val_target.parquet`
- `local_test_main.parquet` — 75k (same schema as train)
- `local_test_extra.parquet`
- `local_test_target.parquet`
- `test_main_features.parquet` — 250k (contest test, no labels)
- `test_extra_features.parquet`
- `sample_submit.parquet`

Features:
- `cat_feature_1`..`cat_feature_67` — categorical (int32)
- `num_feature_1`..`num_feature_132` — numeric (float64) — main set
- `num_feature_133`..`num_feature_2373` — numeric (float64) — extra set
- Many NaN values and outliers

Targets: 41 binary targets grouped into 10 product groups. Strong class imbalance. Correlation within groups.

## Your Analysis Focus
{ANALYSIS_FOCUS}

## What You Must Do
1. Write Python analysis scripts in `/app/workspace/` and run them.
2. Explore the data systematically — load parquets with polars or pandas, compute statistics, find patterns.
3. Write a structured report to `/app/output/analysis_report.md`.
4. Save structured findings to `/app/output/analysis_data.json` (machine-readable).

## Report Structure
Your `/app/output/analysis_report.md` MUST follow this structure:

```
# Dataset Analysis: {title}

## Key Findings
- (bullet list of the most important, actionable insights)

## Detailed Analysis
(organized sections with tables, numbers, and explanations)

## Recommendations for Model Builders
- (concrete, specific suggestions that an ML engineer can act on)
- (include feature names, thresholds, code snippets where helpful)
```

## analysis_data.json Structure
Save machine-readable findings to `/app/output/analysis_data.json`:
```json
{
  "analysis_type": "description of what was analyzed",
  "findings": [
    {
      "category": "e.g. duplicate_columns / null_patterns / feature_correlations",
      "description": "human-readable description",
      "data": {}
    }
  ],
  "recommendations": [
    {
      "action": "e.g. drop_columns / add_feature / change_architecture",
      "details": "specific details",
      "expected_impact": "estimated impact on score"
    }
  ]
}
```

## Rules
- You have **1 GPU** available (use `cuda:0` if needed for heavy computation).
- Data is **READ-ONLY** in `/app/data/`. Write all output to `/app/output/`.
- Be **quantitative** — include exact numbers, not vague statements.
- If analysis scripts take too long, sample the data (e.g. 100k rows) and note it.
- Focus on **actionable** findings — things that directly help build better models.
- Use polars for data loading (faster than pandas for parquet).

## Previous Analysis
{PREVIOUS_ANALYSIS}
