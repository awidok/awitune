# Dataset Analysis: Predictive Power of Extra Features for Hard Targets

## Executive Summary

This analysis investigated the predictive power of extra features (num_feature_133 through num_feature_2373) for improving the hardest targets in the Data Fusion 2026 contest. A critical finding emerged: **the previously reported near-perfect correlation (r=0.946) between num_feature_624 and target_3_1 could not be reproduced**. This has major implications for model building strategies.

## Key Findings

### 1. CRITICAL DISCREPANCY: num_feature_624 Correlation

**Previous Report**: num_feature_624 shows r=0.946 with target_3_1, making it the most predictive feature for this hard target.

**Actual Finding**: On the full 750k dataset:
- **num_feature_624 has only 204 valid values** (99.97% NaN rate)
- **Correlation with target_3_1: r = -0.057** (p=0.42, not statistically significant)
- **Effect size (Cohen's d): -0.20** (small, negligible)

**Analysis of the 204 samples with valid num_feature_624**:
- For target_3_1=1: n=22, mean=-0.159, std=0.849
- For target_3_1=0: n=182, mean=0.044, std=1.134
- Positive rate in valid samples: 10.78% (vs 9.84% overall)

**Conclusion**: The feature is NOT predictive and should NOT be prioritized. The extreme sparsity (99.97% NaN) makes correlation estimates unreliable. The previously reported correlation was likely due to:
1. Sampling error with such sparse features
2. Different data preprocessing or filtering
3. Error in previous analysis

**Action**: Remove num_feature_624 from priority feature lists for target_3_1.

---

### 2. Actual Top Extra Features for Hard Targets

#### target_3_1 (AUC: 0.6351, Positive Rate: 9.84%)

Based on sampled analysis (100k rows), the actual top extra features are:

| Rank | Feature | Correlation | NaN Rate | Valid Count |
|------|---------|-------------|----------|-------------|
| 1 | num_feature_629 | 0.176 | 0.9973 | 1,956 |
| 2 | num_feature_1914 | 0.161 | 0.9961 | 2,982 |
| 3 | num_feature_2275 | 0.147 | 0.9965 | 2,625 |
| 4 | num_feature_1367 | 0.142 | 0.9968 | 2,400 |
| 5 | num_feature_634 | 0.138 | 0.9988 | 900 |

**Signal Strength**: Weak (max r = 0.18)
**Insight**: target_3_1 remains challenging. Extra features provide weak incremental signal over main features. Feature engineering and model architecture improvements needed.

---

#### target_9_6 (AUC: 0.6573, Positive Rate: 22.23%)

Top extra features:

| Rank | Feature | Correlation | NaN Rate | Valid Count |
|------|---------|-------------|----------|-------------|
| 1 | num_feature_2276 | 0.231 | 0.9986 | 1,050 |
| 2 | num_feature_175 | 0.210 | 0.9959 | 3,075 |
| 3 | num_feature_198 | 0.192 | 0.9980 | 1,500 |
| 4 | num_feature_835 | 0.157 | 0.9971 | 2,175 |
| 5 | num_feature_565 | 0.142 | 0.9986 | 1,050 |

**Signal Strength**: Weak-moderate (max r = 0.23)
**Insight**: Better signal than target_3_1 but still requires careful feature selection. Focus on features with lower NaN rates.

---

#### target_9_3 (AUC: 0.6583, Positive Rate: 1.90%)

Top extra features:

| Rank | Feature | Correlation | NaN Rate | Valid Count |
|------|---------|-------------|----------|-------------|
| 1 | num_feature_1907 | 0.215 | 0.9982 | 1,350 |
| 2 | num_feature_1596 | 0.189 | 0.9963 | 2,775 |
| 3 | num_feature_515 | 0.184 | 0.9988 | 900 |
| 4 | num_feature_478 | 0.154 | 0.9959 | 3,075 |
| 5 | num_feature_1875 | 0.136 | 0.9974 | 1,950 |

**Signal Strength**: Weak (max r = 0.22)
**Insight**: Extremely low positive rate (1.9%) combined with weak features makes this target critically dependent on:
- Class-balanced loss functions (pos_weight ≈ 52)
- Oversampling positives
- Calibration layers

---

#### target_2_4 (AUC: 0.7093, Positive Rate: 0.78%)

Top extra features:

| Rank | Feature | Correlation | NaN Rate | Valid Count |
|------|---------|-------------|----------|-------------|
| 1 | num_feature_217 | 0.323 | 0.9987 | 975 |
| 2 | num_feature_479 | 0.199 | 0.9981 | 1,425 |
| 3 | num_feature_607 | 0.142 | 0.9959 | 3,075 |
| 4 | num_feature_275 | 0.137 | 0.9989 | 825 |
| 5 | num_feature_1323 | 0.130 | 0.9889 | 8,325 |

**Signal Strength**: Moderate (max r = 0.32)
**Insight**: Best extra feature signal among hard targets. num_feature_217 shows promise with moderate correlation.

---

#### target_6_1 (AUC: 0.7010, Positive Rate: 0.94%)

Top extra features:

| Rank | Feature | Correlation | NaN Rate | Valid Count |
|------|---------|-------------|----------|-------------|
| 1 | num_feature_572 | 0.307 | 0.9943 | 4,275 |
| 2 | num_feature_634 | 0.261 | 0.9988 | 900 |
| 3 | num_feature_212 | 0.217 | 0.9962 | 2,850 |
| 4 | num_feature_1064 | 0.199 | 0.9882 | 8,850 |
| 5 | num_feature_1034 | 0.195 | 0.9989 | 825 |

**Signal Strength**: Moderate (max r = 0.31)
**Insight**: num_feature_572 shows good correlation with reasonable valid count (4,275 samples).

---

### 3. High-Correlation Features Across All Targets

Features with |r| > 0.3 with any target (from sampled analysis):

| Feature | Target | Correlation | NaN Rate |
|---------|--------|-------------|----------|
| num_feature_598 | target_1_5 | 0.892 | 0.9982 |
| num_feature_572 | target_6_5 | 0.818 | 0.9943 |
| num_feature_1064 | target_2_7 | 0.536 | 0.9882 |
| num_feature_479 | target_3_2 | 0.479 | 0.9981 |
| num_feature_1396 | target_1_1 | 0.353 | 0.9878 |
| num_feature_217 | target_2_4 | 0.323 | 0.9987 |

**Key Observations**:
1. **High correlations are associated with extreme sparsity** - most have >99% NaN rates
2. **Very few features exceed |r| > 0.3** - only 4 features found in sampled analysis
3. **High-correlation features tend to be target-specific** - not universally predictive

**Warning**: Features with >99% NaN rates should be used cautiously. Consider:
- NaN indicator features instead of raw values
- Target encoding or binning
- Only including in tree-based models with native NaN handling

---

### 4. Universal Feature Importance

Features appearing in top-20 across multiple targets:

| Rank | Feature | Appears in N Targets | Max Correlation | Avg Correlation |
|------|---------|----------------------|-----------------|-----------------|
| 1 | num_feature_634 | 12 | 0.261 | 0.078 |
| 2 | num_feature_1462 | 10 | 0.158 | 0.048 |
| 3 | num_feature_2232 | 9 | 0.133 | 0.038 |
| 4 | num_feature_2173 | 8 | 0.312 | 0.098 |
| 5 | num_feature_275 | 8 | 0.225 | 0.067 |
| 6 | num_feature_479 | 8 | 0.479 | 0.122 |
| 7 | num_feature_1367 | 8 | 0.194 | 0.059 |
| 8 | num_feature_2086 | 8 | 0.192 | 0.057 |

**Insight**: num_feature_634 and num_feature_479 are the most universally important extra features. Include these in baseline models.

---

### 5. NaN Pattern Analysis

Extra features have systematically higher NaN rates than main features:

- **Main features**: 58 features with >50% NaN (44% of 132 features)
- **Extra features**: 1,125 features with >50% NaN (50% of 2,241 features)
- **Extra features with 100% NaN**: 16 features (provide zero information, should be dropped)
- **Extra features with ≥99% NaN**: 287 features (provide minimal information)

**Strategy for High-NaN Features**:
1. **Drop features with 100% NaN** - zero information
2. **Create NaN indicator features** for high-NaN but predictive features (e.g., num_feature_634)
3. **For tree-based models** (LightGBM/XGBoost): use native NaN support
4. **For neural networks**: impute with sentinel value (-999) AND add NaN indicator

---

### 6. Comparison: Main vs Extra Features

For hard targets, extra features provide:

- **target_3_1**: Extra max r=0.18 vs Main max r=0.14 → 1.3× improvement (weak)
- **target_9_6**: Extra max r=0.23 vs Main max r=0.09 → 2.6× improvement (moderate)
- **target_9_3**: Extra max r=0.22 vs Main max r=0.07 → 3.1× improvement (moderate)
- **target_2_4**: Extra max r=0.32 vs Main max r=unknown → likely improvement
- **target_6_1**: Extra max r=0.31 vs Main max r=unknown → likely improvement

**Overall**: Extra features provide 1-3× improvement over main features for hard targets, but the absolute correlations remain weak (max 0.32).

---

## Detailed Analysis

### Data Quality Issues

#### Issue 1: Extreme Sparsity

Many extra features have extreme sparsity (>99% NaN):
- **num_feature_624**: 99.97% NaN (204 valid samples)
- **num_feature_1694**: 99.97% NaN (225 valid samples)
- **num_feature_217**: 99.87% NaN (975 valid samples)

**Problem**: Correlations calculated on very sparse features are unreliable and can appear high by chance. With only 200-1000 valid samples out of 750k, the correlation estimate has high variance.

**Solution**: For features with >99% NaN:
1. Use NaN indicator instead of raw value
2. Validate correlations on holdout set
3. Be skeptical of correlations > 0.5 on such features

---

#### Issue 2: Reproducibility Crisis

**Case Study**: num_feature_624

Previous analysis reported: r=0.946 with target_3_1
This analysis found: r=-0.057 with target_3_1

**Possible Causes**:
1. **Sampling error**: Previous analysis may have used a subset where spurious correlation appeared
2. **Data preprocessing**: Different NaN handling could affect results
3. **Data version**: Dataset may have been updated between analyses
4. **Computation error**: Bug in correlation calculation

**Verification Steps Taken**:
- Loaded full 750k dataset (not sampled)
- Computed Pearson correlation on valid (non-NaN) pairs only
- Verified with scipy.stats.pearsonr
- Checked both correlation and effect size (Cohen's d)

**Result**: The reported 0.946 correlation is **NOT REPRODUCIBLE** and appears to be an error.

**Impact**: Models built assuming num_feature_624 is highly predictive for target_3_1 will fail.

---

### Feature Engineering Opportunities

#### 1. NaN Indicator Features

For features with predictive NaN patterns, create binary indicators:

```python
high_nan_predictive_features = [
    'num_feature_634',   # appears in 12 targets' top-20
    'num_feature_479',   # max corr 0.48, appears in 8 targets
    'num_feature_217',   # corr 0.32 with target_2_4
    'num_feature_572',   # corr 0.31 with target_6_1
]

for feat in high_nan_predictive_features:
    df[f'{feat}_is_nan'] = df[feat].isna().astype(int)
```

**Expected Value**: NaN indicators can capture data availability patterns that are themselves predictive.

---

#### 2. Binning High-NaN Features

For features with >95% NaN but moderate correlation, bin into quantiles:

```python
def bin_sparse_feature(df, feature_name, n_bins=5):
    """Bin a sparse feature, treating NaN as its own bin"""
    valid_mask = ~df[feature_name].isna()

    # Create binned version for valid values
    df[f'{feature_name}_binned'] = pd.cut(
        df.loc[valid_mask, feature_name],
        bins=n_bins,
        labels=[f'bin_{i}' for i in range(n_bins)]
    )

    # NaN gets its own category
    df[f'{feature_name}_binned'] = df[f'{feature_name}_binned'].fillna('bin_nan')

    return df
```

**Expected Value**: Reduces noise from sparse features, makes correlations more robust.

---

#### 3. Feature Selection Strategy

Given that most extra features have weak correlations and high NaN rates, use a **two-stage selection**:

**Stage 1**: Filter by NaN rate
- Drop features with 100% NaN (16 features)
- Flag features with ≥99% NaN for careful validation (287 features)

**Stage 2**: Filter by correlation
- Keep features with |r| > 0.15 with any target
- For hard targets, relax threshold to |r| > 0.10

**Stage 3**: Remove redundancy
- For correlated feature groups (|r| > 0.9), keep the one with lower NaN rate

**Expected Result**: From 2,241 extra features, expect to keep:
- ~50-100 high-signal features
- ~200-300 moderate-signal features
- Total: ~250-400 extra features (reducing from 2,241)

---

### Model Architecture Implications

#### For Hard Targets

Given the weak extra feature signal for hard targets:

**target_3_1 (max r=0.18)**:
- Extra features alone insufficient
- Need:
  - Feature interactions (create product features)
  - Non-linear models (gradient boosting, neural networks)
  - Target encoding for categorical features
  - Calibration layer (calibration error = 0.39)

**target_9_3 (max r=0.22, positive rate=1.9%)**:
- Weak features + extreme imbalance
- Need:
  - Class-balanced loss (pos_weight=52)
  - Oversampling positives
  - Ensemble of diverse models
  - Calibration layer (calibration error = 0.41)

**target_9_6 (max r=0.23, positive rate=22.2%)**:
- Best among hard targets
- Reasonable positive rate
- Need:
  - Standard BCE loss acceptable
  - Include top 10-20 extra features
  - Feature engineering for boundary cases

---

### Expected Performance Impact

Based on findings, implementing these recommendations should yield:

| Target | Current AUC | Baseline with Extra Features | Optimistic Estimate |
|--------|-------------|------------------------------|---------------------|
| target_3_1 | 0.6351 | 0.68-0.72 | 0.75 (requires feature engineering) |
| target_9_6 | 0.6573 | 0.72-0.76 | 0.78 |
| target_9_3 | 0.6583 | 0.72-0.76 | 0.80 (requires weighted loss + calibration) |
| target_2_4 | 0.7093 | 0.75-0.78 | 0.80 |
| target_6_1 | 0.7010 | 0.74-0.77 | 0.78 |

**Overall Macro AUC Impact**:
- Current: ~0.80 (average across all 41 targets)
- Improvement from hard targets: +0.01-0.02
- Total: ~0.81-0.82

**Note**: The discrepancy with num_feature_624 means expected improvement for target_3_1 is much more modest than previously thought (+0.04-0.09 instead of +0.30).

---

## Recommendations for Model Builders

### Immediate Actions (Critical Priority)

1. **REMOVE num_feature_624 from priority feature list** - it is NOT predictive for target_3_1

2. **Validate all high-correlation features** on holdout set, especially:
   - Features with |r| > 0.5 (likely spurious due to sparsity)
   - Features with >95% NaN rate

3. **Include universal extra features in baseline**:
   - num_feature_634 (appears in 12 targets' top-20)
   - num_feature_479 (max corr=0.48, 8 targets)
   - num_feature_2173 (max corr=0.31, 8 targets)

4. **Create NaN indicator features** for:
   - num_feature_634
   - num_feature_479
   - num_feature_217
   - num_feature_572

5. **Apply class-balanced loss for target_9_3**:
   ```python
   pos_weight = 52.5  # inverse of 1.9% positive rate
   criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
   ```

---

### Data Preprocessing

```python
import polars as pl
import numpy as np

def preprocess_extra_features(df):
    """Preprocess extra features for modeling"""

    # Drop 100% NaN features
    features_to_drop = [
        # Add the 16 features with 100% NaN here
        # Identified from analysis
    ]

    # Create NaN indicators for high-importance sparse features
    sparse_important = ['num_feature_634', 'num_feature_479', 'num_feature_217', 'num_feature_572']

    for feat in sparse_important:
        if feat in df.columns:
            df = df.with_columns([
                pl.col(feat).is_null().cast(pl.Int32).alias(f'{feat}_is_nan')
            ])

    # For tree models: keep NaN as-is (LightGBM/XGBoost handle natively)
    # For neural networks: impute with sentinel
    # df = df.fill_nan(-999.0)  # Uncomment for NN

    return df
```

---

### Feature Selection

```python
def select_extra_features(df, target_correlations, threshold=0.15):
    """Select extra features based on correlation with any target"""

    # Get all extra features
    extra_features = [c for c in df.columns if c.startswith('num_feature_13')]

    # Compute max absolute correlation across all targets
    feature_max_corr = {}
    for feat in extra_features:
        if feat in target_correlations:
            max_corr = max(abs(corr) for corr in target_correlations[feat].values())
            feature_max_corr[feat] = max_corr

    # Select features above threshold
    selected_features = [
        feat for feat, max_corr in feature_max_corr.items()
        if max_corr >= threshold
    ]

    # Add NaN indicators for selected features
    nan_indicators = [f'{feat}_is_nan' for feat in selected_features if f'{feat}_is_nan' in df.columns]

    return selected_features + nan_indicators
```

---

### Model Training Strategy

For hard targets, use a **target-specific approach**:

**target_3_1**:
```python
# Feature engineering is critical
- Create polynomial features (degree 2)
- Target encode categorical features
- Use gradient boosting (LightGBM) or neural network with 2-3 hidden layers
- Add calibration layer (temperature scaling)
```

**target_9_3**:
```python
# Handle extreme imbalance
- Use weighted BCE (pos_weight=52.5)
- OR: focal loss (gamma=2.0, alpha=0.98)
- Oversample positives (SMOTE or duplication)
- Use smaller model (fewer parameters, higher dropout=0.5)
- Ensemble multiple models
- Calibration layer mandatory
```

**target_9_6**:
```python
# Moderate approach
- Standard BCE loss acceptable
- Include top 10-20 extra features
- Focus on samples near decision boundary
- Consider focal loss with gamma=1.5
- Calibration layer recommended
```

---

### Validation and Monitoring

**Critical**: Monitor these metrics during training:

1. **ROC-AUC** (primary metric)
2. **Calibration error** (Brier score or calibration curve)
3. **Precision-Recall AUC** (especially for target_9_3 with 1.9% positive rate)
4. **F1-score** at optimal threshold

```python
from sklearn.metrics import roc_auc_score, brier_score_loss, calibration_curve
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred_proba, target_name):
    """Comprehensive evaluation including calibration"""

    auc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    print(f"\n{target_name}:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Calibration Error: {calibration_error:.4f}")
    print(f"  Brier Score: {brier:.4f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, marker='o', label=target_name)
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve: {target_name}')
    plt.legend()
    plt.savefig(f'calibration_{target_name}.png')
    plt.close()

    return {'auc': auc, 'calibration_error': calibration_error, 'brier': brier}
```

---

## Analysis Data

Full analysis data saved to:
- `/app/output/comprehensive_analysis_data.json` - All findings in machine-readable format
- `/app/output/top_features_per_target_full.csv` - Top 10 features for each target
- `/app/output/feature_624_verification.json` - Detailed investigation of num_feature_624 discrepancy

---

## Conclusion

This analysis revealed a critical discrepancy in previous findings: num_feature_624 is NOT highly predictive for target_3_1 as previously reported (r=-0.057, not r=0.946). This changes the strategy for improving hard targets significantly.

**Key Takeaways**:

1. **Extra features provide weak-moderate signal** for hard targets (max correlations 0.18-0.32, not 0.95)

2. **Feature sparsity is extreme** - most useful extra features have >99% NaN rates, making them unreliable

3. **Model improvement will be incremental**, not transformational:
   - target_3_1: +0.04-0.09 AUC (not +0.30)
   - Other hard targets: +0.06-0.12 AUC

4. **Success requires a multi-pronged approach**:
   - Careful feature selection (not just adding all extra features)
   - Target-specific loss functions (especially for target_9_3)
   - Feature engineering (interactions, binning, NaN indicators)
   - Calibration layers for probability estimates

5. **Validate all findings** - correlations in high-NaN features are unstable and may not generalize

**Next Steps**:

1. Implement baseline model with top 50-100 extra features
2. Add calibration layers for all targets
3. Apply class-balanced loss for target_9_3
4. Engineer feature interactions for target_3_1
5. Ensemble models for hard targets

**Expected Competition Impact**: Modest improvement from 0.80 to 0.81-0.82 macro AUC, primarily from better calibration and class balancing rather than feature discovery.

---

## Technical Notes

- **Analysis performed on**: Full 750k training dataset
- **Sampling used**: 100k samples for initial correlation analysis (when full dataset computation was too slow)
- **Libraries**: polars 1.0+, numpy, scipy, pandas
- **Computation time**: ~2 hours for full analysis (correlation of 2,241 features × 41 targets on 750k samples)
- **Validation**: All critical findings verified on full dataset

**Reproducibility**: All scripts and data available in `/app/workspace/` and `/app/output/`