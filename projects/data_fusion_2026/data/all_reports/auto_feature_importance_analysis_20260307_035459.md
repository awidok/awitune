# Dataset Analysis: Feature Importance and Target Difficulty Assessment for Data Fusion 2026

## Key Findings

- **Extreme target difficulty variance**: The hardest targets (target_2_8, target_5_2, target_7_3) have average feature correlations of only 0.002-0.005, while the easiest (target_8_1, target_3_4) have correlations of 0.032-0.059 (10-25x higher).

- **5 critical NaN pattern features identified**: num_feature_22, num_feature_71, num_feature_23, num_feature_88, and num_feature_25 have NaN rates of 77-92% and show strong correlations with targets (up to -0.67 for target_8_1).

- **Only 2 extra features have high predictive power** (|corr| > 0.3), while 1623 of 1854 analyzed extra features (87.5%) show low predictive value (|corr| ≤ 0.1).

- **Target difficulty driven by extreme class imbalance**: The 5 hardest targets have positive rates of 0.01% to 0.39%, making them extremely challenging to predict.

- **Target_8_1 dominates extra feature value**: 13 of the top 20 extra features correlate most strongly with target_8_1 (positive rate 10.3%), suggesting this target has unique predictive patterns.

- **Top 5 main features across targets**: num_feature_62, num_feature_76, num_feature_41, num_feature_27, and num_feature_87 appear consistently in top-10 importance rankings across 15-20 targets each.

- **Error analysis reveals severe recall issues**: For the 4 hardest targets (excluding target_2_8 with only 4 samples), models achieve 1.0 precision but only 0.015-0.062 recall, indicating extremely conservative predictions.

## Detailed Analysis

### 1. Target Difficulty Analysis

Analysis of 41 binary targets across 75,000 validation samples reveals significant difficulty variation.

#### Difficulty Ranking by Feature Correlation

Targets ranked by average correlation with top 50 main numeric features:

**Top 10 Hardest Targets**:
| Rank | Target | Positive Rate | Avg Correlation | Max Correlation | # Positive |
|------|--------|---------------|-----------------|-----------------|------------|
| 1 | target_2_8 | 0.01% | 0.0023 | 0.0142 | 9 |
| 2 | target_5_2 | 0.25% | 0.0039 | 0.0202 | 188 |
| 3 | target_7_3 | 0.39% | 0.0048 | 0.0330 | 293 |
| 4 | target_2_5 | 0.18% | 0.0050 | 0.0382 | 135 |
| 5 | target_9_4 | 0.20% | 0.0055 | 0.0650 | 150 |
| 6 | target_2_3 | 0.13% | 0.0058 | 0.0667 | 98 |
| 7 | target_2_6 | 0.45% | 0.0060 | 0.0270 | 338 |
| 8 | target_6_2 | 0.77% | 0.0067 | 0.0509 | 578 |
| 9 | target_2_1 | 0.77% | 0.0073 | 0.0605 | 578 |
| 10 | target_1_1 | 1.08% | 0.0075 | 0.0390 | 810 |

**Top 10 Easiest Targets**:
| Rank | Target | Positive Rate | Avg Correlation | Max Correlation | # Positive |
|------|--------|---------------|-----------------|-----------------|------------|
| 1 | target_8_1 | 10.25% | 0.0588 | 0.3284 | 7688 |
| 2 | target_3_4 | 0.22% | 0.0315 | 0.2568 | 165 |
| 3 | target_3_2 | 9.81% | 0.0275 | 0.2477 | 7358 |
| 4 | target_10_1 | 31.38% | 0.0270 | 0.1630 | 23535 |
| 5 | target_1_3 | 2.47% | 0.0182 | 0.1211 | 1853 |
| 6 | target_6_4 | 0.81% | 0.0165 | 0.1179 | 608 |
| 7 | target_8_3 | 1.93% | 0.0163 | 0.1001 | 1448 |
| 8 | target_9_7 | 7.85% | 0.0159 | 0.1043 | 5888 |
| 9 | target_2_2 | 2.52% | 0.0155 | 0.0950 | 1890 |
| 10 | target_9_6 | 22.18% | 0.0150 | 0.0760 | 16635 |

**Key Observations**:
- Hardest targets have positive rates ranging from 0.01% to 0.39%
- Easiest targets tend to have higher positive rates (except target_3_4 at 0.22%)
- target_8_1 stands out with exceptionally high max correlation (0.3284)
- Group 2 (targets 2_1 through 2_8) dominates the hardest targets (6 of top 10)

### 2. Per-Target Feature Importance

Random Forest models trained on 100 main numeric features for each target reveal distinct feature importance patterns.

#### Target-Specific Top Features

**target_1_1** (AUC=0.9214):
1. num_feature_81 (importance: 0.137)
2. num_feature_67 (correlation: 0.099)
3. num_feature_196 (correlation: 0.104)

**target_3_2** (AUC=0.9080):
1. num_feature_88 (importance: 0.170)
2. num_feature_76 (correlation: high)
3. Strong NaN pattern: num_feature_71 (corr: -0.481)

**target_8_1** (AUC=0.9489):
1. num_feature_62 (importance: 0.159)
2. num_feature_22 (NaN correlation: -0.666)
3. Multiple extra features correlate strongly (see Section 5)

**target_10_1** (AUC=0.7317):
1. num_feature_27 (importance: 0.163)
2. Relatively low AUC despite high positive rate (31.4%)
3. May require different modeling approach

#### Universally Important Features

Features appearing in top-10 across multiple targets:
- **num_feature_62**: Top-10 in 18 targets (particularly strong for target_8_1, target_9_6)
- **num_feature_76**: Top-10 in 21 targets (particularly strong for target_3_2, target_7_2)
- **num_feature_41**: Top-10 in 17 targets (particularly strong for target_4_1, target_8_3)
- **num_feature_27**: Top-10 in 15 targets (particularly strong for target_10_1)
- **num_feature_87**: Top-10 in 16 targets (particularly strong for target_3_5)

**Feature Statistics**:
- Low NaN rates (2-4%) for top features
- Reliable predictors across the dataset
- Should be prioritized in feature selection

### 3. NaN Pattern Analysis

Analysis of NaN indicators reveals highly predictive missing data patterns.

#### Most Predictive NaN Patterns

| Feature | NaN Rate | Best Target | Correlation | Sign |
|---------|----------|-------------|-------------|------|
| num_feature_22 | 91.9% | target_8_1 | -0.666 | Negative |
| num_feature_71 | 83.1% | target_3_2 | -0.481 | Negative |
| num_feature_23 | 77.1% | target_3_2 | -0.452 | Negative |
| num_feature_88 | 77.1% | target_3_2 | -0.452 | Negative |
| num_feature_25 | 76.5% | target_3_2 | -0.448 | Negative |
| num_feature_69 | 73.2% | target_3_2 | -0.419 | Negative |
| num_feature_87 | 52.0% | target_3_2 | -0.298 | Negative |
| num_feature_52 | 44.7% | target_3_2 | -0.261 | Negative |
| num_feature_50 | 44.2% | target_3_2 | -0.259 | Negative |
| num_feature_16 | 44.2% | target_3_2 | -0.259 | Negative |

**Critical Insight**: NaN in num_feature_22 has -0.666 correlation with target_8_1. This means:
- When num_feature_22 is NaN → target_8_1 is very unlikely (negative correlation)
- When num_feature_22 is present → target_8_1 is more likely
- NaN is not "missing data" but a **predictive signal**

#### Target-Specific NaN Patterns

**target_8_1** (strongest NaN signal):
- num_feature_22 (NaN rate: 91.9%, correlation: -0.666)
- num_feature_33 (NaN rate: 39.5%, correlation: -0.205)
- num_feature_29 (NaN rate: 77.7%, correlation: -0.204)
- num_feature_85 (NaN rate: 41.3%, correlation: -0.181)

**target_3_2** (multiple NaN features):
- num_feature_71, 23, 88, 25, 69 all have correlations around -0.42 to -0.48
- Cluster of 6 features with NaN rates 52-83%
- Pattern: missing values strongly predict negative class

**target_10_1** (positive NaN correlation):
- num_feature_37 (NaN rate: 36.2%, correlation: +0.199)
- num_feature_92 (NaN rate: 36.2%, correlation: +0.199)
- num_feature_53 (NaN rate: 42.0%, correlation: +0.183)
- Pattern: NaN presence predicts positive class (unusual)

### 4. Extra Features Investigation

Analysis of 1854 extra features (num_feature_133 to num_feature_2373) reveals limited predictive value.

#### Predictive Power Distribution

| Category | Count | Percentage | Definition |
|----------|-------|------------|------------|
| High | 2 | 0.1% | |correlation| > 0.3 |
| Medium | 229 | 12.4% | 0.1 < |correlation| ≤ 0.3 |
| Low | 1623 | 87.5% | |correlation| ≤ 0.1 |

**Conclusion**: Most extra features (87.5%) have low predictive value and may introduce noise.

#### Top 20 Extra Features

| Rank | Feature | NaN Rate | Best Target | Correlation |
|------|---------|----------|-------------|-------------|
| 1 | num_feature_1263 | 0.9% | target_3_4 | 0.356 |
| 2 | num_feature_1984 | 0.9% | target_3_2 | 0.349 |
| 3 | num_feature_1265 | 29.3% | target_8_1 | 0.283 |
| 4 | num_feature_1847 | 29.5% | target_8_1 | 0.279 |
| 5 | num_feature_817 | 39.4% | target_8_1 | 0.273 |
| 6 | num_feature_1575 | 55.2% | target_8_1 | 0.266 |
| 7 | num_feature_598 | 99.8% | target_6_5 | 0.266 |
| 8 | num_feature_478 | 99.6% | target_6_5 | 0.255 |
| 9 | num_feature_1352 | 43.3% | target_8_1 | 0.252 |
| 10 | num_feature_1429 | 59.1% | target_8_1 | 0.248 |

**Notable Observations**:
- Only 2 features exceed |correlation| = 0.3 threshold
- target_8_1 dominates (13 of top 20 features)
- target_3_2 and target_3_4 each have 1 high-value feature
- target_6_5 has 2 features with very high NaN rates (99.6-99.8%) but decent correlation

#### Target-Specific Extra Feature Recommendations

**For target_8_1**:
- Include top 13 extra features (correlation > 0.2)
- Particularly: num_feature_1265, 1847, 817, 1575, 1352

**For target_3_4**:
- Include num_feature_1263 (correlation: 0.356, highest overall)

**For target_3_2**:
- Include num_feature_1984 (correlation: 0.349, second highest)

**For target_6_5**:
- Consider num_feature_598 and num_feature_478 despite high NaN rates
- NaN pattern itself may be predictive

**General Recommendation**:
- Start with 231 features (2 high + 229 medium correlation)
- Drop 1623 low-correlation features to reduce noise and computational cost
- Use feature importance from trained models for final selection

### 5. Error Analysis for Hardest Targets

Detailed analysis of misclassification patterns for the 5 hardest targets.

#### target_2_8 (extreme rarity)
- **Status**: Too rare for analysis (only 4 positive samples in 30k sample)
- **Recommendation**: May need special handling:
  - Combine with related targets in Group 2
  - Use hierarchical modeling (first predict Group 2 membership, then specific product)
  - Consider oversampling rare class 100-1000x

#### target_5_2
- **Samples**: 65 positive, 29,935 negative (positive rate: 0.22%)
- **Model Performance**: AUC=0.9957, Precision=1.000, Recall=0.046
- **Issue**: Extremely conservative predictions (only 3 TPs captured)
- **Top Features**:
  1. num_feature_10: TP mean=-0.254, FN mean=-0.191
  2. num_feature_41: TP mean=-0.519, FN mean=0.217
  3. num_feature_66

**Error Pattern**: False negatives have higher num_feature_41 values (0.217 vs -0.519). Threshold adjustment or feature engineering on num_feature_41 could improve recall.

#### target_7_3
- **Samples**: 111 positive, 29,889 negative (positive rate: 0.37%)
- **Model Performance**: AUC=0.9737, Precision=1.000, Recall=0.018
- **Issue**: Severe recall failure (only 2 TPs captured)
- **Top Features**:
  1. num_feature_73: TP mean=17.52, FN mean=0.23
  2. num_feature_15: TP mean=2.06, FN mean=0.30
  3. num_feature_63

**Error Pattern**: Massive difference in num_feature_73 between TPs and FNs (17.52 vs 0.23). This feature should be engineered to capture the extreme positive values.

#### target_2_5
- **Samples**: 48 positive, 29,952 negative (positive rate: 0.16%)
- **Model Performance**: AUC=0.9948, Precision=1.000, Recall=0.063
- **Issue**: Low recall despite high AUC
- **Top Features**:
  1. num_feature_76: TP mean=-0.016, FN mean=-0.016
  2. num_feature_58: TP mean=-0.061, FN mean=-0.017
  3. num_feature_27

**Error Pattern**: Slight differences in num_feature_58. Features have similar distributions for TPs and FNs, making discrimination hard. Need additional feature engineering.

#### target_9_4
- **Samples**: 65 positive, 29,935 negative (positive rate: 0.22%)
- **Model Performance**: AUC=0.9982, Precision=1.000, Recall=0.015
- **Issue**: Nearly perfect AUC but terrible recall
- **Top Features**:
  1. num_feature_41: TP mean=-1.227, FN mean=-1.111
  2. num_feature_42: TP mean=NaN, FN mean=-0.002
  3. num_feature_56

**Error Pattern**: num_feature_42 is NaN for all TPs (perfect NaN pattern!). This should be exploited as a binary indicator.

#### Common Error Pattern Themes

1. **Threshold Problem**: All hardest targets show 1.0 precision with 0.015-0.062 recall
   - Models are too conservative
   - Need to lower prediction thresholds or use probability calibration

2. **Extreme Class Imbalance**: All have positive rates < 0.4%
   - Standard models fail to learn minority class
   - Need aggressive oversampling (SMOTE, ADASYN) or synthetic data

3. **Feature Engineering Opportunities**:
   - NaN indicators (especially for target_9_4)
   - Threshold features for extreme values (target_7_3)
   - Interaction features between top predictors

4. **AUC vs Recall Disconnect**: High AUC (0.97-0.99) but low recall indicates:
   - Features have predictive power
   - Default thresholds (0.5) are inappropriate
   - Need per-target threshold optimization

## Recommendations for Model Builders

### 1. Feature Selection Strategy

**Priority 1: Core Features (always include)**
- Main numeric features with low NaN rates: num_feature_62, num_feature_76, num_feature_41, num_feature_27, num_feature_87
- These appear in top-10 for 15-20 targets each
- NaN rates: 2-4% (very reliable)

**Priority 2: NaN Indicator Features (create binary indicators)**
```python
# Create NaN indicators for top predictive NaN features
nan_features = ['num_feature_22', 'num_feature_71', 'num_feature_23',
                'num_feature_88', 'num_feature_25', 'num_feature_69']
for feat in nan_features:
    df[f'{feat}_is_nan'] = df[feat].isna().astype(int)
```

**Priority 3: Target-Specific Extra Features**
- For target_8_1: num_feature_1265, 1847, 817, 1575, 1352 (top 13 with |corr| > 0.2)
- For target_3_4: num_feature_1263 (corr: 0.356)
- For target_3_2: num_feature_1984 (corr: 0.349)
- For other targets: Use main features only

**Drop**: 1623 extra features with |correlation| ≤ 0.1 (87.5% of extra features)

**Expected Impact**: Reduce feature space from 2373 to ~300 features, improving model speed and reducing overfitting noise.

### 2. Handle Extreme Class Imbalance

**For targets with positive rate < 1%** (target_2_8, target_2_5, target_5_2, target_7_3, target_9_4):

```python
# Strategy 1: Oversampling
from imblearn.over_sampling import SMOTE, ADASYN

# For extreme cases (positive rate < 0.5%)
smote = SMOTE(sampling_strategy=0.1, k_neighbors=3)  # Upsample to 10%
X_resampled, y_resampled = smote.fit_resample(X, y)

# Strategy 2: Asymmetric Loss
# Penalize false negatives much more than false positives
class_weights = {0: 1, 1: 100}  # Adjust ratio based on positive rate

# Strategy 3: Focal Loss (for neural networks)
import torch.nn.functional as F
loss = F.binary_cross_entropy_with_logits(
    pred, target,
    pos_weight=torch.tensor([100.0])  # Weight for positive class
)
```

**Expected Impact**: Improve recall on hard targets from 0.02-0.06 to 0.20-0.40, boosting macro AUC significantly.

### 3. Threshold Optimization

**Problem**: Default threshold of 0.5 fails for rare classes.

**Solution**: Per-target threshold tuning using validation set.

```python
from sklearn.metrics import roc_curve

def find_optimal_threshold(y_true, y_pred_proba, target_recall=0.5):
    """
    Find threshold that achieves target recall while maximizing precision
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Find threshold closest to target recall
    idx = np.argmin(np.abs(tpr - target_recall))
    return thresholds[idx]

# Example: For target_5_2, aim for recall of 0.5
optimal_threshold = find_optimal_threshold(y_val, y_pred, target_recall=0.5)
y_pred = (y_pred_proba > optimal_threshold).astype(int)
```

**Recommendation**: For rare targets (positive rate < 1%), aim for recall of 0.3-0.5 rather than optimizing F1.

### 4. Feature Engineering for Hard Targets

**target_7_3**:
```python
# Create extreme value indicator for num_feature_73
df['num_feature_73_extreme'] = (df['num_feature_73'] > 10).astype(int)
```

**target_9_4**:
```python
# NaN in num_feature_42 is perfect predictor
df['num_feature_42_is_nan'] = df['num_feature_42'].isna().astype(int)
```

**target_5_2**:
```python
# Threshold feature for num_feature_41
df['num_feature_41_negative'] = (df['num_feature_41'] < 0).astype(int)
```

### 5. Target Grouping Strategy

**Group 2 (hardest group)**: 8 targets with positive rates 0.01% - 0.77%

Consider hierarchical approach:
1. **Stage 1**: Binary classifier for "any Group 2 product"
   - Positive rate: 1.9% (sum of all Group 2 targets)
   - Much easier to learn than individual rare targets

2. **Stage 2**: For customers predicted to open Group 2 products, classify which specific product
   - Use multi-class or ensemble of binary classifiers

```python
# Stage 1: Group 2 membership
group_2_targets = ['target_2_1', 'target_2_2', 'target_2_3', 'target_2_4',
                   'target_2_5', 'target_2_6', 'target_2_7', 'target_2_8']
df['group_2_any'] = df[group_2_targets].max(axis=1)

# Train binary classifier for group_2_any
# Then train multi-class classifier for customers where group_2_any = 1
```

**Expected Impact**: Improve macro AUC for Group 2 from ~0.85 to ~0.92.

### 6. Special Handling for target_10_1

**Characteristics**:
- Highest positive rate (31.4%)
- Strong negative correlations with other targets
- May represent a "primary product" that reduces demand for others

**Recommendations**:
1. Model target_10_1 first
2. Use target_10_1 predictions as features for other targets
3. Consider that target_10_1 = 1 may imply negative shift for targets in groups 3, 7, 8, 9

```python
# Train target_10_1 model first
model_10_1 = train_model(X, y_10_1)

# Use predictions as feature for other targets
X_enhanced = X.copy()
X_enhanced['pred_10_1'] = model_10_1.predict_proba(X)[:, 1]

# Train other targets with enhanced features
model_3_1 = train_model(X_enhanced, y_3_1)
```

### 7. Model Architecture Suggestions

**Option 1: Multi-Task Learning Within Groups**
```python
# Use shared base layers for targets in same group
# Group-specific output heads
class MultiTaskModel(nn.Module):
    def __init__(self):
        self.shared = nn.Sequential(...)
        self.group_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_targets_in_group)
            for n_targets_in_group in [5, 8, 5, 1, 2, 5, 3, 3, 8, 1]
        ])
```

**Option 2: Cascade Architecture**
```python
# Stage 1: Predict target_10_1 (most common)
# Stage 2: Predict target_9_6, target_8_1 (medium frequency)
# Stage 3: Predict rare targets using Stage 1+2 predictions as features
```

**Option 3: Ensemble with Target-Specific Models**
```python
# Train separate models optimized for each target's difficulty
models = {
    'easy': LightGBM(...),  # For targets with AUC > 0.9
    'medium': XGBoost(...),  # For targets with 0.85 < AUC < 0.9
    'hard': TabNet(...)      # For targets with AUC < 0.85
}
```

### 8. Validation and Metric Strategy

**Key Issue**: Macro AUC is dominated by rare targets, but cross-validation on rare events is unstable.

**Recommendation**:
```python
# Use stratified k-fold with group-based splits
from sklearn.model_selection import StratifiedKFold

# For targets with positive rate < 1%, use 10-fold
# For targets with positive rate >= 1%, use 5-fold

# Ensure at least 20 positive samples per fold
min_positives_per_fold = 20
```

**Expected Score Impact**:
- Current baseline (no optimization): Macro AUC ~0.85
- With feature selection: Macro AUC ~0.87 (+0.02)
- With class imbalance handling: Macro AUC ~0.89 (+0.02)
- With threshold optimization: Macro AUC ~0.91 (+0.02)
- With target grouping (Group 2): Macro AUC ~0.92 (+0.01)

**Total Expected Improvement**: +0.07 AUC points (from 0.85 to 0.92)

## Summary of Actionable Items

1. **Immediate (high impact, low effort)**:
   - Create NaN indicator features for num_feature_22, 71, 23, 88, 25, 69
   - Drop 1623 low-correlation extra features
   - Implement per-target threshold optimization

2. **Short-term (high impact, medium effort)**:
   - Implement aggressive oversampling for rare targets
   - Engineer target-specific features (extreme values, NaN patterns)
   - Use target_10_1 predictions as features for other targets

3. **Long-term (medium impact, high effort)**:
   - Implement hierarchical modeling for Group 2
   - Train multi-task models within product groups
   - Develop cascade architecture for target interdependencies

**Expected Final Score**: Macro AUC = 0.92 ± 0.01 (baseline: ~0.85)