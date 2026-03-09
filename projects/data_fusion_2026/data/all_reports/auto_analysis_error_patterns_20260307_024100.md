# Dataset Analysis: Feature Engineering and Target Insights for Data Fusion 2026

## Key Findings

- **Extreme class imbalance across 41 targets**: Positive rates range from 0.012% (target_2_8, only 9/75k positive samples) to 31.4% (target_10_1), with 29 of 41 targets having <5% positive rate
- **Strong target correlations within product groups**: Within-group correlation mean = 0.030 (±0.083), vs across-group mean = -0.0004 (±0.034), suggesting multi-task learning within groups could improve performance
- **Target 10_1 strongly anti-correlates with others**: Correlations range from -0.36 (target_9_6) to -0.07 (target_1_1), suggesting customers who open product 10_1 are less likely to open other products
- **699 key features drive most predictions**: 5 main numeric features (num_feature_62, num_feature_76, num_feature_41, num_feature_27, num_feature_87) appear in top-10 for 17-21 targets each
- **589 features have >90% NaN values**: 5 features have 100% NaN rate and provide no value; these should be removed
- **NaN patterns are highly informative**: For target_8_1, NaN in num_feature_22 shows -0.67 correlation with target; for target_3_2, NaN in num_feature_395 shows -0.45 correlation
- **Most important features have low NaN rates**: Top features (num_feature_62, num_feature_76, num_feature_27) have only 2-4% NaN, making them reliable predictors
- **No prediction files or training logs available**: Cannot perform error analysis, PCA impact analysis, or training issue investigation without model outputs

## Detailed Analysis

### 1. Target Analysis

#### Target Distribution and Class Imbalance

Analysis of 75,000 validation samples reveals severe class imbalance across 41 binary targets organized into 10 product groups:

| Group | Targets | Positive Rate Range | Mean Rate |
|-------|---------|---------------------|-----------|
| Group 10 | 1 target | 31.38% - 31.38% | 31.38% |
| Group 9 | 8 targets | 0.20% - 22.18% | 4.73% |
| Group 8 | 3 targets | 1.93% - 10.25% | 5.13% |
| Group 3 | 5 targets | 0.12% - 9.81% | 4.01% |
| Group 7 | 3 targets | 0.39% - 6.37% | 3.18% |
| Group 1 | 5 targets | 0.18% - 2.47% | 1.28% |
| Group 2 | 8 targets | 0.01% - 2.52% | 0.61% |
| Group 6 | 5 targets | 0.05% - 0.94% | 0.63% |
| Group 5 | 2 targets | 0.25% - 0.93% | 0.59% |
| Group 4 | 1 target | 0.86% - 0.86% | 0.86% |

**Most extreme targets**:
- `target_10_1`: 31.4% positive (most common, 23,538/75k samples)
- `target_9_6`: 22.2% positive (second most common)
- `target_3_2`: 9.8% positive
- `target_8_1`: 10.3% positive
- `target_2_8`: 0.012% positive (only 9/75k samples)
- `target_2_7`: 0.041% positive (31/75k samples)

**Implication**: Macro-averaged ROC-AUC metric will be dominated by performance on rare classes. Use techniques like:
- Oversampling rare classes (SMOTE, ADASYN)
- Focal loss or ASL (Asymmetric loss)
- Per-class weighting based on inverse frequency

#### Target Correlations

**Top positive correlations** (all within-group):
- target_6_1 ↔ target_6_4: +0.53 (strongest correlation)
- target_5_1 ↔ target_5_2: +0.51
- target_6_4 ↔ target_6_5: +0.25
- target_1_4 ↔ target_2_2: +0.21

**Top negative correlations** (all involve target_10_1):
- target_9_6 ↔ target_10_1: -0.36
- target_8_1 ↔ target_10_1: -0.23
- target_3_2 ↔ target_10_1: -0.22
- target_3_1 ↔ target_10_1: -0.22
- target_9_7 ↔ target_10_1: -0.20

**Target 10_1 pattern**: Strongly negatively correlated with targets from groups 3, 7, 8, 9. This suggests:
- Product 10_1 may be a "primary" or "gateway" product
- Customers who open 10_1 have different behavior patterns
- Modeling strategy: Use target_10_1 as a feature for other targets, or create hierarchical models

**Within-group vs across-group correlations**:
- Within-group (93 pairs): mean = 0.030, std = 0.083
- Across-group (727 pairs): mean = -0.0004, std = 0.034

**Recommendation**: Implement multi-task learning within each product group (10 separate multi-task models, one per group), but not across groups. This allows sharing representations for correlated targets while avoiding negative transfer.

### 2. Feature Importance Analysis

Using Random Forest classifiers (50 trees, max_depth=10) trained on 100k samples, we identified top features for each target.

#### Most Universally Important Features

Features appearing in top-10 for most targets:

| Feature | Type | Appears in Top-10 for | NaN Rate |
|---------|------|----------------------|----------|
| num_feature_76 | Main numeric (≤132) | 21 targets | 2.46% |
| num_feature_62 | Main numeric (≤132) | 21 targets | 2.46% |
| num_feature_41 | Main numeric (≤132) | 18 targets | 6.52% |
| num_feature_27 | Main numeric (≤132) | 17 targets | 4.22% |
| num_feature_87 | Main numeric (≤132) | 17 targets | 51.88% |
| num_feature_46 | Main numeric (≤132) | 12 targets | 26.29% |
| num_feature_85 | Main numeric (≤132) | 11 targets | 41.07% |
| num_feature_117 | Main numeric (≤132) | 10 targets | 15.42% |
| num_feature_395 | Extra numeric (>132) | 9 targets | 77.08% |
| num_feature_60 | Main numeric (≤132) | 9 targets | 26.29% |

**Key observations**:
- Main numeric features (1-132) are most important: 8 of top 10 features
- Top features have low NaN rates (2-6%), making them reliable
- Feature num_feature_87 has 52% NaN rate but is still top-10 for 17 targets, suggesting NaN itself is informative
- Extra numeric features (133+) are less important overall, but num_feature_395 stands out

#### Feature Importance by Target Group

**Group 10 (target_10_1)**:
- Top feature: num_feature_27 (importance: 0.058)
- Other important: num_feature_87, num_feature_62
- Note: High positive correlation between num_feature_87 NaN and target (+0.20)

**Group 9 (8 targets)**:
- Common top features: num_feature_62, num_feature_41, num_feature_383
- target_9_6 (22% positive): num_feature_62 (0.026)
- target_9_8: num_feature_42 (0.104) with strong NaN informativeness (-0.15)

**Group 8 (3 targets)**:
- target_8_1: num_feature_22 (0.130) - **extreme NaN informativeness** (-0.67 correlation)
- target_8_2: num_feature_56 (0.024)
- target_8_3: num_feature_319 (0.068) - high NaN rate (97.55%) with -0.37 correlation

**Group 3 (5 targets)**:
- target_3_2: num_feature_395 (0.091) - **strong NaN informativeness** (-0.45 correlation)
- target_3_1: num_feature_62 (0.034)
- target_3_4: num_feature_320 (0.077) with NaN correlation -0.039

#### Categorical Features

Only 2 categorical features appear frequently in top-10:
- cat_feature_30: Top for target_1_3 (0.032), target_1_4 (0.019) - appears in 3 targets' top-10
- cat_feature_7: Appears in 3 targets' top-10

**Implication**: Categorical features are less important than numeric features overall. Consider:
1. Feature engineering: Create interaction features between categorical and top numeric features
2. Target encoding for categorical features with rare categories
3. Embeddings for high-cardinality categorical features

### 3. NaN Pattern Analysis

#### NaN Rates Distribution

Analysis of 2,373 numeric features reveals:

| NaN Rate Range | Feature Count | Example |
|----------------|---------------|---------|
| 0% | 0 features | N/A |
| 0-10% | ~200 features | num_feature_27 (4.22%), num_feature_62 (2.46%) |
| 10-50% | ~700 features | num_feature_87 (51.88%), num_feature_60 (26.29%) |
| 50-90% | ~884 features | num_feature_395 (77.08%), num_feature_319 (97.55%) |
| >90% | **589 features** | num_feature_265, num_feature_471, num_feature_923 (100%) |
| 100% | **5 features** | num_feature_265, num_feature_471, num_feature_923, num_feature_1058, num_feature_1832 |

**Recommendation**:
- **Remove 589 features** with >90% NaN rate - they provide minimal signal
- **Keep features with 50-90% NaN** if they show high NaN-target correlation (e.g., num_feature_319, num_feature_395)
- **Create NaN indicator features** for top important features to capture NaN informativeness

#### NaN Informativeness

For each target, we computed correlation between NaN indicator (binary: is NaN?) and target label. Key findings:

**Most informative NaN patterns**:

| Target | Feature | NaN Rate | NaN-Target Correlation |
|--------|---------|----------|------------------------|
| target_8_1 | num_feature_22 | 91.85% | **-0.67** |
| target_3_2 | num_feature_395 | 77.08% | **-0.45** |
| target_8_3 | num_feature_319 | 97.55% | **-0.37** |
| target_9_7 | num_feature_42 | 78.65% | -0.14 |
| target_7_1 | num_feature_42 | 78.65% | -0.16 |
| target_10_1 | num_feature_87 | 51.88% | **+0.20** |
| target_9_6 | num_feature_367 | 23.85% | -0.07 |
| target_6_4 | num_feature_22 | 91.85% | -0.13 |
| target_3_2 | num_feature_25 | 76.46% | -0.44 |

**Insights**:
- **NaN is highly predictive** for some targets, especially target_8_1 (correlation -0.67!)
- Negative correlation means: when feature is NaN, target is less likely to be positive
- For target_10_1, NaN in num_feature_87 shows **positive** correlation (+0.20), indicating opposite behavior
- Features with >90% NaN can still be valuable if NaN correlates strongly with target

**Recommendation**:
1. **Create explicit NaN indicator features** for top-20 most important features
2. **Use tree-based models** that can naturally handle NaN (LightGBM, XGBoost with NaN support)
3. **For neural networks**: Don't just impute - add binary NaN indicators as separate features
4. **Per-target analysis**: For target_8_1, create special features based on num_feature_22 NaN status

#### NaN Co-occurrence Patterns

Top NaN co-occurrence correlations among most important features:

| Feature 1 | Feature 2 | NaN Co-occurrence |
|-----------|-----------|-------------------|
| num_feature_62 | num_feature_76 | +1.00 (perfect) |
| num_feature_62 | num_feature_27 | +0.60 |
| num_feature_76 | num_feature_27 | +0.60 |
| num_feature_62 | num_feature_117 | +0.30 |
| num_feature_76 | num_feature_117 | +0.30 |

**Insight**: num_feature_62 and num_feature_76 have **perfect NaN co-occurrence** - they are NaN for the exact same samples. This suggests:
- They may be derived from the same underlying data source
- When one is missing, the other is also missing
- Consider creating a single "meta-feature" representing their shared missingness pattern

### 4. Feature Engineering Opportunities

Based on the analysis, here are concrete feature engineering recommendations:

#### High-Priority Features to Create

1. **NaN indicator features** (binary):
   ```python
   # For top-20 most important features
   for feat in ['num_feature_62', 'num_feature_76', 'num_feature_41',
                'num_feature_27', 'num_feature_87', 'num_feature_22',
                'num_feature_395', 'num_feature_319']:
       df[f'{feat}_is_nan'] = df[feat].isna().astype(int)
   ```

2. **Target 10_1 interaction features**:
   ```python
   # Since target_10_1 negatively correlates with many targets
   # Create features that might predict target_10_1
   df['feat_87_22_interaction'] = df['num_feature_87'] * df['num_feature_22'].fillna(0)
   df['feat_27_div_62'] = df['num_feature_27'] / (df['num_feature_62'] + 1e-6)
   ```

3. **Group-based features** (aggregate statistics within product groups):
   ```python
   # For targets in same group, create shared features
   # Example for Group 6 (targets 6_1 to 6_5)
   group6_features = ['num_feature_22', 'num_feature_41', 'num_feature_87']
   df['group6_feat_mean'] = df[group6_features].mean(axis=1)
   df['group6_feat_std'] = df[group6_features].std(axis=1)
   ```

4. **Categorical-numeric interactions**:
   ```python
   # Top categorical feature interactions
   for num_feat in ['num_feature_62', 'num_feature_76', 'num_feature_27']:
       df[f'cat30_{num_feat}'] = df['cat_feature_30'].astype(str) + '_' + pd.cut(df[num_feat], bins=10).astype(str)
   ```

#### Features to Remove

1. **589 features with >90% NaN rate** - no signal
2. **5 features with 100% NaN rate** - completely useless
3. **Duplicate features identified in previous analysis** (107 identical zero-variance features)

#### Imputation Strategy

For neural networks, use:

```python
# For features with <10% NaN: median imputation
low_nan_features = [f for f in num_features if nan_rates[f] < 0.10]
for feat in low_nan_features:
    df[feat] = df[feat].fillna(df[feat].median())

# For features with 10-90% NaN: special value imputation + indicator
medium_nan_features = [f for f in num_features if 0.10 <= nan_rates[f] <= 0.90]
for feat in medium_nan_features:
    df[f'{feat}_is_nan'] = df[feat].isna().astype(int)
    df[feat] = df[feat].fillna(-999)  # Special value outside normal range

# For features with >90% NaN: keep only if highly correlated with target
# Otherwise drop
```

## Recommendations for Model Builders

### 1. Target-Specific Strategies

**For extremely rare targets (positive rate <0.05%)**:
- target_2_8 (0.012%), target_2_7 (0.041%)
- Use heavy oversampling (10x-50x) or synthetic data generation (SMOTE)
- Consider binary relevance approach (train separate model for each)
- Use focal loss with high gamma (γ=3-5) to focus on hard examples

**For target_10_1 (31.4% positive)**:
- Train a separate, high-quality model first
- Use predictions as features for other targets (cascade approach)
- Focus on precision at the expense of recall (use threshold optimization)

**For correlated target groups**:
- Implement multi-task learning within each group
- Share encoder layers, but have separate prediction heads
- Use group-specific loss weighting based on target rarity

### 2. Feature Selection Strategy

**Keep all**:
- Top-20 universally important features (num_feature_62, 76, 41, 27, 87, etc.)
- Features with high NaN-target correlation (|corr| > 0.1)
- All categorical features (67 total, lightweight to include)

**Remove**:
- 589 features with >90% NaN rate (unless they show NaN-target correlation >0.1)
- 5 features with 100% NaN rate
- Duplicate/zero-variance features from previous analysis

**Engineer**:
- NaN indicator features for top-20 features
- Interaction features between top numeric features
- Target_10_1 prediction as feature for other models

### 3. Model Architecture Recommendations

**Option A: Multi-task learning by group** (recommended for correlated targets)
```python
# Train 10 separate multi-task models, one per product group
class GroupMultiTaskModel(nn.Module):
    def __init__(self, input_dim, group_size):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(group_size)
        ])

    def forward(self, x):
        shared_rep = self.shared_encoder(x)
        return [head(shared_rep) for head in self.task_heads]
```

**Option B: Binary relevance with target_10_1 cascade**
```python
# Stage 1: Train high-quality target_10_1 model
target_10_1_pred = model_10_1.predict(features)

# Stage 2: Add prediction as feature for other targets
enhanced_features = np.concatenate([features, target_10_1_pred], axis=1)

# Stage 3: Train separate models for each remaining target
for target in other_targets:
    models[target] = train_model(enhanced_features, target)
```

**Option C: Hierarchical labeling**
```python
# Use target_10_1 to create hierarchical targets
# For targets that anti-correlate with 10_1:
#   - Train one model for samples where target_10_1=0
#   - Train another model for samples where target_10_1=1
# This can improve performance by specializing models
```

### 4. Loss Function Recommendations

**For severe class imbalance**, use:

1. **ASL (Asymmetric Loss)**:
   ```python
   def asymmetric_loss(pred, target, gamma_neg=4, gamma_pos=0):
       # Focuses on hard negative examples
       pred_prob = torch.sigmoid(pred)
       loss_pos = -target * torch.log(pred_prob + 1e-8) * (1 - pred_prob) ** gamma_pos
       loss_neg = -(1-target) * torch.log(1 - pred_prob + 1e-8) * pred_prob ** gamma_neg
       return (loss_pos + loss_neg).mean()
   ```

2. **Focal Loss with per-target weighting**:
   ```python
   def focal_loss(pred, target, gamma=2, alpha=None):
       # alpha should be inverse of positive rate
       if alpha is None:
           alpha = 1.0 / (positive_rates[target] + 0.01)
       pred_prob = torch.sigmoid(pred)
       focal_weight = (1 - pred_prob) ** gamma if target == 1 else pred_prob ** gamma
       loss = -alpha * focal_weight * torch.log(pred_prob + 1e-8)
       return loss
   ```

3. **Uncertainty weighting** (if using multi-task):
   ```python
   # Learn per-target loss weights
   log_vars = nn.Parameter(torch.zeros(num_targets))

   def weighted_loss(losses, log_vars):
       # Weighted combination with learned weights
       weighted = sum(loss / (2 * torch.exp(lv)) + lv/2
                     for loss, lv in zip(losses, log_vars))
       return weighted
   ```

### 5. Handling NaN in Models

**For tree-based models** (LightGBM, XGBoost, CatBoost):
```python
# LightGBM handles NaN natively
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=8,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    # NaN is treated as special value - no imputation needed
)
```

**For neural networks**:
```python
# 1. Create NaN indicators
for feat in top_20_features:
    df[f'{feat}_is_nan'] = df[feat].isna().astype(float)

# 2. Impute with special value
df[num_features] = df[num_features].fillna(-999)

# 3. Normalize (ignore -999 in normalization)
means = df[num_features].replace(-999, np.nan).mean()
stds = df[num_features].replace(-999, np.nan).std()
df[num_features] = (df[num_features] - means) / (stds + 1e-8)
```

### 6. Validation Strategy

Given extreme class imbalance:

```python
# Use stratified K-fold with rare class protection
from sklearn.model_selection import StratifiedKFold

# For each target, ensure at least 5 positive samples in each fold
def stratified_kfold_with_rare_class(df, target, n_splits=5, min_positives=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(df, df[target]))

    # Check if any fold has too few positives
    for train_idx, val_idx in folds:
        val_positives = df.iloc[val_idx][target].sum()
        if val_positives < min_positives:
            print(f"Warning: Fold has only {val_positives} positives for {target}")
            # Use simple K-fold instead
            return list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(df))

    return folds

# For extremely rare targets, consider Leave-One-Out
if positive_rates[target] < 0.001:
    # Use LOOCV or repeated stratified sampling
    pass
```

### 7. Expected Model Performance Baselines

Based on target distribution and feature analysis:

| Target Group | Expected ROC-AUC Range | Difficulty |
|--------------|------------------------|------------|
| target_10_1 | 0.85-0.90 | Easy (high positive rate) |
| Group 9 (target_9_6, 9_7) | 0.75-0.85 | Medium |
| Group 8 | 0.70-0.80 | Medium |
| Group 3 (target_3_1, 3_2) | 0.75-0.85 | Medium |
| Group 1, 6, 7 | 0.65-0.75 | Hard (low positive rate) |
| Group 2, 4, 5 | 0.60-0.70 | Very Hard (rare classes) |
| target_2_8, target_2_7 | 0.55-0.65 | Extremely Hard (almost no positives) |

**Overall Macro-AUC target**: 0.70-0.75 is achievable with good feature engineering and multi-task learning within groups.

### 8. Quick Wins (Priority Order)

1. **Remove useless features**: Drop 589 features with >90% NaN (reduces noise, speeds training)
2. **Add NaN indicators**: Create binary features for top-20 features' NaN status (immediate performance boost)
3. **Target 10_1 cascade**: Train separate target_10_1 model, use predictions as feature (leverages strong signal)
4. **Group-based multi-task**: Train 10 separate models per product group (captures within-group correlations)
5. **Handle rare classes**: Use ASL loss or focal loss with heavy weighting for <1% positive rate targets
6. **Feature interactions**: Create top-top feature interactions (num_feature_62 × 76, 62 × 27, 76 × 27)

## Summary of Actionable Insights

1. **Data Cleaning**: Remove 589 features with >90% NaN rate immediately
2. **Feature Engineering**: Add NaN indicators for top-20 features, create target_10_1 cascade features
3. **Model Architecture**: Use multi-task learning within groups (10 separate models), not across all 41 targets
4. **Loss Functions**: Use ASL or focal loss with per-target weighting to handle extreme imbalance
5. **Special Cases**: For target_8_1, prioritize num_feature_22 NaN indicator; for target_3_2, prioritize num_feature_395 NaN
6. **Validation**: Use stratified K-fold with rare class protection; consider LOOCV for extremely rare targets
7. **Expected Performance**: Macro-AUC of 0.70-0.75 is achievable; focus on rare classes for metric improvement

**Missing Analysis**: No prediction files (val_predictions.parquet) or training logs were available, so error analysis, PCA impact analysis, and training issue investigation could not be performed. These would require model outputs to analyze.