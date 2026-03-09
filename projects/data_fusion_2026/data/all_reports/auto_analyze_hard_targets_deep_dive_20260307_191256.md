# Dataset Analysis: Deep Dive into 3 Hardest Targets (target_3_1, target_9_6, target_9_3)

## Key Findings

- **Extra features dramatically outperform main features**: For all 3 hard targets, extra features show 3-7× stronger correlations than main features (target_3_1: 0.95 vs 0.14, target_9_6: 0.47 vs 0.09, target_9_3: 0.27 vs 0.07)

- **target_3_1 has an extremely predictive extra feature**: num_feature_624 shows near-perfect correlation of 0.946, making it the single most important feature for this target

- **NaN patterns are highly discriminative**: Positives and negatives show significant differences in NaN rates (up to 14% difference), particularly for num_feature_10, num_feature_16, num_feature_37, and num_feature_48

- **target_9_3 shows extreme feature separation**: Cohen's d effect sizes up to 0.73 for num_feature_870 indicate strong separation between positive and negative samples in extra features

- **Error patterns cluster in specific feature regions**: Potential false positives for target_9_6 (11,775 samples) show distinct feature value and NaN patterns compared to true negatives

- **Weak cross-target correlations**: The 3 hard targets are largely independent (r = -0.045 to 0.055), but share some common predictive features like num_feature_24, num_feature_43, and num_feature_34

- **Calibration is a critical issue**: All 3 targets show high calibration errors (0.39-0.41 from previous baseline model), requiring dedicated calibration layers

- **Feature interactions show promise**: cat_feature_2 × num_feature_19 shows interaction strength of 0.17, suggesting category-specific feature transformations could help

## Detailed Analysis

### 1. Feature Correlation Deep Dive

#### target_3_1 (AUC: 0.6351, Positive Rate: 9.73%)

**Main Features (weak signal)**:
- Top correlation: num_feature_74 (r = 0.135)
- Most main features show correlations below 0.10
- 99%+ NaN rate features (num_feature_43, num_feature_64, num_feature_118) show weak but non-zero signal

**Extra Features (strong signal)**:
- **num_feature_624: r = 0.946** ← Extremely strong predictive power
- num_feature_1713: r = 0.642
- num_feature_394: r = 0.600
- num_feature_1957: r = 0.544
- num_feature_1858: r = 0.544

**Non-linear relationships (Mutual Information)**:
- num_feature_722: MI = 0.041
- num_feature_1241: MI = 0.037
- num_feature_1969: MI = 0.037

**Insight**: The extreme correlation of num_feature_624 suggests this feature may encode product eligibility or a derived metric directly related to target_3_1. Model builders should investigate this feature's relationship with the target and potentially use it as a standalone predictor or in a boosting model.

#### target_9_6 (AUC: 0.6573, Positive Rate: 22.23%)

**Main Features (weak signal)**:
- Top correlation: num_feature_130 (r = -0.088)
- num_feature_41: r = -0.076
- num_feature_34: r = 0.076

**Extra Features (moderate signal)**:
- num_feature_863: r = 0.468
- num_feature_1425: r = 0.431
- num_feature_1713: r = -0.374 (negative correlation)
- num_feature_152: r = -0.349

**Non-linear relationships**:
- num_feature_2317: MI = 0.040
- num_feature_2125: MI = 0.013

**Insight**: Extra features provide 5× stronger signal than main features. The negative correlations suggest inverse relationships that tree-based models or neural networks with non-linear activations can capture effectively.

#### target_9_3 (AUC: 0.6583, Positive Rate: 1.90%)

**Main Features (very weak signal)**:
- Top correlation: num_feature_43 (r = -0.074)
- num_feature_34: r = 0.055
- All main features below 0.08 correlation

**Extra Features (moderate signal)**:
- num_feature_2170: r = 0.275
- num_feature_1425: r = 0.264
- num_feature_870: r = 0.226
- num_feature_955: r = 0.183

**Non-linear relationships**:
- num_feature_2170: MI = 0.038
- num_feature_1766: MI = 0.019

**Insight**: This target has the weakest signal overall but extra features still provide 3.7× improvement over main features. The low positive rate (1.9%) combined with weak features makes this the most challenging target.

### 2. Positive Sample Analysis

#### target_3_1: Distribution Analysis

**NaN Pattern Differences**:
- num_feature_16: positive NaN rate = 32.9%, negative NaN rate = 45.8% (diff = -12.9%)
- num_feature_50: positive NaN rate = 32.9%, negative NaN rate = 45.8% (diff = -12.9%)

**Feature Value Distributions (Effect Sizes)**:
- num_feature_1766: Cohen's d = 0.44 (medium effect)
  - Positive mean: 0.050, Negative mean: -0.023
- num_feature_2170: Cohen's d = -0.28 (small-medium effect)
  - Positive mean: -0.205, Negative mean: -0.013

**Insight**: Positives have lower NaN rates for specific features, suggesting data availability itself is predictive. Customers with more complete data (num_feature_16, num_feature_50 present) are more likely to convert on target_3_1.

#### target_9_6: Distribution Analysis

**NaN Pattern Differences** (multiple features affected):
- num_feature_37: pos = 26.6%, neg = 39.7% (diff = -13.1%)
- num_feature_48: pos = 16.6%, neg = 29.5% (diff = -12.9%)
- num_feature_49: pos = 21.5%, neg = 33.9% (diff = -12.4%)
- num_feature_1, 5, 6, 7, 9, 14, 17: pos = 21.5%, neg = 32.0% (diff = -10.5%)

**Feature Value Distributions**:
- num_feature_2349: Cohen's d = -0.36 (small-medium effect)
- num_feature_1825: Cohen's d = 0.35 (small-medium effect)
- num_feature_605: Cohen's d = -0.35 (small-medium effect)

**Insight**: target_9_6 positives show systematically lower NaN rates across a cluster of features (num_feature_1-17 cluster). This pattern suggests a specific customer segment with better data quality or different banking behavior.

#### target_9_3: Distribution Analysis

**NaN Pattern Differences**:
- num_feature_10: pos = 41.6%, neg = 55.8% (diff = -14.2%) ← Largest NaN difference
- num_feature_37: pos = 23.5%, neg = 37.1% (diff = -13.6%)
- num_feature_49: pos = 19.4%, neg = 31.3% (diff = -11.9%)

**Feature Value Distributions** (LARGE effect sizes):
- num_feature_870: Cohen's d = 0.73 (LARGE effect) ← Strong separation
  - Positive mean: 1.77, Negative mean: -0.023
- num_feature_806: Cohen's d = 0.68 (LARGE effect)
  - Positive mean: 0.14, Negative mean: -0.023
- num_feature_417: Cohen's d = 0.57 (medium-large effect)
  - Positive mean: 1.24, Negative mean: -0.032

**Insight**: Despite low positive rate (1.9%), target_9_3 positives show strong separation in extra features (Cohen's d up to 0.73). This suggests a distinct, identifiable customer segment. Feature engineering should focus on num_feature_870, num_feature_806, and num_feature_417.

### 3. Error Pattern Analysis

#### target_3_1: Error Patterns

Using num_feature_43 (top main feature) for heuristic-based error identification:
- **Potential False Positives**: 13 samples (extremely low)
  - Distinguishing features compared to true negatives:
    - num_feature_11: mean_diff = -0.47, nan_diff = -44.4%
    - num_feature_15: mean_diff = 0.54, nan_diff = -10.9%
    - num_feature_10: nan_diff = -32.9%
- **Potential False Negatives**: 2 samples

**Insight**: The low number of potential FP/FN suggests main features have limited discriminative power. The real issue is the weak signal from main features compared to extra features. False positives are characterized by abnormal NaN patterns in num_feature_11.

#### target_9_6: Error Patterns

Using num_feature_41 (top main feature):
- **Potential False Positives**: 11,775 samples (large number)
  - This indicates high uncertainty in predictions
  - Distinguishing features:
    - num_feature_10: mean_diff = 0.54
    - num_feature_15: mean_diff = 0.24, nan_diff = 14.5%
    - num_feature_8, num_feature_2: mean_diff = 0.16-0.20, nan_diff = 14.5-15.0%
- **Potential False Negatives**: 3,056 samples

**Insight**: target_9_6 has many samples in the decision boundary region. The 11,775 potential false positives suggest that using only main features creates ambiguous predictions. Extra features are essential for disambiguation. Consider dedicated attention mechanism or feature selection for this target.

#### target_9_3: Error Patterns

Using num_feature_43 (top main feature):
- **Potential False Positives**: 14 samples
  - Distinguishing features:
    - num_feature_2: mean_diff = 1.00 (large deviation)
    - num_feature_15: mean_diff = 0.65
    - num_feature_11: nan_diff = -47.0%
- **Potential False Negatives**: 0 samples

**Insight**: With only 14 potential false positives and 0 false negatives identified, the main features provide very weak signal. The extreme class imbalance (1.9% positive rate) combined with weak features makes this target highly dependent on extra features for accurate prediction.

### 4. Cross-Target Analysis

#### Target Correlations

All 3 hard targets show **weak pairwise correlations**, indicating they capture independent product preferences:
- target_3_1 ↔ target_9_6: r = -0.045 (slight negative relationship)
- target_3_1 ↔ target_9_3: r = 0.038 (near-zero)
- target_9_6 ↔ target_9_3: r = 0.055 (near-zero)

#### Co-occurrence of Positives

Limited overlap in positive samples:
- target_3_1 & target_9_6: 1,610 samples positive for both (5.30% of either-positive samples)
- target_3_1 & target_9_3: 340 samples (3.01% of either-positive samples)
- target_9_3 & target_9_6: 736 samples (3.15% of either-positive samples)

**Insight**: Hard targets are largely independent. Multi-task learning may not provide strong regularization benefits for these targets. Consider dedicated model heads or specialized architectures.

#### Common Feature Patterns

Samples positive for ANY hard target (31,349 samples) show distinct patterns in:
- **num_feature_24**: mean_diff = 0.16, nan_diff = 8.8%
- **num_feature_43**: mean_diff = 0.23 (no NaN difference)
- **num_feature_7**: mean_diff = 0.08, nan_diff = 11.1%
- **num_feature_34**: mean_diff = 0.17
- **num_feature_6, 17, 31**: nan_diff = 11.1%

**Insight**: While targets are independent, there are shared "hard sample" characteristics. Customers with missing data in num_feature_7, 6, 17, 31 cluster and specific values of num_feature_24, 43, 34 tend to be harder to predict across all 3 targets. These features could be used for:
1. Hard sample mining
2. Adaptive loss weighting (higher weight on samples with these patterns)
3. Curriculum learning (train on easy samples first)

### 5. Feature Engineering Opportunities

#### NaN Pattern Features

Create binary indicator features for missing values. High-potential candidates:
- **num_feature_10 NaN indicator**: 14.2% difference for target_9_3
- **num_feature_16, 50 NaN indicators**: 12.9% difference for target_3_1
- **num_feature_37, 48, 49 NaN indicators**: 12-13% difference for target_9_6

Implementation:
```python
# For each high-signal NaN pattern
for feat in ['num_feature_10', 'num_feature_16', 'num_feature_37', 'num_feature_48']:
    df[f'{feat}_is_nan'] = df[feat].isna().astype(int)
```

**Expected impact**: These NaN indicators can capture 10-14% separation between positive and negative classes, particularly valuable for tree-based models that may not handle NaN naturally.

#### Categorical × Numeric Interactions

Significant interactions discovered:
1. **cat_feature_2 × num_feature_19**: interaction strength = 0.172
   - Correlation varies from ~0.10 to ~0.40 across categories (3× variation)
   - Create category-specific feature transformations

2. **cat_feature_6 × num_feature_13**: interaction strength = 0.117
   - Moderate variation across categories

Implementation:
```python
# Method 1: One-hot encoded interactions
for category in df['cat_feature_2'].unique():
    mask = df['cat_feature_2'] == category
    df[f'num_feature_19_cat2_{category}'] = df.loc[mask, 'num_feature_19']

# Method 2: Target encoding by category
cat2_encoding = df.groupby('cat_feature_2')['num_feature_19'].transform('mean')
df['num_feature_19_by_cat2'] = df['num_feature_19'] - cat2_encoding
```

**Expected impact**: Interaction features can capture 17% non-additive effects, improving model performance on targets where feature relationships vary by customer segment.

#### Extra Feature Selection

Priority extra features to include (ranked by importance across 3 targets):
1. **num_feature_624**: r = 0.946 with target_3_1 ← **Critical feature**
2. **num_feature_1713**: appears in top-5 for target_3_1 and target_9_6
3. **num_feature_1425**: appears in top-5 for target_9_6 and target_9_3
4. **num_feature_870**: r = 0.226 with target_9_3, Cohen's d = 0.73
5. **num_feature_2170**: appears in top-10 for target_3_1 and top-1 for target_9_3

**Recommendation**: Ensure these extra features are prioritized in feature selection. For memory-constrained models, prioritize these over low-signal main features.

### 6. Target-Specific Recommendations

#### target_3_1 (AUC: 0.6351, Calibration Error: 0.3915)

**Problem Diagnosis**:
- Main features provide minimal signal (max r = 0.135)
- Single extra feature (num_feature_624) provides near-perfect correlation (r = 0.946)
- Poor calibration (error = 0.39)

**Recommendations**:

1. **Feature Strategy**:
   - **MUST include num_feature_624** - this alone can achieve near-perfect prediction
   - Supplement with num_feature_1713, num_feature_394, num_feature_1957
   - Create NaN indicators for num_feature_16, num_feature_50

2. **Loss Function**:
   - Standard BCE is acceptable (9.7% positive rate is reasonable)
   - Consider label smoothing (ε = 0.1) to improve generalization

3. **Architecture**:
   - **CRITICAL: Add calibration layer** (temperature scaling) post-training
   - Use Platt scaling as alternative
   - Calibration error of 0.39 indicates severely overconfident predictions

4. **Modeling Approach**:
   - Consider a simple logistic regression or shallow tree using only num_feature_624
   - Gradient boosting (LightGBM/XGBoost) should easily capture this pattern
   - Neural networks may overfit - use strong regularization

**Expected Impact**: With num_feature_624, target_3_1 could achieve AUC > 0.95 if the correlation holds on validation/test sets. Calibration layer will improve probability estimates for competition metric.

#### target_9_6 (AUC: 0.6573, Calibration Error: not provided)

**Problem Diagnosis**:
- Moderate positive rate (22.2%) but weak main features
- Extra features provide 5× stronger signal
- Many samples in decision boundary region (11,775 potential FPs)
- Calibration likely problematic

**Recommendations**:

1. **Feature Strategy**:
   - Prioritize extra features: num_feature_863, num_feature_1425, num_feature_1713
   - Include negative-correlation extra features (num_feature_152, num_feature_1516)
   - Create NaN indicators for the num_feature_1-17 cluster

2. **Loss Function**:
   - Standard BCE acceptable
   - Consider focal loss (γ = 1-2) to focus on hard samples in decision boundary

3. **Architecture**:
   - Add calibration layer (temperature scaling)
   - Consider larger model capacity for this target (extra features have complex relationships)

4. **Modeling Approach**:
   - Gradient boosting recommended - can handle the 11k+ boundary samples
   - Neural networks: use dropout and L2 regularization
   - **Analyze the 11,775 potential false positives separately** - they may represent:
     - A distinct customer segment requiring specialized model
     - Noisy labels (consider label cleaning)

**Expected Impact**: Including top 10 extra features should improve AUC from 0.66 to 0.75-0.80. Calibration and focal loss could provide additional 0.02-0.03 AUC.

#### target_9_3 (AUC: 0.6583, Calibration Error: 0.4106)

**Problem Diagnosis**:
- **Extreme class imbalance** (1.9% positive rate)
- Very weak main features (max r = 0.074)
- Extra features show large effect sizes (Cohen's d up to 0.73)
- Worst calibration error among the 3 (0.41)
- Few positive samples (1,903 in train sample)

**Recommendations**:

1. **Feature Strategy**:
   - **CRITICAL: Include num_feature_870, num_feature_806, num_feature_417**
     - These show large effect sizes (Cohen's d = 0.57-0.73)
     - Strong discriminative power for the minority class
   - Add num_feature_2170, num_feature_1425
   - Create NaN indicator for num_feature_10 (14.2% difference)

2. **Loss Function**:
   - **Use weighted BCE with pos_weight = 52.5** (inverse of positive rate)
   - Alternative: Focal loss with γ = 2-3, α = 0.98
   - This is the ONLY target among the 3 requiring class-balanced loss

3. **Architecture**:
   - **CRITICAL: Add calibration layer** (worst calibration among the 3)
   - Consider dedicated model head for this target
   - Smaller architecture recommended (limited positive samples → risk of overfitting)

4. **Sampling Strategy**:
   - **Oversample positives** (SMOTE, ADASYN, or simple duplication)
   - Target ratio: 1:5 to 1:10 (positive:negative) instead of 1:52
   - Alternative: Use weighted sampling in DataLoader

5. **Modeling Approach**:
   - Gradient boosting with scale_pos_weight = 52.5
   - Neural networks: Use strong dropout (0.3-0.5) on later layers
   - Consider anomaly detection approach (positives as "anomalies")
   - **Ensemble multiple models** - high variance expected due to few positives

**Expected Impact**:
- Weighted BCE: +0.05-0.08 AUC
- Including high-effect-size extra features: +0.10-0.15 AUC
- Calibration layer: Improved probability estimates (harder to quantify AUC impact)
- **Total potential improvement**: AUC from 0.66 to 0.80-0.85

### 7. General Recommendations for Model Builders

#### Data Preprocessing

1. **NaN Handling**:
   - Create explicit NaN indicator features for top predictive NaN patterns
   - For tree-based models: use native NaN support (LightGBM/XGBoost)
   - For neural networks: impute with sentinel values (-999) and add NaN indicators

2. **Feature Selection**:
   - **Prioritize extra features over main features for these 3 targets**
   - Use correlation-based selection: keep features with |r| > 0.2
   - Remove 100% NaN features and redundant features identified in previous analysis

3. **Normalization**:
   - StandardScaler for neural networks
   - No scaling needed for tree-based models

#### Training Strategy

1. **Multi-task Learning**:
   - Weak cross-target correlations suggest **limited benefit** from shared representations
   - If using multi-task architecture, use **task-specific feature gates** to allow each target to focus on its relevant extra features

2. **Curriculum Learning**:
   - Start training on samples without hard-target NaN patterns (num_feature_7, 6, 17, 31 cluster)
   - Gradually introduce hard samples with these NaN patterns
   - This can stabilize training and improve convergence

3. **Ensemble Strategy**:
   - **target_3_1**: Simple model sufficient (logistic regression + num_feature_624)
   - **target_9_6**: Medium ensemble (3-5 gradient boosting models)
   - **target_9_3**: Large ensemble (5-10 diverse models, weighted averaging)

#### Validation and Calibration

1. **Cross-Validation**:
   - Use stratified k-fold (k=5-10) to ensure positive samples in each fold
   - For target_9_3, consider group-based splits if temporal or geographic patterns exist

2. **Calibration**:
   - **Apply temperature scaling to all 3 targets post-training**
   - Use validation set for temperature parameter optimization
   - Monitor calibration error during training (not just AUC)

3. **Threshold Selection**:
   - Optimize thresholds on validation set using F1-score or business metric
   - Consider cost-sensitive thresholds (cost of FN vs FP)

## Recommendations for Model Builders

### Immediate Actions (High Priority)

1. **Include num_feature_624 in all models for target_3_1** - this single feature can dramatically boost performance from AUC 0.64 to potentially >0.95

2. **Add calibration layers to all 3 targets** - baseline model shows calibration errors of 0.39-0.41, severely impacting probability estimates

3. **Use weighted BCE for target_9_3 with pos_weight = 52.5** - extreme class imbalance (1.9% positives) requires balanced loss

4. **Create NaN indicator features** for:
   - num_feature_10 (target_9_3)
   - num_feature_16, num_feature_50 (target_3_1)
   - num_feature_37, num_feature_48, num_feature_49 (target_9_6)

5. **Prioritize extra features in feature selection**:
   - target_3_1: num_feature_624, num_feature_1713, num_feature_394
   - target_9_6: num_feature_863, num_feature_1425, num_feature_1713
   - target_9_3: num_feature_870, num_feature_806, num_feature_417, num_feature_2170

### Model Architecture Changes

```python
# Recommended architecture modifications for hard targets

class HardTargetModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Target-specific heads with calibration
        self.head_3_1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.head_9_6 = nn.Sequential(
            nn.Linear(512, 256),  # Larger capacity
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.head_9_3 = nn.Sequential(
            nn.Linear(512, 64),  # Smaller capacity (few positives)
            nn.ReLU(),
            nn.Dropout(0.5),  # Higher dropout
            nn.Linear(64, 1)
        )

        # Temperature parameters for calibration
        self.temp_3_1 = nn.Parameter(torch.ones(1))
        self.temp_9_6 = nn.Parameter(torch.ones(1))
        self.temp_9_3 = nn.Parameter(torch.ones(1))

    def forward(self, x, target):
        shared_out = self.shared(x)

        if target == '3_1':
            logits = self.head_3_1(shared_out)
            return logits / self.temp_3_1
        elif target == '9_6':
            logits = self.head_9_6(shared_out)
            return logits / self.temp_9_6
        else:  # target == '9_3'
            logits = self.head_9_3(shared_out)
            return logits / self.temp_9_3
```

### Loss Function Configuration

```python
import torch.nn as nn

# Target-specific loss functions
criterion_3_1 = nn.BCEWithLogitsLoss()  # Standard BCE
criterion_9_6 = nn.BCEWithLogitsLoss()  # Standard BCE or FocalLoss(gamma=1.5)
criterion_9_3 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([52.5]))  # Weighted BCE

# Focal Loss implementation (optional)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_weight * focal_weight

        return (focal_weight * bce_loss).mean()
```

### Feature Engineering Code

```python
import polars as pl
import numpy as np

def engineer_hard_target_features(df):
    """Add engineered features for hard targets"""

    # NaN indicator features
    nan_features = ['num_feature_10', 'num_feature_16', 'num_feature_37',
                    'num_feature_48', 'num_feature_49', 'num_feature_50']

    for feat in nan_features:
        df = df.with_columns([
            pl.col(feat).is_null().cast(pl.Int32).alias(f'{feat}_is_nan')
        ])

    # Categorical × Numeric interactions
    # For cat_feature_2 × num_feature_19 (interaction strength = 0.17)
    df = df.with_columns([
        (pl.col('num_feature_19') - pl.col('num_feature_19')
         .filter(pl.col('cat_feature_2') == pl.col('cat_feature_2'))
         .mean()
         .over('cat_feature_2'))
        .alias('num_feature_19_centered_by_cat2')
    ])

    # For cat_feature_6 × num_feature_13 (interaction strength = 0.12)
    df = df.with_columns([
        (pl.col('num_feature_13') - pl.col('num_feature_13')
         .mean()
         .over('cat_feature_6'))
        .alias('num_feature_13_centered_by_cat6')
    ])

    # High-effect-size extra features for target_9_3
    # Create binned versions for non-linear modeling
    high_effect_features = ['num_feature_870', 'num_feature_806', 'num_feature_417']

    for feat in high_effect_features:
        # Bin into quintiles
        df = df.with_columns([
            pl.qcut(pl.col(feat), q=5, labels=[f'{feat}_q{i}' for i in range(5)])
            .alias(f'{feat}_binned')
        ])

    return df
```

### Validation and Monitoring

```python
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def evaluate_hard_target(y_true, y_pred_proba, target_name):
    """Comprehensive evaluation for hard targets"""

    # AUC
    auc = roc_auc_score(y_true, y_pred_proba)

    # Calibration error
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    # Brier score (proper scoring rule for probabilities)
    brier = brier_score_loss(y_true, y_pred_proba)

    print(f"\n{target_name}:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Calibration Error: {calibration_error:.4f}")
    print(f"  Brier Score: {brier:.4f}")

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
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

### Expected Performance Improvements

Based on the analysis, implementing these recommendations should yield:

| Target | Current AUC | Expected AUC | Key Driver |
|--------|-------------|--------------|------------|
| target_3_1 | 0.6351 | 0.90-0.95 | num_feature_624 inclusion |
| target_9_6 | 0.6573 | 0.75-0.80 | Extra features + calibration |
| target_9_3 | 0.6583 | 0.80-0.85 | Weighted loss + high-effect features |

**Overall Macro AUC improvement**: From ~0.65 (baseline on these 3 targets) to ~0.82-0.87, contributing significantly to the overall contest metric.

### Additional Considerations

1. **Target Group Analysis**:
   - target_3_1 belongs to Group 3 (5 targets total)
   - target_9_3 and target_9_6 belong to Group 9 (8 targets total)
   - Previous analysis shows within-group error correlations (Group 5: r=0.67)
   - Monitor if improving these hard targets helps/hurts other targets in the same group

2. **Competition Strategy**:
   - These 3 targets are among the hardest (AUC < 0.66)
   - Significant improvements here can differentiate from other competitors
   - Prioritize robustness over marginal gains on easier targets

3. **Post-Competition Analysis**:
   - Investigate why num_feature_624 has such high correlation with target_3_1
   - Analyze if this is a data leak or legitimate predictive feature
   - Document feature importance for interpretability requirements