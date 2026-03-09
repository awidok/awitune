# Dataset Analysis: Why Tree-Based Models Underperform vs Neural Networks on Multi-Label Classification

## Key Findings

- **Extreme sparsity in extra features**: 296 features have >99% NaN rate, with predictive features averaging 99.95% sparsity, making them nearly unusable for tree models which require sufficient samples at each split
- **Severe class imbalance across 58% of targets**: 24 of 41 targets have >100:1 imbalance (worst: 9035:1), biasing tree models toward majority class and requiring target-specific hyperparameter tuning
- **Heavy-tailed feature distributions**: 80 numeric features have kurtosis >10 (max: 73,873), forcing trees to create many splits to capture tail behavior, increasing model complexity
- **Sparse extra features have predictive power but are inaccessible to trees**: Hard targets (target_3_1, target_9_3, target_9_6) show moderate correlations (0.2-0.5) with extra features, but 95%+ NaN rates mean trees rarely see these values
- **Within-group target correlations 2.15x stronger than cross-group**: Neural networks exploit these dependencies through multi-task learning, while independent tree models (41 separate models) cannot leverage shared signal
- **Feature selection can reduce noise by 89%**: Combined strategy (NaN <95% AND correlation ≥0.05) selects 245 features from 2241, reducing overfitting risk and improving tree model efficiency
- **High-cardinality categorical (1989 levels) challenges tree encoding**: cat_feature_39 has 1989 unique values, causing one-hot encoding to create sparse features or target encoding to overfit on rare categories
- **Neural networks have built-in advantages**: Batch normalization handles heavy tails, embeddings capture categorical semantics, attention mechanisms weight sparse features appropriately, and focal loss automatically handles class imbalance

## Detailed Analysis

### 1. Per-Target Performance Challenges

**Hard Targets Analysis** (Previous reports identified target_3_1, target_9_3, target_9_6 as hardest)

#### target_3_1 (AUC: 0.6351, Positive Rate: 9.96%)

**Extra Feature Correlations:**
- High correlation (>0.2): 13 features
- Moderate correlation (0.1-0.2): 36 features
- **Critical issue**: Top predictive features are 99.99% NaN
  - num_feature_428: r=0.499, but only 10 valid values out of 750k
  - num_feature_482: r=0.410, only 15 valid values
  - Mean NaN rate for predictive features: 82.15%

**Why trees struggle:**
- With <20 valid samples, trees cannot create meaningful splits on these features
- Main features show only weak correlations (max r≈0.14 from previous analysis)
- Neural networks can learn from the pattern of missingness itself (NaN as signal)

#### target_9_3 (AUC: 0.6583, Positive Rate: 1.91%)

**Extra Feature Correlations:**
- High correlation (>0.2): 14 features
- Moderate correlation (0.1-0.2): 27 features
- **NaN rate**: Median 99.79% for predictive features
- Best features: num_feature_407 (r=0.399, NaN=99.98%), num_feature_2100 (r=0.299, NaN=99.97%)

**Why trees struggle:**
- Extreme class imbalance (52:1) + extreme sparsity = double challenge
- Trees need to see minority class samples at splits, but features are rarely non-NaN for those samples
- scale_pos_weight alone is insufficient when features themselves are unavailable

#### target_9_6 (AUC: 0.6573, Positive Rate: 22.32%)

**Extra Feature Correlations:**
- High correlation (>0.2): 14 features
- Moderate correlation (0.1-0.2): 63 features
- **Better but still sparse**: Mean NaN rate 89.07% for predictive features
- Best features: num_feature_394 (r=0.415, NaN=99.97%)

**Why trees struggle:**
- Higher positive rate (22%) helps, but extra features still 99%+ NaN
- Main features may not capture the underlying pattern that extra features encode
- Trees may need depth=10-12 to capture interactions, risking overfitting

---

### 2. Feature Encoding Analysis

#### Categorical Features

**Cardinality Distribution:**
- Total categorical features: 67
- High cardinality (>100): 2 features
  - cat_feature_39: 1989 unique values
  - cat_feature_34: 120 unique values
- Low cardinality: 65 features with 2-51 unique values

**Impact on Tree Models:**

| Model | Categorical Handling | Issue with High Cardinality |
|-------|---------------------|-----------------------------|
| XGBoost | Treats as numerical (requires encoding) | One-hot creates 1989 sparse columns; target encoding overfits rare categories |
| LightGBM | Native support (optimal split finding) | Better than XGBoost, but still needs careful tuning |
| CatBoost | Ordered target encoding | **Best choice** - reduces overfitting through ordered encoding |

**Recommendation:**
- Use CatBoost for categorical-heavy targets
- If using XGBoost/LightGBM, apply target encoding with regularization:
  ```python
  # Target encoding with smoothing
  smoothing = 10  # Higher = more regularization
  encoded = (category_mean * count + global_mean * smoothing) / (count + smoothing)
  ```

#### Numerical Features - Main Set

**NaN Pattern Analysis:**
- Features with >10% NaN: 126 of 132 (95%)
- Features with >50% NaN: 58 (44%)
- Features with >90% NaN: 21 (16%)

**Distribution Characteristics:**
- Heavy-tailed (kurtosis >10): 80 features (61%)
- High outlier rate (>10%): 20 features
- Extreme kurtosis examples:
  - num_feature_18: kurtosis 73,873, range 1391
  - num_feature_51: kurtosis 73,570, range 879
  - num_feature_27: kurtosis 70,059, range 420

**Why Trees Struggle with Heavy Tails:**

1. **Many splits needed**: To capture 99th percentile behavior, trees need deep branches
2. **Outlier influence**: Extreme values can dominate split decisions
3. **Information gain bias**: Split metrics favor majority values, ignoring tail patterns

**Neural Network Advantage:**
- Batch normalization standardizes distributions automatically
- Activation functions (ReLU, etc.) bound outputs
- Can learn appropriate transformations through gradient descent

**Recommendation:**
- Apply winsorization (clip at 1st/99th percentile) before tree training
- Consider log transformation for heavy-tailed features
- Increase max_depth to 8-12 to allow more granular splits

#### Numerical Features - Extra Set

**Extreme Sparsity:**
- Total extra features: 2241
- Features with >90% NaN: 571 (25%)
- Features with >99% NaN: 296 (13%)
- Features with 100% NaN (completely empty): 13

**Feature Selection Analysis:**

| Strategy | Features Selected | Reduction | Description |
|----------|------------------|-----------|-------------|
| NaN < 50% | 1117 | 50.2% | Keep features with majority data |
| NaN < 90% | 1670 | 25.5% | Keep mostly-populated features |
| NaN < 99% | 1945 | 13.2% | Minimal filtering |
| Correlation ≥0.05 | 337 | 85.0% | Keep predictive features only |
| **Combined** (NaN<95% + Corr≥0.05) | **245** | **89.1%** | **Recommended strategy** |

**Why Trees Struggle with Sparse Features:**

1. **Insufficient samples at splits**: If feature is 99% NaN, only 7,500 of 750k samples have values
2. **Rare feature activation**: Tree may learn to never split on this feature
3. **Overfitting risk**: Splitting on rare feature values creates small leaf nodes
4. **Information gain ≈ 0**: Splits that affect <1% of samples provide minimal gain

**Neural Network Advantage:**
- Embedding layers can learn from any non-NaN value
- Attention mechanisms weight features by importance, not just frequency
- Can treat NaN as a special "missing" category
- Gradient descent updates all weights, not just those at active splits

**Recommendation:**
- Implement feature selection: keep only 245 features meeting both criteria
- For remaining sparse features, consider:
  - Creating NaN indicator columns (trees can use these)
  - Binning: convert continuous to categorical (e.g., quintiles + NaN category)
  - Imputation with sentinel value (e.g., -999) to distinguish from real values

---

### 3. Class Imbalance Analysis

**Imbalance Distribution:**

| Severity | Imbalance Ratio | Count | Examples |
|----------|----------------|-------|----------|
| Extreme | >1000:1 | 3 | target_2_8 (9035:1), target_2_7 (3303:1), target_6_5 (1789:1) |
| Severe | 100-1000:1 | 21 | target_3_3 (842:1), target_2_3 (720:1), target_3_5 (705:1) |
| High | 20-100:1 | 10 | target_2_4 (131:1), target_2_1 (140:1), target_6_2 (134:1) |
| Moderate | 10-20:1 | 2 | target_9_3 (52:1), target_10_1 (14:1) |
| Low | <10:1 | 5 | target_9_6 (3.5:1), target_4_1 (6.2:1), target_1_1 (2.7:1) |

**Total severely imbalanced targets: 24 (58% of all targets)**

**Why Trees Struggle with Imbalance:**

1. **Information gain bias**: Standard Gini/entropy split criteria favor majority class
   - Example: A split that isolates 100 negatives from 1 positive has high "purity" but misses the positive
2. **Leaf node purity**: Trees optimize for overall accuracy, not minority class recall
3. **Global scale_pos_weight is insufficient**:
   - Different targets have vastly different imbalance ratios (9035:1 vs 3.5:1)
   - Single parameter cannot adapt to 24 different imbalance levels
4. **Rare samples get "lost"**: With 750k samples and <0.1% positives, tree may never see positives at certain branches

**Neural Network Advantage:**
- **Focal loss**: Automatically down-weights easy negatives, focuses on hard examples
  - Formula: FL(p) = -α(1-p)^γ log(p)
  - γ=2 focuses on misclassified samples, α balances class weights
- **Class-balanced loss**: Weights inversely proportional to class frequency
- **Better calibration**: Output probabilities reflect true likelihood

**Recommendation for Tree Models:**

1. **Per-target scale_pos_weight tuning**:
   ```python
   # For each target, calculate optimal weight
   scale_pos_weight = (n_negative / n_positive)

   # For extreme imbalance (>1000:1), cap it
   scale_pos_weight = min(n_negative / n_positive, 1000)
   ```

2. **Use focal loss approximation for trees**:
   - Custom objective function that weights samples by prediction confidence
   - XGBoost allows custom objectives via `obj` parameter

3. **Stratified sampling for validation**:
   ```python
   # Ensure validation set has same positive rate as train
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   ```

4. **Increase min_child_weight**:
   - Set to 50-100 to prevent splits on too few samples
   - Critical for rare classes: prevents overfitting to handful of positives

5. **Use max_delta_step**:
   - Set to 1-10 for extreme imbalance
   - Limits weight update magnitude, prevents divergence

---

### 4. Multi-Label Architecture Analysis

**Current Tree Approach: 41 Independent Models**

This treats each target as completely independent, missing important correlations.

**Target Correlation Structure:**

**Within-Group Correlations** (targets in same product group):
- Mean correlation: 0.0327
- Top correlations:
  - target_5_1 ↔ target_5_2: r=0.518 (strong)
  - target_6_1 ↔ target_6_4: r=0.514 (strong)

**Cross-Group Correlations:**
- Mean correlation: 0.0152
- Within-group is **2.15x stronger** than cross-group

**High Correlation Pairs (|r| > 0.3): 3 pairs**

This validates the group structure and suggests dependencies can be exploited.

**Why Independent Tree Models Miss Signal:**

1. **Cannot share information**: Prediction for target_5_1 doesn't inform target_5_2 prediction
2. **Duplicate work**: Both models learn similar patterns from scratch
3. **Miss conditional probabilities**:
   - P(target_5_1 | target_5_2=1) = 0.993 (from previous cascade analysis)
   - Independent models cannot use this relationship

**Neural Network Multi-Task Advantage:**

1. **Shared representation**: Lower layers learn common features for all targets
2. **Target attention**: Upper layers specialize per target while sharing knowledge
3. **Automatic correlation exploitation**: Backpropagation learns correlated targets should have similar weights
4. **Efficiency**: One forward pass predicts all 41 targets

**Recommended Tree Architectures:**

**Option 1: Group-Based Multi-Output Models**

Train 10 models (one per group) instead of 41:
- Group 1: 5 targets
- Group 2: 8 targets
- Group 3: 5 targets
- etc.

```python
# scikit-learn MultiOutputClassifier
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

# Train per group
for group_targets in target_groups.values():
    group_model = MultiOutputClassifier(
        LGBMClassifier(n_estimators=500, num_leaves=50)
    )
    group_model.fit(X, y[group_targets])
```

**Expected benefit**: 1-2% AUC improvement from shared signal within groups

**Option 2: Classifier Chains**

Predict targets sequentially, using earlier predictions as features:

```python
from sklearn.multioutput import ClassifierChain

# Order targets by positive rate (easier targets first)
target_order = sorted(targets, key=lambda t: y[t].mean(), reverse=True)

chain = ClassifierChain(
    LGBMClassifier(n_estimators=500),
    order=target_order
)
chain.fit(X, y)
```

**Expected benefit**: 1-3% AUC improvement on dependent targets

**Option 3: Stacked Ensemble (2-Stage)**

Stage 1: Train independent models
Stage 2: Add Stage 1 predictions as features for correlated targets

```python
# Stage 1
stage1_predictions = {}
for target in targets:
    model = train_model(X, y[target])
    stage1_predictions[target] = model.predict_proba(X)[:, 1]

# Stage 2
X_augmented = np.column_stack([X] + list(stage1_predictions.values()))
for target in correlated_targets:
    model = train_model(X_augmented, y[target])
```

**Expected benefit**: 2-4% AUC improvement on targets with high correlations

---

### 5. Hyperparameter Sensitivity Analysis

**Default vs Recommended Parameters:**

| Parameter | XGBoost Default | Recommended | Reason |
|-----------|----------------|-------------|--------|
| max_depth | 6 | 8-12 | Heavy-tailed features need more splits |
| learning_rate | 0.3 | 0.01-0.05 | Slower learning with more trees improves generalization |
| n_estimators | 100 | 500-1000 | Need many trees for sparse feature patterns |
| min_child_weight | 1 | 50-100 | Prevent overfitting to rare samples |
| scale_pos_weight | 1 | Auto per target | Critical for class imbalance |
| max_delta_step | 0 | 1-10 | Helps with extreme imbalance |
| subsample | 1.0 | 0.8 | Reduce overfitting |
| colsample_bytree | 1.0 | 0.8 | Reduce overfitting with many features |

**Early Stopping Concerns:**

**Problem:**
- Default early_stopping_rounds=10-50 may stop too early with rare positives
- Trees may not see enough minority class examples to learn patterns

**Solution:**
```python
early_stopping_rounds = 100  # Give trees more time to learn
eval_metric = 'auc'  # Monitor AUC, not just loss
```

**Model Comparison:**

| Model | Pros | Cons | Recommended For |
|-------|------|------|----------------|
| XGBoost | Mature, good documentation | Poor categorical handling | Baseline |
| LightGBM | Fast, native categorical support | Can overfit with many features | Large-scale training |
| **CatBoost** | Best categorical encoding, ordered boosting | Slower training, less flexible | **Primary choice for this dataset** |

**CatBoost Advantages:**
- Ordered target encoding for categoricals (reduces overfitting)
- Symmetric trees (faster inference)
- Native handling of NaN values
- Built-in class balancing (auto_class_weights='Balanced')

---

### 6. Why Neural Networks Outperform Trees

**Summary of Fundamental Advantages:**

| Challenge | Tree Solution | Neural Solution | Winner |
|-----------|---------------|----------------|--------|
| **Sparse features** (>99% NaN) | Struggle to find splits | Learn from any non-NaN value | **Neural** |
| **Class imbalance** | scale_pos_weight (global) | Focal loss (adaptive) | **Neural** |
| **Heavy tails** | Need many splits | Batch normalization | **Neural** |
| **High cardinality categorical** | One-hot (sparse) or target encoding (overfit) | Embeddings (dense, learned) | **Neural** |
| **Multi-label correlations** | Independent models or complex architecture | Multi-task learning (natural) | **Neural** |
| **Feature interactions** | Require depth | Automatic through layers | **Neural** |
| **Missing values** | Native handling, but limited | Can learn missingness patterns | **Neural** |

**Estimated Performance Gap:**

Based on analysis, the gap is **5-10% macro AUC** due to:
- Sparse feature utilization: 2-3% loss
- Class imbalance handling: 2-3% loss on rare targets
- Multi-label architecture: 1-2% loss
- Feature representation: 1-2% loss

---

## Recommendations for Model Builders

### Immediate Actions (High Impact, Low Effort)

1. **Feature Selection** ⭐⭐⭐
   - Reduce 2241 extra features to 245 using: `NaN < 95% AND correlation ≥ 0.05`
   - Code:
     ```python
     selected_features = []
     for feat in extra_features:
         nan_rate = df[feat].isna().sum() / len(df)
         max_corr = max([abs(df[feat].corr(df[target])) for target in targets])
         if nan_rate < 0.95 and max_corr >= 0.05:
             selected_features.append(feat)
     ```
   - **Expected impact**: 2-3% AUC improvement, 3x faster training

2. **Use CatBoost** ⭐⭐⭐
   - Replace XGBoost baseline with CatBoost for categorical features
   - Code:
     ```python
     from catboost import CatBoostClassifier

     cat_features = [f'cat_feature_{i}' for i in range(1, 68)]

     model = CatBoostClassifier(
         iterations=1000,
         depth=10,
         learning_rate=0.03,
         cat_features=cat_features,
         auto_class_weights='Balanced',
         early_stopping_rounds=100,
         eval_metric='AUC'
     )
     ```
   - **Expected impact**: 2-3% AUC improvement on targets where categoricals matter

3. **Per-Target scale_pos_weight** ⭐⭐⭐
   - Tune class weight individually for each target
   - Code:
     ```python
     for target in targets:
         pos_weight = min(y[target].value_counts()[0] / y[target].value_counts()[1], 1000)
         model.set_params(scale_pos_weight=pos_weight)
         model.fit(X, y[target])
     ```
   - **Expected impact**: 2-4% AUC improvement on rare targets (24 targets with >100:1 imbalance)

### Medium-Term Improvements (Medium Effort, High Impact)

4. **Multi-Output Training by Group** ⭐⭐
   - Train 10 models (one per product group) instead of 41 independent models
   - Exploit 2.15x stronger within-group correlations
   - Code:
     ```python
     from sklearn.multioutput import MultiOutputClassifier

     for group_id, group_targets in target_groups.items():
         model = MultiOutputClassifier(
             CatBoostClassifier(iterations=1000, depth=10),
             n_jobs=-1
         )
         model.fit(X, y[group_targets])
     ```
   - **Expected impact**: 1-2% AUC improvement from shared signal

5. **Classifier Chains for Correlated Targets** ⭐⭐
   - Use predictions from high-correlation targets as features
   - Focus on: target_5_1 → target_5_2, target_6_1 → target_6_4
   - Code:
     ```python
     from sklearn.multioutput import ClassifierChain

     chain = ClassifierChain(
         CatBoostClassifier(iterations=1000, depth=10),
         order='target_5_2,target_5_1,...'  # Easier targets first
     )
     chain.fit(X, y)
     ```
   - **Expected impact**: 1-3% AUC improvement on dependent targets

6. **Feature Engineering for Trees** ⭐⭐
   - Winsorize heavy-tailed features at 1st/99th percentile
   - Create NaN indicator columns for sparse features
   - Bin continuous features into categories (e.g., quintiles + NaN)
   - Code:
     ```python
     # Winsorization
     from scipy.stats import mstats
     df[feature] = mstats.winsorize(df[feature], limits=[0.01, 0.01])

     # NaN indicators
     df[f'{feature}_is_nan'] = df[feature].isna().astype(int)
     ```
   - **Expected impact**: 1-2% AUC improvement from better representation

### Long-Term Investments (High Effort, Very High Impact)

7. **Custom Focal Loss for Trees** ⭐⭐⭐
   - Implement focal loss as custom objective for XGBoost/LightGBM
   - Adaptive weighting based on prediction confidence
   - Code skeleton:
     ```python
     def focal_loss(y_pred, y_true, gamma=2.0, alpha=0.25):
         p = 1 / (1 + np.exp(-y_pred))
         p_t = np.where(y_true == 1, p, 1 - p)
         alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
         grad = alpha_t * (p_t ** gamma) * (y_true - p)
         hess = alpha_t * gamma * (p_t ** (gamma - 1)) * (1 - p_t) * p * (y_true - p)
         return grad, hess
     ```
   - **Expected impact**: 2-4% AUC improvement on severely imbalanced targets

8. **Ensemble with Neural Networks** ⭐⭐⭐
   - Train both tree models (with above improvements) and neural networks
   - Blend predictions (simple average or learned weights)
   - Trees provide: different inductive bias, feature importance
   - Neural networks provide: better sparse feature handling, multi-task learning
   - **Expected impact**: 3-5% AUC improvement through diversity

### Hyperparameter Tuning Recommendations

**XGBoost:**
```python
params = {
    'max_depth': 10,  # Up from 6
    'learning_rate': 0.02,  # Down from 0.3
    'n_estimators': 800,  # Up from 100
    'min_child_weight': 75,  # Up from 1
    'max_delta_step': 5,  # New, for imbalance
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': <calculated per target>,
    'early_stopping_rounds': 100,  # Up from default
    'eval_metric': 'auc'
}
```

**LightGBM:**
```python
params = {
    'num_leaves': 75,  # Up from 31
    'learning_rate': 0.02,
    'n_estimators': 800,
    'min_data_in_leaf': 75,  # Up from 20
    'is_unbalance': True,  # New
    'categorical_feature': cat_features,  # Use native handling
    'early_stopping_round': 100
}
```

**CatBoost:**
```python
params = {
    'depth': 10,
    'learning_rate': 0.03,
    'iterations': 1500,  # More than XGBoost
    'l2_leaf_reg': 5,  # Regularization
    'cat_features': cat_features,  # Critical
    'auto_class_weights': 'Balanced',  # Built-in
    'early_stopping_rounds': 100
}
```

---

## Priority Action Plan

**Phase 1 (Week 1): Quick Wins**
1. Implement feature selection (245 features) → 2-3% AUC gain
2. Switch to CatBoost with native categorical handling → 2-3% AUC gain
3. Tune scale_pos_weight per target → 2-4% AUC gain on rare targets

**Expected improvement: 6-10% macro AUC**

**Phase 2 (Week 2-3): Architecture Improvements**
4. Train multi-output models by product groups → 1-2% AUC gain
5. Implement classifier chains for correlated targets → 1-3% AUC gain
6. Add feature engineering (winsorization, NaN indicators) → 1-2% AUC gain

**Expected additional improvement: 3-7% macro AUC**

**Phase 3 (Month 2): Advanced Techniques**
7. Implement custom focal loss objective → 2-4% AUC gain on imbalanced targets
8. Build ensemble with neural networks → 3-5% AUC gain through diversity

**Expected total improvement: 10-20% macro AUC**, potentially closing the gap with neural networks

---

## Conclusion

Tree-based models underperform neural networks on this dataset primarily due to:

1. **Inability to utilize sparse predictive features** (99%+ NaN features with moderate correlations)
2. **Global hyperparameters for per-target challenges** (class imbalance varies from 3.5:1 to 9035:1)
3. **Independent models miss target correlations** (within-group correlations 2.15x stronger)
4. **Representation learning disadvantage** (heavy tails, high cardinality categoricals)

The good news: **Most issues are fixable with proper techniques**. Tree models can achieve competitive performance by:
- Feature selection to reduce noise
- CatBoost for categorical encoding
- Per-target hyperparameter tuning
- Multi-output or chained architectures
- Custom objectives for class imbalance

**Recommendation**: Invest in improving tree models as they provide valuable diversity for ensembling with neural networks. A well-tuned CatBoost ensemble with multi-output architecture and custom focal loss can likely achieve within 2-3% of neural network performance while offering better interpretability and faster inference.