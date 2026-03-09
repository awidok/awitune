# Dataset Analysis: Optimal Training Strategies per Target Group

## Key Findings

- **Hardest target groups identified**: group_3, group_9, group_2 — these groups have extreme class imbalance (>100:1) and weak feature signals

- **Easiest target groups**: group_10, group_5, group_4 — these groups have moderate class imbalance and strong feature correlations

- **High focal gamma needed**: group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_9 — use gamma=2.5-3.0 due to extreme class imbalance

- **Groups with strong feature signal**:  — max feature correlation >0.10

- **Groups with weak feature signal**: group_1, group_2, group_3, group_7, group_8, group_9, group_10 — max feature correlation <0.05, require feature engineering

## Detailed Analysis

### 1. Group Characteristics

| Group | # Targets | Avg Positive Rate | Avg Imbalance | Max Imbalance |
|-------|-----------|-------------------|---------------|---------------|
| group_1 | 5 | 0.0126 | 202.4:1 | 543.5:1 |
| group_2 | 8 | 0.0060 | 1738.8:1 | 8909.9:1 |
| group_3 | 5 | 0.0401 | 416.5:1 | 856.1:1 |
| group_4 | 1 | 0.0081 | 121.9:1 | 121.9:1 |
| group_5 | 2 | 0.0060 | 248.2:1 | 390.6:1 |
| group_6 | 5 | 0.0061 | 472.6:1 | 1820.9:1 |
| group_7 | 3 | 0.0314 | 95.4:1 | 235.8:1 |
| group_8 | 3 | 0.0515 | 29.9:1 | 51.2:1 |
| group_9 | 8 | 0.0472 | 140.0:1 | 508.3:1 |
| group_10 | 1 | 0.3151 | 2.2:1 | 2.2:1 |

### 2. Group Difficulty Ranking

| Rank | Group | Difficulty Score | Imbalance Factor | Signal Factor | Variance Factor |
|------|-------|------------------|------------------|---------------|-----------------|
| 1 | group_3 | 0.665 | 0.874 | 0.820 | 0.029 |
| 2 | group_9 | 0.633 | 0.716 | 0.766 | 0.037 |
| 3 | group_2 | 0.632 | 1.080 | 0.716 | 0.064 |
| 4 | group_6 | 0.559 | 0.892 | 0.605 | 0.303 |
| 5 | group_7 | 0.558 | 0.661 | 0.802 | 0.021 |
| 6 | group_1 | 0.531 | 0.769 | 0.698 | 0.043 |
| 7 | group_8 | 0.508 | 0.497 | 0.752 | 0.020 |
| 8 | group_4 | 0.437 | 0.696 | 0.570 | 0.000 |
| 9 | group_5 | 0.425 | 0.799 | 0.451 | 0.000 |
| 10 | group_10 | 0.371 | 0.167 | 0.803 | 0.000 |

### 3. Focal Loss Gamma Recommendations

| Group | Recommended Gamma | Rationale | Avg Positive Rate | Avg Imbalance |
|-------|-------------------|-----------|-------------------|---------------|
| group_1 | 3.0 | Extreme class imbalance (>100:1) | 0.0126 | 202.4:1 |
| group_2 | 3.0 | Extreme class imbalance (>100:1) | 0.0060 | 1738.8:1 |
| group_3 | 3.0 | Extreme class imbalance (>100:1) | 0.0401 | 416.5:1 |
| group_4 | 3.0 | Extreme class imbalance (>100:1) | 0.0081 | 121.9:1 |
| group_5 | 3.0 | Extreme class imbalance (>100:1) | 0.0060 | 248.2:1 |
| group_6 | 3.0 | Extreme class imbalance (>100:1) | 0.0061 | 472.6:1 |
| group_7 | 2.5 | High class imbalance (50-100:1) | 0.0314 | 95.4:1 |
| group_8 | 2.0 | Moderate class imbalance (20-50:1) | 0.0515 | 29.9:1 |
| group_9 | 3.0 | Extreme class imbalance (>100:1) | 0.0472 | 140.0:1 |
| group_10 | 1.0 | Balanced classes (<10:1) | 0.3151 | 2.2:1 |

### 4. Top Features by Group

#### group_1

- **Max correlation**: 0.0454
- **Features with |r|>0.01**: 127
- **Top 5 features**:

  1. `num_feature_993`: r=0.0454
  2. `num_feature_1981`: r=0.0380
  3. `num_feature_1773`: r=0.0308
  4. `num_feature_2018`: r=0.0253
  5. `num_feature_555`: r=0.0251

#### group_2

- **Max correlation**: 0.0426
- **Features with |r|>0.01**: 143
- **Top 5 features**:

  1. `num_feature_2068`: r=0.0426
  2. `num_feature_2037`: r=0.0391
  3. `num_feature_2274`: r=0.0356
  4. `num_feature_1929`: r=0.0336
  5. `num_feature_77`: r=0.0334

#### group_3

- **Max correlation**: 0.0270
- **Features with |r|>0.01**: 142
- **Top 5 features**:

  1. `num_feature_2219`: r=0.0270
  2. `num_feature_1848`: r=0.0265
  3. `num_feature_1441`: r=0.0245
  4. `num_feature_1905`: r=0.0244
  5. `num_feature_539`: r=0.0230

#### group_4

- **Max correlation**: 0.0645
- **Features with |r|>0.01**: 111
- **Top 5 features**:

  1. `num_feature_1223`: r=0.0645
  2. `num_feature_2073`: r=0.0396
  3. `num_feature_2031`: r=0.0396
  4. `num_feature_1580`: r=0.0342
  5. `num_feature_1465`: r=0.0313

#### group_5

- **Max correlation**: 0.0823
- **Features with |r|>0.01**: 115
- **Top 5 features**:

  1. `num_feature_71`: r=0.0823
  2. `num_feature_2068`: r=0.0411
  3. `num_feature_886`: r=0.0377
  4. `num_feature_1398`: r=0.0349
  5. `num_feature_1540`: r=0.0319

#### group_6

- **Max correlation**: 0.0593
- **Features with |r|>0.01**: 111
- **Top 5 features**:

  1. `num_feature_1041`: r=0.0593
  2. `num_feature_2163`: r=0.0394
  3. `num_feature_1270`: r=0.0381
  4. `num_feature_1767`: r=0.0345
  5. `num_feature_455`: r=0.0335

#### group_7

- **Max correlation**: 0.0296
- **Features with |r|>0.01**: 150
- **Top 5 features**:

  1. `num_feature_555`: r=0.0296
  2. `num_feature_1147`: r=0.0288
  3. `num_feature_1159`: r=0.0254
  4. `num_feature_1803`: r=0.0237
  5. `num_feature_697`: r=0.0233

#### group_8

- **Max correlation**: 0.0372
- **Features with |r|>0.01**: 127
- **Top 5 features**:

  1. `num_feature_2037`: r=0.0372
  2. `num_feature_2219`: r=0.0348
  3. `num_feature_1368`: r=0.0296
  4. `num_feature_1929`: r=0.0289
  5. `num_feature_441`: r=0.0280

#### group_9

- **Max correlation**: 0.0350
- **Features with |r|>0.01**: 112
- **Top 5 features**:

  1. `num_feature_1237`: r=0.0350
  2. `num_feature_299`: r=0.0241
  3. `num_feature_1483`: r=0.0224
  4. `num_feature_674`: r=0.0224
  5. `num_feature_2019`: r=0.0220

#### group_10

- **Max correlation**: 0.0296
- **Features with |r|>0.01**: 117
- **Top 5 features**:

  1. `num_feature_1237`: r=0.0296
  2. `num_feature_1785`: r=0.0294
  3. `num_feature_1328`: r=0.0292
  4. `num_feature_1483`: r=0.0290
  5. `num_feature_368`: r=0.0287

### 5. Within-Group Target Correlations

| Group | Mean Correlation | Max Correlation | Strongly Correlated Pairs (>0.3) |
|-------|------------------|-----------------|----------------------------------|
| group_1 | 0.0137 | 0.0665 | 0 |
| group_2 | 0.0146 | 0.1573 | 0 |
| group_3 | 0.0162 | 0.0504 | 0 |
| group_4 | 0.0000 | 0.0000 | 0 |
| group_5 | 0.5169 | 0.5169 | 1 |
| group_6 | 0.1254 | 0.5157 | 1 |
| group_7 | 0.0460 | 0.0548 | 0 |
| group_8 | -0.0182 | -0.0073 | 0 |
| group_9 | 0.0043 | 0.0509 | 0 |
| group_10 | 0.0000 | 0.0000 | 0 |

### 6. Cross-Group Correlations

No groups show strong cross-group correlations (>0.5). Each group should be modeled independently.

## Recommendations for Model Builders

### 1. Use Group Specific Focal Loss

**Rationale**: Different groups have different class imbalance levels

**Implementation**: Use different focal gamma per group based on recommendations

**Recommended gamma values**:
```python
FOCAL_GAMMA = {
    'group_1': 3.0,
    'group_2': 3.0,
    'group_3': 3.0,
    'group_4': 3.0,
    'group_5': 3.0,
    'group_6': 3.0,
    'group_7': 2.5,
    'group_8': 2.0,
    'group_9': 3.0,
    'group_10': 1.0,
}
```

**Expected impact**: Reduce false negatives on minority class for high-imbalance groups

### 2. Specialized Treatment For Hard Groups

**Rationale**: Some groups are significantly harder due to class imbalance and weak features

**Strategies**:

- Use higher focal gamma for extreme imbalance
- Apply oversampling or SMOTE for minority class
- Use ensemble of models with different random seeds
- Feature engineering to create interaction features

**Hard groups**: group_1, group_2, group_3, group_6, group_7, group_8, group_9

**Expected impact**: Improve AUC on hard groups by 0.02-0.05

### 3. Group Specific Feature Selection

**Rationale**: Different groups have different important features

**Implementation**: Use group-specific feature importance for feature selection

**Expected impact**: Reduce overfitting and improve model interpretability

### 4. Adaptive Architecture By Group Size

**Rationale**: Single-target groups can use dedicated models while multi-target groups benefit from multi-task learning

**Single-target groups**: group_4, group_10
**Multi-target groups**: group_1, group_2, group_3, group_5, group_6, group_7, group_8, group_9

**Expected impact**: Balance computational cost with performance gains
