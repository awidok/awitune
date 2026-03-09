# Dataset Analysis: Correlation Structure Analysis for Data Fusion 2026

## Key Findings

- **5 feature clusters** with high internal correlation (|r| > 0.8) identified among 132 main numeric features
- **38 highly correlated feature pairs** found, indicating significant redundancy
- **137 categorical feature pairs** show high association (Cramer's V > 0.5)
- **190 NaN co-occurrence patterns** identified among high-NaN features
- **0 features have VIF > 10**, indicating potential multicollinearity issues
- **20 natural feature groups** identified through hierarchical clustering

## Detailed Analysis

### 1. Feature Correlation Matrix (Main Numeric Features)

Computed correlation matrix for all 132 main numeric features.

**Highly Correlated Clusters**: 5 clusters with |r| > 0.8

| Cluster ID | Size | Avg Internal Correlation |
|------------|------|--------------------------|
| 0 | 9 | nan |
| 1 | 8 | 0.934 |
| 2 | 2 | 0.996 |
| 3 | 2 | 0.911 |
| 4 | 2 | 0.970 |

### 2. Categorical Feature Correlations (Cramer's V)

Analyzed 30 categorical features using Cramer's V statistic.

**High Association Pairs**: 137 pairs with Cramer's V > 0.5

**Top 10 Categorical Associations**:

1. cat_feature_2 ↔ cat_feature_25: V = 1.000
2. cat_feature_13 ↔ cat_feature_26: V = 1.000
3. cat_feature_20 ↔ cat_feature_24: V = 1.000
4. cat_feature_3 ↔ cat_feature_14: V = 1.000
5. cat_feature_3 ↔ cat_feature_29: V = 1.000
6. cat_feature_5 ↔ cat_feature_14: V = 1.000
7. cat_feature_5 ↔ cat_feature_29: V = 1.000
8. cat_feature_7 ↔ cat_feature_14: V = 1.000
9. cat_feature_7 ↔ cat_feature_29: V = 1.000
10. cat_feature_8 ↔ cat_feature_14: V = 1.000

### 3. Feature-Target Correlations

Computed correlations between main numeric features and first 20 targets.

**Top Features per Target** (first 5 targets):


**target_1_1**:
  1. num_feature_75: r = 0.0961
  2. num_feature_67: r = 0.0953
  3. num_feature_112: r = 0.0872
  4. num_feature_34: r = 0.0700
  5. num_feature_81: r = 0.0601

**target_1_2**:
  1. num_feature_43: r = 0.0980
  2. num_feature_34: r = -0.0425
  3. num_feature_112: r = 0.0319
  4. num_feature_118: r = -0.0205
  5. num_feature_64: r = -0.0197

**target_1_3**:
  1. num_feature_132: r = 0.1837
  2. num_feature_2: r = 0.1371
  3. num_feature_15: r = 0.0874
  4. num_feature_8: r = 0.0868
  5. num_feature_104: r = 0.0751

**target_1_4**:
  1. num_feature_64: r = 0.3761
  2. num_feature_43: r = 0.2883
  3. num_feature_131: r = 0.1014
  4. num_feature_34: r = 0.0800
  5. num_feature_81: r = -0.0734

**target_1_5**:
  1. num_feature_45: r = 0.0839
  2. num_feature_107: r = 0.0219
  3. num_feature_28: r = -0.0215
  4. num_feature_15: r = 0.0197
  5. num_feature_100: r = -0.0176

### 4. NaN Pattern Analysis

**Features with High NaN Rates**:

| Feature | NaN Rate |
|---------|----------|
| num_feature_43 | 99.88% |
| num_feature_54 | 99.85% |
| num_feature_64 | 99.85% |
| num_feature_34 | 99.76% |
| num_feature_118 | 99.65% |
| num_feature_74 | 99.59% |
| num_feature_81 | 99.55% |
| num_feature_75 | 98.99% |
| num_feature_45 | 98.85% |
| num_feature_28 | 98.39% |

**NaN Co-occurrence**: 190 feature pairs have Jaccard similarity > 0.5

**NaN Pattern Correlations with Targets**:

**target_1_1**:
  - num_feature_81: r = -0.1995 (NaN rate: 99.55%)
  - num_feature_4: r = -0.1253 (NaN rate: 95.67%)

**target_1_3**:
  - num_feature_115: r = -0.1282 (NaN rate: 63.55%)
  - num_feature_84: r = -0.1282 (NaN rate: 63.55%)
  - num_feature_89: r = -0.1282 (NaN rate: 63.55%)

### 5. Feature Redundancy Analysis (VIF)

Computed Variance Inflation Factor for 50 low-NaN features.

**Features with Highest VIF** (potential redundancy):

| Feature | VIF |
|---------|-----|
| num_feature_80 | 1.00 |
| num_feature_73 | 1.00 |
| num_feature_6 | 1.00 |
| num_feature_5 | 1.00 |
| num_feature_32 | 1.00 |
| num_feature_31 | 1.00 |
| num_feature_14 | 1.00 |
| num_feature_7 | 1.00 |
| num_feature_17 | 1.00 |
| num_feature_20 | 1.00 |

**Recommendation**: 0 features have VIF > 10 and may be redundant

### 6. Natural Feature Groupings (Hierarchical Clustering)

Performed hierarchical clustering on the correlation matrix to identify natural feature groups.

**Identified 20 feature groups**:

- **Group 1**: 2 features - num_feature_30, num_feature_98
- **Group 2**: 112 features
- **Group 3**: 1 features - num_feature_21
- **Group 4**: 1 features - num_feature_123
- **Group 5**: 1 features - num_feature_20
- **Group 6**: 1 features - num_feature_49
- **Group 7**: 1 features - num_feature_93
- **Group 8**: 1 features - num_feature_113
- **Group 9**: 1 features - num_feature_122
- **Group 10**: 1 features - num_feature_89
- **Group 11**: 1 features - num_feature_101
- **Group 12**: 1 features - num_feature_9
- **Group 13**: 1 features - num_feature_14
- **Group 14**: 1 features - num_feature_22
- **Group 15**: 1 features - num_feature_32
- **Group 16**: 1 features - num_feature_54
- **Group 17**: 1 features - num_feature_70
- **Group 18**: 1 features - num_feature_80
- **Group 19**: 1 features - num_feature_106
- **Group 20**: 1 features - num_feature_115

### Target-Feature Group Alignment

**Most Important Feature Group per Target**:

| Target | Best Feature Group | Avg |r| |
|--------|-------------------|----------|
| target_1_1 | Group 7 | 0.0161 |
| target_1_2 | Group 2 | 0.0076 |
| target_1_3 | Group 2 | 0.0204 |
| target_1_4 | Group 2 | 0.0221 |
| target_1_5 | Group 2 | 0.0057 |
| target_2_1 | Group 2 | 0.0100 |
| target_2_2 | Group 2 | 0.0214 |
| target_2_3 | Group 4 | 0.0082 |
| target_2_4 | Group 2 | 0.0109 |
| target_2_5 | Group 5 | 0.0165 |
| target_2_6 | Group 2 | 0.0065 |
| target_2_7 | Group 2 | 0.0051 |
| target_2_8 | Group 3 | 0.0319 |
| target_3_1 | Group 2 | 0.0121 |
| target_3_2 | Group 2 | 0.0233 |
| target_3_3 | Group 6 | 0.0113 |
| target_3_4 | Group 2 | 0.0073 |
| target_3_5 | Group 2 | 0.0222 |
| target_4_1 | Group 4 | 0.0259 |
| target_5_1 | Group 5 | 0.0169 |

## Recommendations for Model Builders

### 1. Feature Pruning
- **Remove/combine highly correlated features**: 5 clusters identified. Consider keeping only one representative feature per cluster.
- **Address multicollinearity**: 0 features have VIF > 10. Consider:
  - Removing redundant features
  - Using PCA for dimensionality reduction
  - Regularization techniques (L1/L2)

### 2. NaN Pattern Engineering
- **Create NaN indicator features**: For high-NaN features that correlate with targets, create binary NaN indicators
- **Group NaN patterns**: Features with high NaN co-occurrence may represent missing data patterns - consider creating meta-features
- **Imputation strategy**: Use group-aware imputation or model-based imputation rather than simple mean/median

### 3. Feature Grouping Strategy
- **Group-specific processing**: 20 natural feature groups identified. Consider:
  - Separate feature transformers per group
  - Attention mechanisms to weight feature groups
  - Group-specific neural network branches
- **Target-aware feature selection**: Different targets benefit from different feature groups - consider target-specific feature subsets

### 4. Categorical Feature Handling
- **High-association categorical pairs**: 137 pairs with Cramer's V > 0.5
- Consider interaction features between highly associated categorical features
- Use target encoding or embedding approaches for categorical features

### 5. Correlation-Based Feature Engineering
- **Create interaction features** from highly correlated numeric pairs
- **Ratio features** for features that move together
- **Difference features** for features that are highly correlated but may contain unique information

**Sample Size**: Analysis performed on 100,000 rows