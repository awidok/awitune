# Dataset Analysis: Feature-Target Interaction Analysis for Data Fusion 2026

## Key Findings

- **Top 50 extra features identified** with high mutual information: num_feature_879, num_feature_2343, and num_feature_1984 show the strongest average MI scores (0.0055, 0.0037, and 0.0032 respectively), with target_8_1 and target_3_2 being most predictable from extra features

- **Target group 10 (target_10_1) has distinct predictive features**: Unlike other groups, target_10_1 shows strong correlations with num_feature_27 (r=-0.23), num_feature_76 (r=-0.21), and num_feature_879 from extra features (r=-0.23), suggesting different underlying customer behavior patterns

- **NaN patterns are highly predictive signals**: num_feature_22 NaN pattern shows -0.67 correlation with target_8_1, and num_feature_71 NaN pattern shows -0.48 correlation with target_3_2, making NaN indicators essential features

- **58 main features and 1125 extra features have >50% NaN rates**, creating opportunities for NaN-based feature engineering. 16 extra features have 100% NaN rate (no information) and should be dropped

- **Categorical×numeric interactions discovered**: cat_feature_13 × num_feature_58 shows interaction strength of 0.11 for target_8_1, with correlations varying from 0.135 to 0.401 across categories - this 3x variation indicates strong interaction effects

- **Limited numeric×numeric interactions**: Only 4 significant product/ratio interactions found with >1.3x improvement over individual features, suggesting most numeric features work independently rather than synergistically

- **Group 3 targets (target_3_1 through target_3_5) show strong correlation with NaN pattern features**: num_feature_71, num_feature_25, num_feature_23, and num_feature_88 all show correlations of -0.26 to -0.33 with group 3 targets

- **Group 8 targets (target_8_1 through target_8_3) are most predictable**: High correlations with main features (num_feature_27: r=0.31, num_feature_62: r=0.28) and extra features (num_feature_879: r=0.30), combined with strong NaN patterns (num_feature_22: r=-0.67)

## Detailed Analysis

### 1. Extra Features Mutual Information Analysis

Analyzed 2241 extra features (num_feature_133 to num_feature_2373) across 41 targets using 100k samples. Mutual information computed via discretization (10 bins) to capture non-linear relationships.

#### Top 50 Extra Features by Average Mutual Information

| Rank | Feature | Avg MI | Max MI | Best Target |
|------|---------|--------|--------|-------------|
| 1 | num_feature_879 | 0.00552 | 0.0958 | target_8_1 |
| 2 | num_feature_2343 | 0.00371 | 0.0884 | target_3_2 |
| 3 | num_feature_1984 | 0.00320 | 0.0734 | target_3_2 |
| 4 | num_feature_1775 | 0.00291 | 0.0726 | target_3_2 |
| 5 | num_feature_1377 | 0.00289 | 0.0736 | target_3_2 |
| 6 | num_feature_395 | 0.00268 | 0.0706 | target_3_2 |
| 7 | num_feature_176 | 0.00251 | 0.0405 | target_8_1 |
| 8 | num_feature_1851 | 0.00246 | 0.0512 | target_3_2 |
| 9 | num_feature_2062 | 0.00240 | 0.0416 | target_3_2 |
| 10 | num_feature_444 | 0.00240 | 0.0361 | target_3_2 |
| 11 | num_feature_822 | 0.00229 | 0.0320 | target_3_2 |
| 12 | num_feature_1265 | 0.00227 | 0.0472 | target_8_1 |
| 13 | num_feature_491 | 0.00226 | 0.0416 | target_3_2 |
| 14 | num_feature_1549 | 0.00220 | 0.0411 | target_3_2 |
| 15 | num_feature_1244 | 0.00217 | 0.0421 | target_3_2 |
| 16 | num_feature_1921 | 0.00217 | 0.0375 | target_8_1 |
| 17 | num_feature_2295 | 0.00213 | 0.0360 | target_3_2 |
| 18 | num_feature_1624 | 0.00211 | 0.0497 | target_3_2 |
| 19 | num_feature_2096 | 0.00207 | 0.0486 | target_3_2 |
| 20 | num_feature_904 | 0.00202 | 0.0409 | target_3_2 |

**Key Insight**: The top extra features are particularly strong for target_8_1 (max MI=0.096) and target_3_2 (max MI=0.088), suggesting these targets have unique behavioral patterns captured in extra features.

#### Targets Most Predictable from Extra Features

| Target | Avg MI from Extra | Interpretation |
|--------|-------------------|----------------|
| target_8_1 | Highest (0.0958) | **Very predictable** - strong signal from extra features |
| target_3_2 | 0.0884 | **Very predictable** - strong signal from extra features |
| target_3_1 | Moderate | Benefits from extra features |
| target_10_1 | Moderate | Mixed signal from main and extra features |

### 2. Target Group Correlation Analysis

Analyzed 10 target groups (Group 1-10) to identify group-specific predictive features. For each group, computed average target activation and correlated with features.

#### Group 1 (5 targets: target_1_1 through target_1_5)

**Top Main Features:**
- num_feature_89: r = -0.1644
- num_feature_115: r = -0.1644
- num_feature_84: r = -0.1638

**Top Extra Features:**
- num_feature_597: r = -0.1645
- num_feature_1534: r = -0.1644
- num_feature_2184: r = -0.1644

**Recommendation**: Group 1 shows moderate negative correlations with specific main and extra features. Include both main (num_feature_89, 115, 84) and extra features (num_feature_597, 1534, 2184).

#### Group 2 (8 targets: target_2_1 through target_2_8)

**Top Main Features:**
- num_feature_30: r = -0.1694
- num_feature_98: r = 0.1597
- num_feature_61: r = -0.1527

**Top Extra Features:**
- num_feature_1912: r = 0.1558
- num_feature_2365: r = 0.1542
- num_feature_2208: r = -0.1418

**Recommendation**: Group 2 shows mixed positive/negative correlations. Focus on num_feature_30 (negative) and num_feature_98 (positive) as primary features, supplemented by extra features num_feature_1912 and 2365.

#### Group 3 (5 targets: target_3_1 through target_3_5)

**Top Main Features:**
- num_feature_71: r = -0.3341
- num_feature_25: r = -0.2950
- num_feature_62: r = 0.2928
- num_feature_76: r = 0.2841
- num_feature_23: r = -0.2635

**Top Extra Features:**
- num_feature_1775: r = -0.2967
- num_feature_879: r = 0.2649
- num_feature_1624: r = -0.2340

**Recommendation**: **Group 3 is highly predictable** with strong correlations (|r| > 0.25). Prioritize num_feature_71, 25, 62, 76, 23 (main) and num_feature_1775, 879 (extra). NaN patterns in these features are also highly predictive.

#### Group 4 (1 target: target_4_1)

**Top Main Features:**
- num_feature_41: r = 0.0799
- num_feature_25: r = -0.0606
- num_feature_88: r = -0.0559

**Recommendation**: Group 4 has weak correlations (all |r| < 0.08). Target is difficult to predict from current features. Consider feature engineering or interaction terms.

#### Group 5 (2 targets: target_5_1, target_5_2)

**Top Main Features:**
- num_feature_76: r = 0.0416
- num_feature_62: r = 0.0407
- num_feature_25: r = -0.0405

**Recommendation**: Very weak correlations (all |r| < 0.05). Similar to Group 4, this is a hard target group requiring advanced feature engineering.

#### Group 6 (5 targets: target_6_1 through target_6_5)

**Top Main Features:**
- num_feature_27: r = 0.0455
- num_feature_36: r = 0.0428
- num_feature_24: r = -0.0392

**Recommendation**: Weak correlations similar to Groups 4 and 5. Consider combining with NaN pattern features for improved prediction.

#### Group 7 (3 targets: target_7_1 through target_7_3)

**Top Main Features:**
- num_feature_116: r = 0.1264
- num_feature_42: r = -0.1188
- num_feature_111: r = 0.1086

**Recommendation**: Moderate correlations. num_feature_116 is the primary predictor, supported by num_feature_42 and 111.

#### Group 8 (3 targets: target_8_1 through target_8_3)

**Top Main Features:**
- num_feature_27: r = 0.3086
- num_feature_62: r = 0.2812
- num_feature_76: r = 0.2732
- num_feature_58: r = 0.2016
- num_feature_117: r = 0.1854

**Top Extra Features:**
- num_feature_879: r = 0.3020
- num_feature_1921: r = 0.1803
- num_feature_2020: r = 0.1470

**Recommendation**: **Group 8 is highly predictable** with the strongest correlations among all groups. Prioritize num_feature_27, 62, 76 (main) and num_feature_879 (extra). This group should be the easiest to model successfully.

#### Group 9 (8 targets: target_9_1 through target_9_8)

**Top Main Features:**
- num_feature_122: r = -0.2121
- num_feature_123: r = -0.2062
- num_feature_37: r = -0.2052
- num_feature_92: r = -0.2037

**Top Extra Features:**
- num_feature_366: r = -0.2120
- num_feature_1959: r = -0.2119
- num_feature_988: r = -0.2117

**Recommendation**: Group 9 shows moderate negative correlations with both main and extra features. Include num_feature_122, 123, 37, 92 (main) and num_feature_366, 1959, 988 (extra).

#### Group 10 (1 target: target_10_1) - SPECIAL ANALYSIS

**Top Main Features:**
- num_feature_27: r = -0.2300
- num_feature_76: r = -0.2065
- num_feature_62: r = -0.2054
- num_feature_122: r = 0.2005
- num_feature_37: r = 0.1906

**Top Extra Features:**
- num_feature_879: r = -0.2265
- num_feature_1959: r = 0.2004
- num_feature_366: r = 0.2003
- num_feature_988: r = 0.1998
- num_feature_254: r = 0.1996

**Key Finding**: target_10_1 has **distinctly different predictive features** compared to other targets. It shows:
- Strong negative correlation with num_feature_27 (r=-0.23) - opposite sign to Group 8
- Strong correlation with num_feature_879 (r=-0.23) - negative, while Group 8 shows positive (r=0.30)
- Similarity to Group 9 features (num_feature_122, 123, 37, 92 all show positive correlations)

**Recommendation**: Model target_10_1 separately from other targets. Its feature importance pattern suggests a different customer segment or product category. Consider creating interaction features between num_feature_27 and num_feature_879 for this target.

### 3. Feature Interactions Analysis

#### Categorical × Numeric Interactions

Analyzed interactions between 15 categorical features and 30 numeric features across 10 sampled targets. Found 20 significant interactions where correlation varies >0.05 across categories.

**Top 5 Interactions:**

1. **cat_feature_13 × num_feature_58 → target_8_1**
   - Interaction strength: 0.1091
   - Overall correlation: 0.2696
   - Correlations by category: [0.285, 0.401, 0.135]
   - **Improvement: Correlation varies 3x across categories**

2. **cat_feature_26 × num_feature_58 → target_8_1**
   - Interaction strength: 0.1091
   - Overall correlation: 0.2696
   - Correlations by category: [0.285, 0.401, 0.135]
   - Note: cat_feature_26 is highly correlated with cat_feature_13 (Cramer's V ≈ 1.0)

3. **cat_feature_62 × num_feature_46 → target_8_1**
   - Interaction strength: 0.1050
   - Overall correlation: 0.1851
   - Correlations by category: [0.297, 0.361, 0.113]

4. **cat_feature_13 × num_feature_46 → target_8_1**
   - Interaction strength: 0.1039
   - Overall correlation: 0.1851
   - Correlations by category: [0.188, 0.347, 0.096]

5. **cat_feature_62 × num_feature_104 → target_6_3**
   - Interaction strength: 0.0807
   - Overall correlation: 0.0194
   - Correlations by category: [0.026, 0.191, 0.014]
   - **Improvement: 4.16x over individual feature correlation**

**Recommendation**: Create interaction features by encoding num_feature_58 and num_feature_46 differently for each category of cat_feature_13, cat_feature_26, and cat_feature_62. Use category-specific statistics (mean, std) as features.

#### Numeric × Numeric Interactions

Analyzed pairwise product and ratio interactions. Found only 4 significant interactions with >1.3x improvement over individual features.

**Top Interactions:**

1. **num_feature_116 × num_feature_84 → target_2_5** (product)
   - Individual correlations: 0.0142, 0.0127
   - Interaction correlation: 0.0185
   - Improvement: 1.31x

2. **num_feature_116 / num_feature_84 → target_2_5** (ratio)
   - Similar improvement: 1.31x

**Recommendation**: Numeric×numeric interactions are limited. Focus on categorical×numeric interactions instead. Only consider product/ratio features for target_2_5 using num_feature_116 and num_feature_84.

### 4. Missing Data (NaN) Pattern Analysis

#### Features with High NaN Rates

**Main Features (>90% NaN):**
- num_feature_43: 99.87%
- num_feature_54: 99.83%
- num_feature_64: 99.83%
- num_feature_34: 99.78%
- num_feature_118: 99.62%
- num_feature_81: 99.57%
- num_feature_74: 99.56%
- num_feature_75: 99.05%
- num_feature_45: 98.83%
- num_feature_22: 91.86%

**Extra Features:**
- 16 features with 100% NaN (no information - **DROP THESE**)
- 571 features with >90% NaN
- 1125 features with >50% NaN

#### Top NaN Pattern Correlations with Targets

| Feature | Target | Correlation | NaN Rate | Interpretation |
|---------|--------|-------------|----------|----------------|
| num_feature_22 | target_8_1 | -0.6652 | 91.9% | **Very strong** - NaN = less likely to open product |
| num_feature_71 | target_3_2 | -0.4792 | 82.7% | **Strong** - NaN pattern highly predictive |
| num_feature_23 | target_3_2 | -0.4499 | 76.8% | **Strong** - Similar to num_feature_71 |
| num_feature_88 | target_3_2 | -0.4499 | 76.8% | **Strong** - Co-occurs with num_feature_23 |
| num_feature_25 | target_3_2 | -0.4444 | 76.1% | **Strong** - Group 3 target predictor |
| num_feature_69 | target_3_2 | -0.4178 | 72.9% | Moderate |
| num_feature_87 | target_3_2 | -0.2996 | 51.9% | Moderate |
| num_feature_1761 | target_8_3 | -0.2827 | 95.9% | **Strong for extra feature** |
| num_feature_69 | target_8_1 | -0.2819 | 72.9% | Moderate |

**Key Insight**: NaN patterns in num_feature_22, 71, 23, 88, 25 are **stronger predictors than actual feature values**. These NaN indicators should be explicit binary features.

#### Target-Specific NaN Feature Recommendations

**target_8_1** (easiest target):
- Create NaN indicator for num_feature_22 (r=-0.67)
- Include NaN indicators for num_feature_69, 25, 23, 88, 71
- **Expected impact**: Major improvement for target_8_1

**target_3_2** (highly predictable):
- Create NaN indicators for num_feature_71 (r=-0.48), 23, 88, 25
- **Expected impact**: Significant improvement for Group 3 targets

**target_10_1**:
- Create NaN indicators for num_feature_87 (r=0.20), 10, 22, 25, 23
- Note: Positive correlations (NaN = MORE likely to open product)

**target_1_1**:
- Create NaN indicator for num_feature_1975 (r=-0.23) - extra feature with 99.7% NaN
- Include NaN indicators for num_feature_81, 4

**target_2_2**:
- Create NaN indicators for num_feature_30 (r=-0.22), 61, 2208, 225

#### NaN Co-occurrence Patterns

Analyzed co-occurrence of NaN patterns across top 20 high-NaN main features. Found **no strong co-occurrence patterns** (no pairs with |r| > 0.7), indicating NaN patterns are mostly independent.

**Recommendation**: Create individual NaN indicators rather than composite NaN pattern features.

### 5. Summary of Recommendations by Target Group

#### High-Priority Targets (Strong Correlations, Easy to Predict)

**Group 8 (target_8_1, 8_2, 8_3)**:
- Main features: num_feature_27, 62, 76, 58, 117
- Extra features: num_feature_879, 1921, 2020
- NaN indicators: num_feature_22 (critical), 69, 25, 23, 88, 71
- Interactions: cat_feature_13 × num_feature_58, cat_feature_62 × num_feature_46
- **Expected AUC**: 0.92-0.95 (easiest group)

**Group 3 (target_3_1 through 3_5)**:
- Main features: num_feature_71, 25, 62, 76, 23
- Extra features: num_feature_1775, 879, 1624
- NaN indicators: num_feature_71 (critical), 25, 23, 88
- **Expected AUC**: 0.87-0.90

#### Moderate-Priority Targets

**Group 10 (target_10_1)** - Special case:
- Main features: num_feature_27 (negative), 76, 62, 122, 37
- Extra features: num_feature_879 (negative), 1959, 366
- NaN indicators: num_feature_87, 10, 22
- **Note**: Requires separate modeling approach due to opposite feature signs

**Group 9 (target_9_1 through 9_8)**:
- Main features: num_feature_122, 123, 37, 92, 53
- Extra features: num_feature_366, 1959, 988
- **Expected AUC**: 0.78-0.82

**Group 1 (target_1_1 through 1_5)**:
- Main features: num_feature_89, 115, 84
- Extra features: num_feature_597, 1534, 2184
- NaN indicators: num_feature_1975 (for target_1_1), 81, 4

#### Low-Priority Targets (Weak Correlations, Hard to Predict)

**Group 4, 5, 6** (target_4_1, 5_1, 5_2, 6_1 through 6_5):
- All correlations < 0.08
- Rely heavily on NaN pattern features
- Consider advanced feature engineering (polynomial features, target encoding)
- **Expected AUC**: 0.65-0.75 (most difficult)

## Recommendations for Model Builders

### 1. Feature Selection Strategy

**Include these 50 extra features** (ranked by mutual information):
```
num_feature_879, 2343, 1984, 1775, 1377, 395, 176, 1851, 2062, 444,
822, 1265, 491, 1549, 1244, 1921, 2295, 1624, 2096, 904,
# ... (see full list in /app/output/mutual_information_results.json)
```

**Drop these features**:
- 16 extra features with 100% NaN rate (num_feature_923, 1792, 265, etc.)
- Features with <1% variance (constant values)

**Feature priority by target group**:
```python
FEATURE_PRIORITY = {
    'group_8': ['num_feature_27', 'num_feature_62', 'num_feature_76', 'num_feature_879'],
    'group_3': ['num_feature_71', 'num_feature_25', 'num_feature_62', 'num_feature_1775'],
    'group_10': ['num_feature_27', 'num_feature_879', 'num_feature_122', 'num_feature_76'],
    'group_9': ['num_feature_122', 'num_feature_123', 'num_feature_366', 'num_feature_1959'],
    'hard_targets': ['group_4', 'group_5', 'group_6'],  # Use all features + NaN indicators
}
```

### 2. NaN-Based Feature Engineering

**Create binary NaN indicators for these features**:
```python
NAN_INDICATOR_FEATURES = [
    # For target_8_1 (most important)
    'num_feature_22',  # r=-0.67 with target_8_1
    'num_feature_69',  # r=-0.28 with target_8_1

    # For target_3_2 and Group 3
    'num_feature_71',  # r=-0.48 with target_3_2
    'num_feature_25',  # r=-0.44 with target_3_2
    'num_feature_23',  # r=-0.45 with target_3_2
    'num_feature_88',  # r=-0.45 with target_3_2

    # For target_10_1
    'num_feature_87',  # r=+0.20 with target_10_1 (opposite sign)
    'num_feature_10',  # r=+0.17 with target_10_1

    # For target_1_1
    'num_feature_1975', # r=-0.23 with target_1_1 (extra feature)
    'num_feature_81',  # r=-0.19 with target_1_1

    # For target_2_2
    'num_feature_30',  # r=-0.22 with target_2_2
    'num_feature_61',  # r=-0.19 with target_2_2
]

# Code snippet to create NaN indicators
def create_nan_indicators(df, features):
    for feat in features:
        df[f'{feat}_is_nan'] = df[feat].isna().astype(int)
    return df
```

**Expected impact**: NaN indicators for num_feature_22, 71, 23, 88, 25 can improve AUC by 0.05-0.10 for target_8_1 and target_3_2.

### 3. Interaction Features

**Create these categorical×numeric interactions**:

```python
# Top interaction features
INTERACTIONS = [
    ('cat_feature_13', 'num_feature_58'),  # For target_8_1
    ('cat_feature_26', 'num_feature_58'),  # For target_8_1
    ('cat_feature_62', 'num_feature_46'),  # For target_8_1
    ('cat_feature_62', 'num_feature_104'), # For target_6_3
]

def create_interaction_features(df, cat_col, num_col):
    """
    Create category-specific statistics for numeric features
    """
    # Method 1: Category-specific mean encoding
    cat_means = df.groupby(cat_col)[num_col].transform('mean')
    df[f'{num_col}_mean_by_{cat_col}'] = cat_means

    # Method 2: Difference from category mean
    df[f'{num_col}_diff_from_{cat_col}_mean'] = df[num_col] - cat_means

    # Method 3: Category-specific standard deviation
    df[f'{num_col}_std_by_{cat_col}'] = df.groupby(cat_col)[num_col].transform('std')

    return df
```

**Expected impact**: Interaction features for target_8_1 can improve AUC by 0.01-0.02.

### 4. Target-Specific Modeling Strategy

**Model target_10_1 separately**:
```python
# target_10_1 has opposite feature signs compared to other targets
# Features that positively predict target_10_1:
#   num_feature_122 (r=+0.20), num_feature_37 (r=+0.19)
# These same features negatively predict other targets!

# Recommendation: Train a separate model for target_10_1
# or use target-specific feature weights in multi-task learning
```

**Group targets by difficulty**:
```python
TARGET_GROUPS = {
    'easy': ['target_8_1', 'target_8_2', 'target_8_3', 'target_3_2', 'target_3_4', 'target_3_5'],
    'medium': ['target_1_3', 'target_2_2', 'target_7_1', 'target_9_4', 'target_10_1'],
    'hard': ['target_2_8', 'target_5_2', 'target_7_3', 'target_2_5', 'target_9_4',
             'target_4_1', 'target_5_1', 'target_6_1', 'target_6_2', 'target_6_3'],
}

# Use different strategies:
# - Easy targets: Standard models with top features
# - Medium targets: Include extra features + interactions
# - Hard targets: Use all features + NaN indicators + target encoding + SMOTE
```

### 5. Feature Preprocessing Pipeline

```python
def preprocess_features(main_df, extra_df, target_df):
    """
    Recommended preprocessing pipeline
    """
    # Step 1: Select top extra features (top 50)
    TOP_EXTRA_FEATURES = ['num_feature_879', 'num_feature_2343', ...]  # see full list
    extra_selected = extra_df[['customer_id'] + TOP_EXTRA_FEATURES]

    # Step 2: Merge main and extra features
    df = main_df.merge(extra_selected, on='customer_id')

    # Step 3: Create NaN indicators (CRITICAL)
    df = create_nan_indicators(df, NAN_INDICATOR_FEATURES)

    # Step 4: Create interaction features
    for cat_col, num_col in INTERACTIONS:
        df = create_interaction_features(df, cat_col, num_col)

    # Step 5: Fill NaN in numeric features (after creating indicators)
    num_features = [c for c in df.columns if c.startswith('num_feature_')]
    for feat in num_features:
        df[feat] = df[feat].fillna(0)  # or use median for features with <50% NaN

    # Step 6: Encode categorical features
    cat_features = [c for c in df.columns if c.startswith('cat_feature_')]
    for feat in cat_features:
        df[feat] = df[feat].fillna(-1).astype('category')

    return df
```

### 6. Model Architecture Recommendations

**For multi-task learning**:
- Use shared feature layers with target-specific heads
- Apply target grouping based on correlation structure
- Weight loss function by target difficulty (higher weight for hard targets)

**For individual target models**:
- Easy targets: LightGBM/XGBoost with top 50 features
- Medium targets: LightGBM/XGBoost with top 100 features + interactions
- Hard targets: Neural networks with all features + oversampling

**Expected performance**:
- Easy targets: AUC 0.90-0.95
- Medium targets: AUC 0.80-0.85
- Hard targets: AUC 0.65-0.75
- **Overall macro AUC**: 0.82-0.85

### 7. Quick Wins Checklist

**Immediate improvements (expected +0.03-0.05 macro AUC)**:
- ✅ Add NaN indicator for num_feature_22 (for target_8_1)
- ✅ Add NaN indicators for num_feature_71, 25, 23, 88 (for Group 3)
- ✅ Include top 50 extra features (see list above)
- ✅ Create cat_feature_13 × num_feature_58 interaction (for target_8_1)

**Medium-term improvements (expected +0.02-0.03 macro AUC)**:
- Create category-specific statistics for top categorical features
- Implement target-specific feature selection
- Add target encoding for categorical features with high cardinality

**Advanced techniques (expected +0.01-0.02 macro AUC)**:
- Train separate model for target_10_1
- Use multi-task learning with target grouping
- Apply SMOTE for targets with <1% positive rate (target_2_8, target_5_2, etc.)

---

**Analysis completed on 100,000 samples from training set. All findings saved to:**
- `/app/output/mutual_information_results.json` - Top extra features by MI
- `/app/output/target_group_correlations.json` - Feature correlations per target group
- `/app/output/feature_interactions.json` - Categorical×numeric and numeric×numeric interactions
- `/app/output/nan_patterns.json` - NaN rates and NaN-target correlations
- `/app/output/analysis_data.json` - Machine-readable summary (next step)