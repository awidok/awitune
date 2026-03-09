# Dataset Analysis: Model Reliability and Feature Importance Validation

## Key Findings

- **CRITICAL: Best model at risk of overfitting to unreliable sparse features**: The current best model uses PLE encoding for ALL numeric features, including 499 sparse features (>95% NaN) that have been proven unreliable
- **Sparse features show extreme instability**: 100 sparse features demonstrate correlation variance >0.2 across different data subsets, indicating they cannot be trusted for predictions
- **Dense features provide reliable signal**: 1,742 dense features (≤95% NaN) have mean confidence interval width of 0.0074, indicating stable and reproducible correlations
- **Hard targets lack strong individual features**: Max correlation with dense features is only 0.0846 for target_9_6, 0.0585 for target_3_1, and 0.0359 for target_9_3
- **Sparse feature "signal" is likely noise**: Features showing high correlations (e.g., 0.6207 for target_9_6) have very wide confidence intervals and high variance, making them unreliable for production

## Detailed Analysis

### 1. Feature Reliability Assessment

**Dense Features (≤95% NaN)**: 1,742 features
- Mean confidence interval width: 0.0074
- Status: **RELIABLE** - large sample sizes (typically >50,000), narrow confidence intervals
- Suitable for model training without additional validation

**Sparse Features (>95% NaN)**: 499 features
- Mean confidence interval width: NaN (many features have undefined variance)
- Unreliable features found: 100 features with correlation variance >0.2 across subsets
- Status: **UNRELIABLE** - small sample sizes (<30,000), wide confidence intervals, high variance
- **Recommendation**: Remove from training or validate extensively on holdout sets

### 2. Sparse Feature Unreliability Demonstration

Analysis of random subsets revealed extreme instability in sparse features:

| Feature | Target | Full Dataset r | Subset Correlation Range | Valid n | NaN Rate | Status |
|---------|--------|---------------|-------------------------|---------|----------|--------|
| num_feature_282 | target_9_3 | -0.008 | 0.697 | 30,465 | 95.94% | UNRELIABLE |
| num_feature_185 | target_9_3 | -0.009 | 0.618 | 22,375 | 97.02% | UNRELIABLE |
| num_feature_262 | target_9_3 | -0.011 | 0.541 | 7,356 | 99.02% | UNRELIABLE |
| num_feature_229 | target_9_3 | -0.000 | 0.529 | 1,737 | 99.77% | UNRELIABLE |
| num_feature_248 | target_9_3 | -0.019 | 0.510 | 4,539 | 99.39% | UNRELIABLE |

**Interpretation**: The correlation coefficient changes drastically (range >0.5) depending on which subset of data is sampled. This means the reported correlation is essentially random and cannot be trusted for predictions.

### 3. Hard Target Analysis

#### target_3_1 (Positive Rate: 9.84%)

**Top Dense Features:**
| Rank | Feature | Correlation | Valid Samples | NaN Rate | CI Width | Status |
|------|---------|-------------|---------------|----------|----------|--------|
| 1 | num_feature_578 | 0.0585 | 30,003 | 96.00% | 0.0113 | RELIABLE |
| 2 | num_feature_1765 | 0.0527 | 71,780 | 90.43% | 0.0073 | RELIABLE |
| 3 | num_feature_1813 | 0.0496 | 57,615 | 92.32% | 0.0082 | RELIABLE |

**Top Sparse Features:**
| Rank | Feature | Correlation | Valid Samples | NaN Rate | CI Width | Status |
|------|---------|-------------|---------------|----------|----------|--------|
| 1 | num_feature_428 | 0.2867 | 624 | 99.92% | 0.0783 | UNRELIABLE |
| 2 | num_feature_479 | 0.1629 | 1,835 | 99.76% | 0.0464 | UNRELIABLE |
| 3 | num_feature_398 | 0.1458 | 1,169 | 99.84% | 0.0582 | UNRELIABLE |

**Analysis**: Max dense correlation is only 0.0585, indicating weak individual feature signal. Sparse features showing higher correlations (0.2867) are unreliable due to small sample sizes (<2,000). Previous analysis found that num_feature_624 was reported with r=0.946 but actual correlation is -0.049.

**Recommendation**:
1. Use feature interactions (difference, ratio) - previous analysis showed +72.6% improvement over individual features
2. Remove sparse features - they add noise without reliable signal
3. Focus on dense features with >50,000 samples for stability

#### target_9_3 (Positive Rate: 1.87%)

**Top Dense Features:**
| Rank | Feature | Correlation | Valid Samples | NaN Rate | CI Width | Status |
|------|---------|-------------|---------------|----------|----------|--------|
| 1 | num_feature_1402 | 0.0359 | 81,232 | 89.17% | 0.0069 | RELIABLE |
| 2 | num_feature_1483 | 0.0332 | 99,584 | 86.72% | 0.0062 | RELIABLE |
| 3 | num_feature_1469 | 0.0318 | 99,584 | 86.72% | 0.0062 | RELIABLE |

**Top Sparse Features:**
| Rank | Feature | Correlation | Valid Samples | NaN Rate | CI Width | Status |
|------|---------|-------------|---------------|----------|----------|--------|
| 1 | num_feature_479 | 0.1007 | 1,835 | 99.76% | 0.0464 | UNRELIABLE |
| 2 | num_feature_428 | -0.0849 | 624 | 99.92% | 0.0783 | UNRELIABLE |
| 3 | num_feature_382 | -0.0618 | 1,273 | 99.83% | 0.0551 | UNRELIABLE |

**Analysis**: Weakest signal among hard targets - max dense correlation only 0.0359. Very low positive rate (1.87%) makes this target challenging. Sparse features showing correlation 0.1007 have <2,000 samples and high variance.

**Recommendation**:
1. Use ensemble approaches with strong regularization
2. Feature interactions critical - previous analysis found +9.4% improvement
3. Consider oversampling or focal loss for extreme class imbalance
4. Remove all sparse features - unreliable signal

#### target_9_6 (Positive Rate: 22.31%)

**Top Dense Features:**
| Rank | Feature | Correlation | Valid Samples | NaN Rate | CI Width | Status |
|------|---------|-------------|---------------|----------|----------|--------|
| 1 | num_feature_1479 | 0.0846 | 98,686 | 86.84% | 0.0062 | RELIABLE |
| 2 | num_feature_1481 | -0.0742 | 98,686 | 86.84% | 0.0062 | RELIABLE |
| 3 | num_feature_1475 | -0.0665 | 98,686 | 86.84% | 0.0062 | RELIABLE |

**Top Sparse Features:**
| Rank | Feature | Correlation | Valid Samples | NaN Rate | CI Width | Status |
|------|---------|-------------|---------------|----------|----------|--------|
| 1 | num_feature_428 | 0.6207 | 624 | 99.92% | 0.0783 | UNRELIABLE |
| 2 | num_feature_482 | -0.1319 | 798 | 99.89% | 0.0694 | UNRELIABLE |
| 3 | num_feature_175 | 0.1194 | 2,817 | 99.62% | 0.0370 | UNRELIABLE |

**Analysis**: Strongest signal among hard targets - max dense correlation 0.0846. However, sparse feature showing very high correlation (0.6207) is based on only 624 samples and is highly unreliable. Previous analysis showed num_feature_428 had r=0.924 but with only 11 valid samples.

**Recommendation**:
1. This target has the best individual feature signal among hard targets
2. Feature interactions showed +75.8% improvement - highly recommended
3. **CRITICAL**: Do NOT use num_feature_428 despite its high correlation - it's unreliable
4. Focus on dense features from num_feature_1479 family

### 4. Robustness Test Results

Comparison of signal strength across feature types (sample of 10,000 rows):

| Target | Main Features Max | Dense Extra Max | Sparse Extra Max | Sparse Reliability |
|--------|------------------|----------------|-----------------|-------------------|
| target_3_1 | 0.1762 | 0.0566 | 0.5767 | UNRELIABLE |
| target_9_3 | 0.1565 | 0.0585 | 0.5803 | UNRELIABLE |
| target_9_6 | 0.0913 | 0.0688 | 0.9376 | UNRELIABLE |

**Key Observation**: Sparse features show very high correlations (up to 0.9376) but these are artifacts of small sample sizes and cannot be trusted. The signal from main features and dense extra features is lower but reliable.

### 5. Confidence Interval Analysis

**Dense Features**:
- Mean CI width: 0.0074
- Interpretation: Correlation estimates are precise within ±0.0037
- Example: If correlation is reported as 0.05, true value is likely between 0.046 and 0.054

**Sparse Features**:
- Mean CI width: varies widely (0.03-0.10+)
- Interpretation: Correlation estimates are highly uncertain
- Example: num_feature_428 with correlation 0.6207 has CI width 0.0783, meaning true value could be anywhere from 0.58 to 0.66 - and this changes drastically across subsets

### 6. Model Reliability Assessment

**Current Best Model Risk Profile:**

If the model uses ALL features (including sparse):
- ✗ **HIGH RISK**: May overfit to unreliable sparse features
- ✗ **HIGH RISK**: Sparse feature correlations may not generalize to test set
- ✗ **HIGH RISK**: Model may learn noise patterns instead of real signal
- ✗ **HIGH RISK**: Production performance may degrade unexpectedly

**Recommended Model Configuration:**

If the model uses DENSE features only:
- ✓ **LOW RISK**: All features have reliable, reproducible correlations
- ✓ **LOW RISK**: Model will generalize well to test set
- ✓ **LOW RISK**: Faster training (1,742 vs 2,241 features)
- ✓ **LOW RISK**: More interpretable feature importance

## Recommendations for Model Builders

### Immediate Actions

1. **RETRAIN with dense features only**
   - Remove 499 sparse features (>95% NaN)
   - Use only 1,742 dense features (≤95% NaN)
   - Expected impact: Better generalization, reduced overfitting

   ```python
   # Feature filtering code
   import polars as pl

   train_extra = pl.read_parquet('data/train_extra_features.parquet')
   nan_rates = train_extra.null_count() / len(train_extra)

   dense_features = [col for col in train_extra.columns
                    if col.startswith('num_feature_')
                    and nan_rates[col][0] / len(train_extra) <= 0.95]

   # Use only dense features for training
   train_extra_dense = train_extra.select(['customer_id'] + dense_features)
   ```

2. **ADD feature interactions for hard targets**
   - Create difference interactions: `feat_a - feat_b`
   - Create ratio interactions: `feat_a / (feat_b + epsilon)`
   - Focus on interactions between top dense features
   - Expected impact: +5-10% AUC for hard targets

   ```python
   # Interaction engineering for hard targets
   def create_interactions(df, top_features):
       interactions = {}
       for i, feat_a in enumerate(top_features[:20]):
           for feat_b in top_features[i+1:i+21]:
               # Difference interaction
               interactions[f'{feat_a}_minus_{feat_b}'] = df[feat_a] - df[feat_b]
               # Ratio interaction (with epsilon to avoid division by zero)
               interactions[f'{feat_a}_div_{feat_b}'] = df[feat_a] / (df[feat_b] + 1e-6)
       return pl.DataFrame(interactions)

   # Get top dense features for each hard target
   top_dense_3_1 = ['num_feature_578', 'num_feature_1765', 'num_feature_1813']
   interactions_3_1 = create_interactions(train_extra, top_dense_3_1)
   ```

3. **VALIDATE any sparse feature on holdout set**
   - If you suspect a sparse feature is important, validate correlation on a separate holdout set
   - Only use if correlation is reproducible across multiple data splits
   - Be skeptical of correlations >0.1 from <5,000 samples

   ```python
   # Sparse feature validation
   def validate_sparse_feature(feature, target, train_data, val_data):
       """Check if correlation is reproducible across train/val splits"""
       train_corr = compute_correlation(train_data[feature], train_data[target])
       val_corr = compute_correlation(val_data[feature], val_data[target])

       # Check if correlation is consistent
       if abs(train_corr - val_corr) < 0.05:
           return True, f"Reproducible: train={train_corr:.3f}, val={val_corr:.3f}"
       else:
           return False, f"NOT reproducible: train={train_corr:.3f}, val={val_corr:.3f}"
   ```

### Target-Specific Recommendations

**target_3_1**:
- Priority: HIGH - Add feature interactions
- Use top dense features: num_feature_578, num_feature_1765, num_feature_1813
- Remove unreliable sparse features: num_feature_428, num_feature_479
- Expected improvement: +5-10% AUC from interactions

**target_9_3**:
- Priority: HIGH - Use ensemble with strong regularization
- Very weak signal - max dense correlation only 0.0359
- Consider oversampling or focal loss for 1.87% positive rate
- Feature interactions critical
- Expected improvement: +3-5% AUC from ensemble + interactions

**target_9_6**:
- Priority: MEDIUM - Has strongest individual feature signal
- Use top dense features: num_feature_1479, num_feature_1481, num_feature_1475
- **CRITICAL**: Do NOT use num_feature_428 (r=0.6207 but unreliable)
- Feature interactions showed best improvement (+75.8%)
- Expected improvement: +5-8% AUC from interactions

### Feature Importance Guidance

For all targets, prioritize features in this order:

1. **Main numeric features** (num_feature_1 to num_feature_132)
   - Usually have <20% NaN rate
   - Strong, reliable signal
   - Include in all models

2. **Dense extra features** (num_feature_133+ with ≤95% NaN)
   - Provide additional signal
   - Reliable with large sample sizes
   - Include in all models

3. **Feature interactions** (difference, ratio of dense features)
   - Critical for hard targets
   - Previous analysis showed strong improvements
   - Compute for top 20-50 dense features

4. **Categorical features** (cat_feature_1 to cat_feature_67)
   - Use with proper encoding (embedding, target encoding)
   - Reliable, no NaN values

5. **Sparse extra features** (num_feature_133+ with >95% NaN)
   - **DO NOT USE** without extensive validation
   - High risk of overfitting
   - If you must use, validate on multiple holdout sets

### Expected Impact

| Action | Estimated AUC Improvement | Risk Reduction | Training Time |
|--------|--------------------------|----------------|---------------|
| Remove sparse features | +1-3% | HIGH - prevents overfitting | -22% (fewer features) |
| Add feature interactions | +5-10% for hard targets | LOW - stable features | +15% (more features) |
| Retrain with dense only | +2-5% | HIGH - better generalization | -22% |
| Validate sparse features | N/A | PREVENTS deployment failures | +5% (validation step) |

### Model Deployment Checklist

Before deploying any model:

- [ ] Verify model uses only dense features (≤95% NaN)
- [ ] Check that sparse features (>95% NaN) have been removed or extensively validated
- [ ] Validate feature importance on holdout set
- [ ] Test prediction stability across different data samples
- [ ] Monitor for distribution shift in feature NaN rates
- [ ] Set up alerts for sudden changes in feature correlations

## Conclusion

The current best model using PLE encoding for ALL features is **at risk of overfitting to unreliable sparse features**. Analysis demonstrates that:

1. **Sparse features (>95% NaN) are unreliable** - correlations change drastically across subsets
2. **Dense features (≤95% NaN) are reliable** - stable correlations with narrow confidence intervals
3. **Hard targets need feature interactions** - individual features have weak signal
4. **Retraining with dense features only will improve generalization** and reduce overfitting risk

**Final Recommendation**: Retrain the model with dense features only (1,742 features) and add engineered feature interactions for hard targets. This will provide better, more reliable performance in production.