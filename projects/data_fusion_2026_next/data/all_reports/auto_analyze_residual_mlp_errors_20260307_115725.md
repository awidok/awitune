# Dataset Analysis: Deep Residual MLP Validation Error Patterns

## Key Findings
- **Mean AUC: 0.8004** across 41 targets (range: 0.6351 - 0.9733)
- **3 worst performing targets**: target_3_1 (0.6351), target_9_6 (0.6573), target_9_3 (0.6583)
- **3 best performing targets**: target_2_8 (0.9733), target_3_5 (0.9508), target_8_1 (0.9495)
- **Worst calibrated targets**: target_8_2 (err=0.4359), target_9_3 (err=0.4106), target_3_1 (err=0.3915)
- **750 hard samples** identified (top 1% error across targets)
- **Strongest within-group error correlation**: Group 5 (r=0.6736)

## Detailed Analysis
### 1. Per-Target Performance Metrics
| Target | AUC | Positive Rate | # Positive | Calibration Error | Pred Separation |
|--------|-----|---------------|------------|-------------------|----------------|
| target_10_1 | 0.7219 | 0.3138 | 23538 | 0.2447 | 0.1414 |
| target_1_1 | 0.8886 | 0.0108 | 813 | 0.2663 | 0.4322 |
| target_1_2 | 0.8039 | 0.0035 | 261 | 0.2108 | 0.2509 |
| target_1_3 | 0.8503 | 0.0247 | 1849 | 0.3050 | 0.3665 |
| target_1_4 | 0.8052 | 0.0230 | 1726 | 0.3684 | 0.2784 |
| target_1_5 | 0.8375 | 0.0018 | 135 | 0.0930 | 0.2074 |
| target_2_1 | 0.7952 | 0.0077 | 577 | 0.2941 | 0.2808 |
| target_2_2 | 0.9116 | 0.0252 | 1890 | 0.2100 | 0.5213 |
| target_2_3 | 0.8068 | 0.0013 | 101 | 0.0792 | 0.1279 |
| target_2_4 | 0.7093 | 0.0078 | 582 | 0.2851 | 0.1195 |
| target_2_5 | 0.7584 | 0.0018 | 132 | 0.1126 | 0.1341 |
| target_2_6 | 0.7008 | 0.0045 | 340 | 0.2146 | 0.1203 |
| target_2_7 | 0.8802 | 0.0004 | 31 | 0.0371 | 0.1970 |
| target_2_8 | 0.9733 | 0.0001 | 9 | 0.0089 | 0.1827 |
| target_3_1 | 0.6351 | 0.0974 | 7306 | 0.3915 | 0.0538 |
| target_3_2 | 0.8752 | 0.0981 | 7359 | 0.2665 | 0.4077 |
| target_3_3 | 0.7367 | 0.0012 | 93 | 0.0769 | 0.0864 |
| target_3_4 | 0.9296 | 0.0022 | 164 | 0.0611 | 0.5186 |
| target_3_5 | 0.9508 | 0.0014 | 106 | 0.0481 | 0.4326 |
| target_4_1 | 0.7971 | 0.0086 | 645 | 0.3348 | 0.2495 |
| target_5_1 | 0.7231 | 0.0093 | 694 | 0.3665 | 0.1623 |
| target_5_2 | 0.7258 | 0.0025 | 185 | 0.1553 | 0.1231 |
| target_6_1 | 0.7010 | 0.0094 | 703 | 0.3536 | 0.1139 |
| target_6_2 | 0.7198 | 0.0077 | 574 | 0.2849 | 0.1597 |
| target_6_3 | 0.7380 | 0.0060 | 453 | 0.2406 | 0.1536 |
| target_6_4 | 0.8385 | 0.0081 | 610 | 0.2984 | 0.3144 |
| target_6_5 | 0.9120 | 0.0005 | 39 | 0.0234 | 0.2407 |
| target_7_1 | 0.7746 | 0.0637 | 4777 | 0.3337 | 0.2551 |
| target_7_2 | 0.7735 | 0.0277 | 2077 | 0.3661 | 0.2257 |
| target_7_3 | 0.7531 | 0.0039 | 292 | 0.2656 | 0.1913 |
| target_8_1 | 0.9495 | 0.1025 | 7691 | 0.1907 | 0.6219 |
| target_8_2 | 0.7768 | 0.0321 | 2404 | 0.4359 | 0.2041 |
| target_8_3 | 0.8489 | 0.0193 | 1445 | 0.3372 | 0.3545 |
| target_9_1 | 0.7383 | 0.0040 | 298 | 0.2204 | 0.1769 |
| target_9_2 | 0.8107 | 0.0361 | 2708 | 0.2760 | 0.2688 |
| target_9_3 | 0.6583 | 0.0186 | 1394 | 0.4106 | 0.0772 |
| target_9_4 | 0.8776 | 0.0020 | 148 | 0.1063 | 0.3190 |
| target_9_5 | 0.8254 | 0.0069 | 518 | 0.3147 | 0.3083 |
| target_9_6 | 0.6573 | 0.2218 | 16634 | 0.3275 | 0.0804 |
| target_9_7 | 0.7349 | 0.0785 | 5885 | 0.3219 | 0.1776 |
| target_9_8 | 0.9100 | 0.0103 | 775 | 0.2285 | 0.5521 |

### 2. Hardest Targets Analysis
Top 10 hardest targets (lowest AUC):

| Rank | Target | AUC | Positive Rate | Max Feature Correlation |
|------|--------|-----|---------------|--------------------------|
| 1 | target_3_1 | 0.6351 | 0.0974 | 0.0443 |
| 2 | target_9_6 | 0.6573 | 0.2218 | -0.0795 |
| 3 | target_9_3 | 0.6583 | 0.0186 | -0.0333 |
| 4 | target_2_6 | 0.7008 | 0.0045 | 0.0391 |
| 5 | target_6_1 | 0.7010 | 0.0094 | 0.0391 |
| 6 | target_2_4 | 0.7093 | 0.0078 | 0.0000 |
| 7 | target_6_2 | 0.7198 | 0.0077 | 0.0000 |
| 8 | target_10_1 | 0.7219 | 0.3138 | 0.0000 |
| 9 | target_5_1 | 0.7231 | 0.0093 | 0.0000 |
| 10 | target_5_2 | 0.7258 | 0.0025 | 0.0000 |

### 3. Hard Samples Analysis
Identified **750 hard samples** with consistently high prediction error.

- Mean error (hard samples): 1.0582
- Mean error (normal samples): 0.4239
- Ratio: 2.50x

**Significant feature differences**:

- num_feature_1: NaN rate diff = -0.1938
- num_feature_2: NaN rate diff = -0.3602
- num_feature_3: NaN rate diff = -0.1602
- num_feature_4: NaN rate diff = -0.0882
- num_feature_5: NaN rate diff = -0.1892

### 4. Target Group Analysis
| Group | Size | Mean AUC | Std AUC | Mean Calibration Error |
|-------|------|----------|---------|------------------------|
| 1 | 5 | 0.8371 | 0.0315 | 0.2487 |
| 2 | 8 | 0.8170 | 0.0912 | 0.1552 |
| 3 | 5 | 0.8255 | 0.1209 | 0.1688 |
| 4 | 1 | 0.7971 | 0.0000 | 0.3348 |
| 5 | 2 | 0.7245 | 0.0014 | 0.2609 |
| 6 | 5 | 0.7819 | 0.0806 | 0.2402 |
| 7 | 3 | 0.7671 | 0.0099 | 0.3218 |
| 8 | 3 | 0.8584 | 0.0708 | 0.3213 |
| 9 | 8 | 0.7766 | 0.0887 | 0.2757 |
| 10 | 1 | 0.7219 | 0.0000 | 0.2447 |

**Within-Group Error Correlations**:

- Group 1: Mean correlation = 0.4842 (10 pairs)
- Group 2: Mean correlation = 0.1555 (28 pairs)
- Group 3: Mean correlation = 0.1896 (10 pairs)
- Group 5: Mean correlation = 0.6736 (1 pairs)
- Group 6: Mean correlation = 0.2391 (10 pairs)
- Group 7: Mean correlation = 0.2641 (3 pairs)
- Group 8: Mean correlation = -0.0085 (3 pairs)
- Group 9: Mean correlation = 0.1655 (28 pairs)

### 5. Confusion Patterns
Top 10 most correlated error pairs:

| Target 1 | Target 2 | Error Correlation | Target 1 AUC | Target 2 AUC |
|----------|----------|-------------------|--------------|-------------|
| target_5_1 | target_5_2 | 0.6736 | 0.7231 | 0.7258 |
| target_6_1 | target_6_4 | 0.6687 | 0.7010 | 0.8385 |
| target_1_4 | target_2_2 | 0.6424 | 0.8052 | 0.9116 |
| target_6_2 | target_9_5 | 0.6412 | 0.7198 | 0.8254 |
| target_2_7 | target_6_5 | 0.6129 | 0.8802 | 0.9120 |
| target_1_2 | target_1_5 | 0.6117 | 0.8039 | 0.8375 |
| target_3_5 | target_6_4 | 0.5783 | 0.9508 | 0.8385 |
| target_2_1 | target_9_2 | 0.5739 | 0.7952 | 0.8107 |
| target_3_5 | target_6_5 | 0.5711 | 0.9508 | 0.9120 |
| target_1_3 | target_1_4 | 0.5705 | 0.8503 | 0.8052 |

## Recommendations for Model Builders

1. **Focus on ultra-rare targets**
   - Details: Targets target_3_1, target_9_6, target_9_3 have AUC < 0.75 and need specialized handling (oversampling, focal loss, or transfer learning)
   - Expected Impact: +0.05-0.10 Macro AUC

2. **Engineer NaN pattern features**
   - Details: Hard samples show distinct NaN patterns. Create explicit NaN indicator features for top differentiating features
   - Expected Impact: +0.02-0.03 Macro AUC

3. **Implement multi-task learning for Group 5**
   - Details: Group 5 shows strong within-group error correlation (0.6736). Joint modeling can reduce errors
   - Expected Impact: +0.02-0.04 AUC for targets in this group

4. **Apply calibration**
   - Details: Targets target_8_2, target_9_3, target_3_1 show poor calibration. Apply Platt scaling or isotonic regression
   - Expected Impact: +0.01-0.02 Macro AUC (through better threshold selection)

5. **Address correlated errors**
   - Details: Targets target_5_1 and target_5_2 have highly correlated errors (r=0.6736). Consider joint prediction or feature sharing
   - Expected Impact: +0.01-0.03 AUC for affected targets
