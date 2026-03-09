# Dataset Analysis: Meta-Report for Data Fusion 2026 Contest

## Key Findings

- **Целевые группы имеют разную предсказуемость**: Группа 8 (target_8_1 - target_8_3) наиболее предсказуема с корреляциями до 0.31 и 436 информативными extra-фичами для target_8_1, в то время как Группа 3 содержит и самые легкие (target_3_5: AUC=0.95), и самые сложные (target_3_1: AUC=0.64) таргеты

- **NaN-паттерны критически важны**: num_feature_22 NaN-паттерн коррелирует с target_8_1 на уровне -0.67, num_feature_71 NaN-паттерн с target_3_2 на -0.48. Ложные срабатывания для target_3_1 имеют существенно другие NaN-рейты (num_feature_10: 24% vs 56%)

- **532 из 2373 фич можно удалить**: 16 extra-фич с 100% NaN rate (нулевая информация), 287 extra-фич с ≥99% NaN rate, и 245 redundant фич из корреляционных кластеров (включая массивный кластер из 223 highly correlated фич с |r| ≥ 0.95)

- **137 пар категориальных фич дублируют информацию**: cat_feature_2 ↔ cat_feature_25, cat_feature_13 ↔ cat_feature_26, cat_feature_20 ↔ cat_feature_24 показывают Cramer's V = 1.0 (полная ассоциация)

- **3 таргета невозможно предсказать baseline-моделью**: target_2_7, target_2_8, target_6_5 имеют AUC=0.50 из-за экстремальной редкости (9-39 позитивных сэмплов из 75k), требуют специальные подходы

- **Топ-3 extra-фичи универсально важны**: num_feature_477 (top-10 для 10 таргетов, max corr=0.89), num_feature_1504 (8 таргетов, max corr=0.73), num_feature_1176 (7 таргетов, max corr=0.64)

- **Калибровка модели проблемная**: target_8_2 (calibration error=0.44), target_9_3 (0.41), target_3_1 (0.39) требуют рекалибровки или специальных loss-функций

- **Ошибки коррелируют внутри групп**: Группа 5 показывает корреляцию ошибок 0.67, Группа 1 - 0.36, что указывает на потенциал multi-task learning

## Detailed Analysis

### 1. Обзор проведенных исследований

Проведено 5 независимых анализов датасета:

1. **Feature-Target Interactions** (auto_analyze_feature_target_interactions_20260307_155310)
   - Анализ mutual information 2241 extra-фич
   - Выявлены categorical×numeric interactions
   - Исследованы NaN-паттерны как предиктивные сигналы

2. **Residual MLP Error Patterns** (auto_analyze_residual_mlp_errors_20260307_115725)
   - Анализ ошибок baseline модели (mean AUC=0.80)
   - Выявлены проблемы калибровки
   - Определены 750 hard samples

3. **Feature Correlations** (auto_analyze_feature_correlations_20260307_110713)
   - Корреляционная структура 132 main numeric фич
   - Categorical feature associations (Cramer's V)
   - NaN co-occurrence patterns

4. **Extra Features Importance** (auto_analyze_extra_features_importance_20260307_085412)
   - Предсказательная сила 2241 extra-фич
   - Редундантность и NaN-рейты
   - Target-specific benefit scores

5. **Hard Targets Analysis** (auto_error_analysis_hard_targets_20260307_041300)
   - Детальный анализ 38 сложных таргетов (AUC < 0.85)
   - Error correlations и within-group dependencies
   - Feature importance для hardest targets

### 2. Синтез ключевых инсайтов

#### 2.1 Структура таргетов

**Топ-5 легких таргетов** (по AUC baseline модели):
| Target | AUC | Positive Rate | Key Features |
|--------|-----|---------------|--------------|
| target_2_8 | 0.973 | 0.01% | num_feature_10 (r=0.015) |
| target_3_5 | 0.951 | 0.14% | NaN patterns (num_feature_71) |
| target_8_1 | 0.950 | 10.25% | num_feature_27 (r=0.31), num_feature_879 (r=0.30), NaN num_feature_22 (r=-0.67) |
| target_3_4 | 0.930 | 0.22% | num_feature_132, num_feature_2 |
| target_2_2 | 0.912 | 2.52% | num_feature_477 (r=0.89) |

**Топ-5 сложных таргетов**:
| Target | AUC | Positive Rate | Key Issues | Potential Solutions |
|--------|-----|---------------|------------|---------------------|
| target_3_1 | 0.635 | 9.74% | High calibration error (0.39), weak correlations (max r=0.12) | NaN indicators, special loss |
| target_9_6 | 0.657 | 22.18% | Error correlation with target_10_1 (-0.39) | Multi-task learning |
| target_9_3 | 0.658 | 1.86% | Calibration error (0.41), few informative features | Feature engineering |
| target_2_4 | 0.709 | 0.78% | Weak features (max r=0.06) | Oversampling, focal loss |
| target_6_1 | 0.701 | 0.94% | Moderate extra features (21 with |r|>0.1) | Use extra features |

**Группы таргетов**:
- **Группа 8** (target_8_1, target_8_2, target_8_3): Самая предсказуемая, 436 extra-фич с |corr|>0.1 для target_8_1
- **Группа 3** (target_3_1 - target_3_5): Максимальная вариативность (AUC от 0.64 до 0.95), сильная корреляция с NaN-паттернами
- **Группа 10** (target_10_1): Уникальные features (num_feature_27: r=-0.23, num_feature_879: r=-0.23), отличается от других групп

#### 2.2 Feature Engineering Opportunities

**Критически важные фичи по всем исследованиям**:

1. **NaN-индикаторы** (самый сильный сигнал):
   - num_feature_22 NaN → target_8_1 (r=-0.67)
   - num_feature_71 NaN → target_3_2 (r=-0.48)
   - num_feature_25, num_feature_23, num_feature_88 NaN → Group 3 targets (r=-0.26 to -0.33)

2. **Top main features**:
   - num_feature_27: Важен для Group 8 (r=0.31) и Group 10 (r=-0.23)
   - num_feature_62: Group 8 (r=0.28)
   - num_feature_76: Group 10 (r=-0.21)
   - num_feature_132: target_1_3 (r=0.18), target_3_4 (top feature)

3. **Top extra features**:
   - num_feature_879: Наиболее универсальная (avg MI=0.0055, max correlation с target_8_1: r=0.30)
   - num_feature_477: Топ-10 для 10 таргетов, max corr=0.89 с target_2_2
   - num_feature_2343, num_feature_1984, num_feature_1775: Высокий MI с Group 3 targets

**Categorical×Numeric interactions**:
- cat_feature_13 × num_feature_58 для target_8_1: корреляции варьируют от 0.135 до 0.401 по категориям (3x variation)
- Рекомендация: создать interaction features или использовать tree-based models

**Features для удаления**:
- 16 extra features с 100% NaN rate (нулевая информативность)
- 287 extra features с ≥99% NaN rate
- 245 features из correlation clusters (особенно кластер из 223 features с |r| ≥ 0.95)
- Дубликаты categorical features: cat_feature_25 (дублирует cat_feature_2), cat_feature_26 (дублирует cat_feature_13), cat_feature_24 (дублирует cat_feature_20)

#### 2.3 Проблемы и решения

**Проблема 1: Экстремальная редкость 3 таргетов**
- target_2_8 (9 positives), target_2_7 (31 positives), target_6_5 (39 positives)
- Все имеют AUC=0.50 (random chance)
- **Решение**: Oversampling (SMOTE), focal loss, threshold tuning, ensemble с отдельными моделями для редких классов

**Проблема 2: Плохая калибровка для 15 таргетов**
- calibration error > 0.30 для: target_8_2 (0.44), target_9_3 (0.41), target_3_1 (0.39), target_1_4 (0.37), target_7_2 (0.37)
- **Решение**: Platt scaling, isotonic regression, или calibration-aware loss functions

**Проблема 3: Within-group error correlations**
- Группа 5: mean error correlation = 0.67
- Группа 1: mean error correlation = 0.36
- **Решение**: Multi-task learning architectures с shared representations

**Проблема 4: Высокая размерность с редундантностью**
- 2373 features total, из них 532 можно удалить без потери информации
- 58 main features и 1125 extra features имеют >50% NaN rate
- **Решение**: Feature selection + NaN-aware models (XGBoost, LightGBM handle NaN natively)

#### 2.4 Target-Specific Recommendations

| Target Group | Key Strategy | Top Features | Expected Improvement |
|--------------|--------------|--------------|---------------------|
| **Group 8** (target_8_1-8_3) | Использовать все extra features + NaN indicators | num_feature_27, num_feature_879, num_feature_22 NaN | +5-10% AUC |
| **Group 3** (target_3_1-3_5) | NaN patterns + calibration fix | num_feature_71 NaN, num_feature_25 NaN, num_feature_23 NaN | +8-15% AUC для target_3_1 |
| **Group 10** (target_10_1) | Separate model (distinct feature set) | num_feature_27 (neg), num_feature_76 (neg), num_feature_879 (neg) | +3-5% AUC |
| **Rare targets** (target_2_7, target_2_8, target_6_5) | Oversampling + focal loss + threshold tuning | Unique features per target | +10-20% AUC (baseline random) |

### 3. Приоритеты для ML инженеров

#### Приоритет 1: Feature Engineering (High Impact, Low Effort)
1. Создать NaN indicator features для топ-10 фич с высокими NaN rates
2. Добавить interaction features: cat_feature_13 × num_feature_58
3. Удалить 532 redundant/empty features

**Код для NaN indicators**:
```python
# Top NaN features to create indicators
nan_indicator_features = [
    'num_feature_22',  # corr with target_8_1: -0.67
    'num_feature_71',  # corr with target_3_2: -0.48
    'num_feature_25',  # Group 3 targets
    'num_feature_23',  # Group 3 targets
    'num_feature_88',  # Group 3 targets
]

for feat in nan_indicator_features:
    df[f'{feat}_is_nan'] = df[feat].isna().astype(int)
```

**Ожидаемый импакт**: +3-5% macro AUC

#### Приоритет 2: Model Architecture Changes (High Impact, Medium Effort)
1. Multi-task learning для групп с high error correlation (Groups 5, 1)
2. Target-specific heads для групп с разными feature importance (Group 10 vs others)
3. Calibration layers для проблемных таргетов (target_8_2, target_9_3, target_3_1)

**Ожидаемый импакт**: +5-8% macro AUC

#### Приоритет 3: Handling Rare Targets (Critical for 3 targets)
1. Oversampling techniques (SMOTE, ADASYN) для target_2_7, target_2_8, target_6_5
2. Focal loss с alpha/beta tuning
3. Ensemble: отдельные модели для редких классов

**Ожидаемый импакт**: +10-20% AUC для 3 самых редких таргетов (baseline random)

#### Приоритет 4: Extra Features Optimization (Medium Impact, Low Effort)
1. Использовать топ-50 extra features из MI analysis
2. Обратить внимание на num_feature_879, num_feature_477, num_feature_1504
3. Для target_8_1: использовать все 436 informative extra features

**Ожидаемый импакт**: +2-4% macro AUC

## Recommendations for Model Builders

### 1. Feature Selection Strategy

**Обязательно использовать**:
- NaN indicator features для: num_feature_22, num_feature_71, num_feature_25, num_feature_23, num_feature_88
- Top main features: num_feature_27, num_feature_62, num_feature_76, num_feature_132
- Top extra features: num_feature_879, num_feature_477, num_feature_2343, num_feature_1504
- Interaction: cat_feature_13 × num_feature_58

**Удалить**:
- 16 extra features с 100% NaN rate
- 287 extra features с ≥99% NaN rate
- 245 redundant features из correlation clusters
- Дубликаты categorical: cat_feature_25, cat_feature_26, cat_feature_24

### 2. Target-Specific Modeling

```python
# Recommended modeling approach by target groups
target_strategies = {
    'group_8': {
        'targets': ['target_8_1', 'target_8_2', 'target_8_3'],
        'strategy': 'Use all extra features + NaN indicators',
        'features': ['num_feature_27', 'num_feature_879', 'num_feature_22_nan'],
        'loss': 'standard BCE',
    },
    'group_3': {
        'targets': ['target_3_1', 'target_3_2', 'target_3_3', 'target_3_4', 'target_3_5'],
        'strategy': 'NaN patterns focus + calibration fix',
        'features': ['num_feature_71_nan', 'num_feature_25_nan', 'num_feature_23_nan'],
        'loss': 'calibration-aware or focal loss for target_3_1',
    },
    'rare': {
        'targets': ['target_2_7', 'target_2_8', 'target_6_5'],
        'strategy': 'Oversampling + focal loss + threshold tuning',
        'features': 'target-specific (see reports)',
        'loss': 'focal loss with alpha=0.75, gamma=2',
    },
}
```

### 3. Architecture Recommendations

1. **Base model**: Multi-task learning с shared backbone + target-specific heads
2. **Shared layers**: Process main + selected extra features
3. **Target-specific heads**: Отдельные для каждой группы (Group 8, Group 3, Group 10, Rare targets)
4. **Calibration**: Добавить Platt scaling post-processing для таргетов с calibration error > 0.3
5. **Ensembling**: Ensemble моделей для rare targets с threshold tuning

### 4. Training Tips

- **Use tree-based models** (XGBoost/LightGBM/CatBoost) - они natively handle NaN
- **Stratified sampling** для редких классов при создании validation folds
- **Class weights**: Для таргетов с positive rate < 1% использовать вес positive класса = 1/positive_rate
- **Early stopping**: По macro-averaged AUC на validation set
- **Learning rate**: Начать с 0.01 для tree-based, 0.001 для neural networks

### 5. Expected Performance Improvements

| Strategy | Current Baseline AUC | Expected AUC | Improvement |
|----------|---------------------|--------------|-------------|
| NaN indicators | 0.800 | 0.830 | +3.0% |
| Feature selection (remove redundant) | 0.800 | 0.810 | +1.0% |
| Multi-task learning | 0.800 | 0.845 | +4.5% |
| Calibration fix | 0.800 | 0.820 | +2.0% |
| Rare target handling | 0.800 | 0.815 | +1.5% |
| **Combined** | **0.800** | **0.880-0.900** | **+8-10%** |

### 6. Quick Wins (можно сделать за 1 день)

1. Добавить NaN indicators для 5 топ-фич
2. Удалить 532 redundant/empty features
3. Использовать топ-50 extra features из MI analysis
4. Настроить class weights для редких таргетов
5. Добавить Platt scaling для проблемных таргетов

**Ожидаемый результат**: +4-6% macro AUC за минимальное время

## References to Detailed Reports

1. **Feature-Target Interactions** → auto_analyze_feature_target_interactions_20260307_155310
   - Mutual information analysis
   - Categorical×numeric interactions
   - NaN patterns as predictive signals

2. **Residual MLP Error Patterns** → auto_analyze_residual_mlp_errors_20260307_115725
   - Baseline model performance (AUC=0.80)
   - Calibration issues
   - Hard samples identification

3. **Feature Correlations** → auto_analyze_feature_correlations_20260307_110713
   - Correlation structure
   - Categorical associations
   - NaN co-occurrence patterns

4. **Extra Features Importance** → auto_analyze_extra_features_importance_20260307_085412
   - Predictive power of 2241 extra features
   - Redundancy analysis
   - Target-specific benefit scores

5. **Hard Targets Analysis** → auto_error_analysis_hard_targets_20260307_041300
   - Detailed analysis of 38 difficult targets
   - Error correlations
   - Feature importance for hardest targets