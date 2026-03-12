# Per-Target Analytics Report (Base Models Only, No Stacking)

Total base-model experiments with valid test metrics: **154**

## Overall Top 10 by Test Macro ROC-AUC (Base Models)

| # | Macro AUC | Experiment |
|---|-----------|------------|
| 1 | 0.850842 | `auto_neural_feature_interactions_20260311_020246_279282_1` |
| 2 | 0.850641 | `auto_per_target_temperature_calibration_20260311_073410_237682_0` |
| 3 | 0.850464 | `auto_per_target_best_model_14_models_20260312_023713_968132_0` |
| 4 | 0.850413 | `auto_tabnet_attention_feature_selection_20260312_134748_340898_0` |
| 5 | 0.850351 | `auto_target_specific_feature_aug_neural_v2_20260312_054502_913453_0` |
| 6 | 0.850297 | `auto_target_specific_feature_aug_neural_20260312_015122_935036_0` |
| 7 | 0.850280 | `auto_cascade_aware_value_neural_v4_20260312_142410_477248_0` |
| 8 | 0.850232 | `auto_feature_attention_transformer_20260309_172525_907595_1` |
| 9 | 0.850227 | `auto_pseudo_labeling_iterative_20260310_164521_084406_0` |
| 10 | 0.850225 | `auto_per_target_calibration_thresholds_20260309_172715_612459_0` |

## Per-Target Top 5 Base Models

### target_10_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.775289 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 2 | 0.774978 | 0.849378 | `auto_target_specific_feature_selection_20260310_114815_158919_3` |
| 3 | 0.774875 | 0.850464 | `auto_per_target_best_model_14_models_20260312_023713_968132_0` |
| 4 | 0.774875 | 0.850126 | `auto_hyperparameter_optimization_bayesian_20260310_175506_439115_2` |
| 5 | 0.774803 | 0.850122 | `auto_proven_techniques_combined_neural_20260311_081743_479178_0` |

### target_1_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.922010 | 0.848820 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| 2 | 0.921946 | 0.849360 | `auto_mlp_label_smoothing_mixup_20260310_014926_561827_1` |
| 3 | 0.921884 | 0.849631 | `auto_asymmetric_focal_loss_hard_targets_20260310_194446_272637_0` |
| 4 | 0.921822 | 0.846120 | `auto_hard_target_attention_specialist_20260312_112153_479704_0` |
| 5 | 0.921802 | 0.850413 | `auto_tabnet_attention_feature_selection_20260312_134748_340898_0` |

### target_1_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.860886 | 0.850012 | `auto_target_adaptive_focal_loss_20260310_153553_599365_0` |
| 2 | 0.859986 | 0.849631 | `auto_asymmetric_focal_loss_hard_targets_20260310_194446_272637_0` |
| 3 | 0.859349 | 0.841505 | `auto_soft_cascade_multi_task_neural_20260312_114039_442601_0` |
| 4 | 0.859244 | 0.850153 | `auto_focal_loss_hard_targets_20260310_230152_592679_2` |
| 5 | 0.858977 | 0.850028 | `auto_mc_dropout_tta_inference_20260310_235403_438253_0` |

### target_1_3

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.883793 | 0.850028 | `auto_mc_dropout_tta_inference_20260310_235403_438253_0` |
| 2 | 0.883611 | 0.850464 | `auto_per_target_best_model_14_models_20260312_023713_968132_0` |
| 3 | 0.883611 | 0.849631 | `auto_asymmetric_focal_loss_hard_targets_20260310_194446_272637_0` |
| 4 | 0.883606 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 5 | 0.883517 | 0.850220 | `auto_feature_pruning_clean_baseline_20260310_015911_003517_1` |

### target_1_4

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.849605 | 0.849261 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| 2 | 0.849442 | 0.849823 | `auto_ultra_sparse_feature_attention_v3_20260312_054800_928381_0` |
| 3 | 0.849088 | 0.848898 | `auto_hard_target_specialist_neural_v4_20260312_173126_069303_1` |
| 4 | 0.848960 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 5 | 0.848927 | 0.849566 | `auto_cascade_exploit_target_5_2_v2_20260312_111830_187181_0` |

### target_1_5

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.921385 | 0.849261 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| 2 | 0.919665 | 0.850351 | `auto_target_specific_feature_aug_neural_v2_20260312_054502_913453_0` |
| 3 | 0.919605 | 0.849184 | `auto_cascade_rule_target_5_2_specialist_20260312_112703_239409_0` |
| 4 | 0.918743 | 0.849338 | `auto_soft_cascade_multi_task_neural_20260312_171354_006911_1` |
| 5 | 0.918631 | 0.850297 | `auto_target_specific_feature_aug_neural_20260312_015122_935036_0` |

### target_2_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.845187 | 0.848820 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| 2 | 0.844710 | 0.848537 | `auto_ultra_sparse_feature_neural_v4_20260312_112437_925742_2` |
| 3 | 0.844638 | 0.850280 | `auto_cascade_aware_value_neural_v4_20260312_142410_477248_0` |
| 4 | 0.844616 | 0.848722 | `auto_adaptive_per_target_loss_weighting_20260311_050739_042141_0` |
| 5 | 0.844603 | 0.848308 | `auto_hard_target_cascade_model_20260310_221853_495284_1` |

### target_2_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.945373 | 0.849261 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| 2 | 0.945284 | 0.850028 | `auto_mc_dropout_tta_inference_20260310_235403_438253_0` |
| 3 | 0.945208 | 0.850220 | `auto_feature_pruning_clean_baseline_20260310_015911_003517_1` |
| 4 | 0.945201 | 0.842371 | `auto_hard_target_specialist_lightgbm_20260310_180615_889685_1` |
| 5 | 0.945196 | 0.850122 | `auto_proven_techniques_combined_neural_20260311_081743_479178_0` |

### target_2_3

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.812375 | 0.847603 | `auto_neural_label_smoothing_mixup_20260311_044346_471195_1` |
| 2 | 0.812108 | 0.841394 | `auto_mixture_of_experts_target_groups_20260310_202727_558689_0` |
| 3 | 0.811953 | 0.850280 | `auto_cascade_aware_value_neural_v4_20260312_142410_477248_0` |
| 4 | 0.811302 | 0.841887 | `auto_target_specific_feature_selection_nn_20260311_102934_384372_2` |
| 5 | 0.810168 | 0.849303 | `auto_two_stage_cascade_neural_v2_20260312_064028_614508_0` |

### target_2_4

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.797991 | 0.850148 | `auto_swa_generalization_improvement_20260310_114815_152227_1` |
| 2 | 0.797947 | 0.850218 | `auto_temperature_scaling_calibration_20260309_193517_837986_0` |
| 3 | 0.797935 | 0.850225 | `auto_per_target_calibration_thresholds_20260309_172715_612459_0` |
| 4 | 0.797740 | 0.850220 | `auto_feature_pruning_clean_baseline_20260310_015911_003517_1` |
| 5 | 0.797723 | 0.848119 | `auto_hard_target_feature_interactions_20260309_172715_619047_3` |

### target_2_5

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.820000 | 0.847109 | `auto_ultra_sparse_feature_attention_neural_20260312_041212_415921_0` |
| 2 | 0.818531 | 0.842954 | `auto_dcnv2_cross_layer_network_20260310_014926_564214_2` |
| 3 | 0.818409 | 0.846041 | `auto_neural_nan_pattern_cross_attention_20260311_025247_492544_0` |
| 4 | 0.818312 | 0.846416 | `auto_target_specific_focal_loss_tuning_20260310_120527_853217_2` |
| 5 | 0.817780 | 0.847603 | `auto_neural_label_smoothing_mixup_20260311_044346_471195_1` |

### target_2_6

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.779945 | 0.850413 | `auto_tabnet_attention_feature_selection_20260312_134748_340898_0` |
| 2 | 0.779552 | 0.849566 | `auto_cascade_exploit_target_5_2_v2_20260312_111830_187181_0` |
| 3 | 0.778784 | 0.849859 | `auto_multi_task_cascade_target_10_source_20260312_042333_858711_0` |
| 4 | 0.778457 | 0.849378 | `auto_transformer_teacher_distillation_20260310_141617_820348_0` |
| 5 | 0.778363 | 0.849360 | `auto_mlp_label_smoothing_mixup_20260310_014926_561827_1` |

### target_2_7

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.932221 | 0.849989 | `auto_ultra_sparse_feature_specialist_v3_20260312_062104_289883_0` |
| 2 | 0.931436 | 0.849204 | `auto_cascade_aware_neural_target_5_2_20260312_052126_534539_0` |
| 3 | 0.930487 | 0.843505 | `auto_cascade_soft_constraint_loss_neural_20260312_073719_460368_0` |
| 4 | 0.926070 | 0.850842 | `auto_neural_feature_interactions_20260311_020246_279282_1` |
| 5 | 0.926070 | 0.850464 | `auto_per_target_best_model_14_models_20260312_023713_968132_0` |

### target_2_8

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.995357 | 0.817690 | `auto_catboost_optimized_with_target_encoding_20260310_181907_913024_0` |
| 2 | 0.994624 | 0.849631 | `auto_asymmetric_focal_loss_hard_targets_20260310_194446_272637_0` |
| 3 | 0.994347 | 0.849989 | `auto_ultra_sparse_feature_specialist_v3_20260312_062104_289883_0` |
| 4 | 0.994344 | 0.849647 | `auto_cascade_hard_target_improvement_20260311_182754_916306_4` |
| 5 | 0.994257 | 0.848415 | `auto_two_stage_cascade_target_5_1_to_5_2_20260312_072741_107525_1` |

### target_3_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.716465 | 0.849265 | `auto_hard_target_specialist_heads_20260309_172525_914217_4` |
| 2 | 0.715322 | 0.848818 | `auto_neural_hybrid_feature_selection_20260311_183407_858009_2` |
| 3 | 0.714949 | 0.849018 | `auto_ultra_sparse_feature_specialist_neural_20260312_053915_137855_0` |
| 4 | 0.714851 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 5 | 0.714834 | 0.850064 | `auto_nan_pattern_target_encoding_features_20260310_120527_857482_4` |

### target_3_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.916578 | 0.841971 | `smart_lgbm_cross_target_meta_masking_20260311_214200` |
| 2 | 0.915300 | 0.848818 | `auto_neural_hybrid_feature_selection_20260311_183407_858009_2` |
| 3 | 0.915160 | 0.831811 | `auto_catboost_sweet_spot_per_target_weights_20260312_173126_067242_0` |
| 4 | 0.915087 | 0.848993 | `auto_lr_schedule_ablation_study_20260310_015911_005833_3` |
| 5 | 0.915044 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |

### target_3_3

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.806093 | 0.849265 | `auto_hard_target_specialist_heads_20260309_172525_914217_4` |
| 2 | 0.802876 | 0.837591 | `auto_asymmetric_bce_hard_targets_20260310_014926_568763_4` |
| 3 | 0.800177 | 0.847662 | `auto_tabresnet_skip_connections_20260310_221853_491726_0` |
| 4 | 0.798718 | 0.848820 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| 5 | 0.798493 | 0.840993 | `auto_autoint_feature_interactions_20260310_150559_261086_0` |

### target_3_4

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.966495 | 0.849360 | `auto_mlp_label_smoothing_mixup_20260310_014926_561827_1` |
| 2 | 0.965932 | 0.849378 | `auto_transformer_teacher_distillation_20260310_141617_820348_0` |
| 3 | 0.965526 | 0.845915 | `manual_isar_v1` |
| 4 | 0.965291 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 5 | 0.965263 | 0.848941 | `auto_target_3_1_specialist_neural_20260312_005152_879530_0` |

### target_3_5

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.978547 | 0.840993 | `auto_autoint_feature_interactions_20260310_150559_261086_0` |
| 2 | 0.978483 | 0.848308 | `auto_hard_target_cascade_model_20260310_221853_495284_1` |
| 3 | 0.978481 | 0.848722 | `auto_adaptive_per_target_loss_weighting_20260311_050739_042141_0` |
| 4 | 0.978236 | 0.841502 | `auto_swa_onecycle_lr_schedule_20260310_191114_630469_0` |
| 5 | 0.978224 | 0.848801 | `auto_regularized_neural_v2_early_peak_20260312_155514_900471_0` |

### target_4_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.867798 | 0.841971 | `smart_lgbm_cross_target_meta_masking_20260311_214200` |
| 2 | 0.867417 | 0.843793 | `auto_dense_feature_optimized_mlp_20260311_022720_569825_1` |
| 3 | 0.865671 | 0.849647 | `auto_cascade_hard_target_improvement_20260311_182754_916306_4` |
| 4 | 0.865533 | 0.848818 | `auto_neural_hybrid_feature_selection_20260311_183407_858009_2` |
| 5 | 0.865295 | 0.849598 | `auto_two_stage_cascade_source_specialist_20260312_040502_195982_0` |

### target_5_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.777879 | 0.849480 | `auto_distilled_feature_interaction_net_20260311_053308_443204_0` |
| 2 | 0.776590 | 0.848722 | `auto_adaptive_per_target_loss_weighting_20260311_050739_042141_0` |
| 3 | 0.776583 | 0.848820 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| 4 | 0.776552 | 0.849378 | `auto_target_specific_feature_selection_20260310_114815_158919_3` |
| 5 | 0.776549 | 0.848308 | `auto_hard_target_cascade_model_20260310_221853_495284_1` |

### target_5_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.769494 | 0.849042 | `auto_target_3_1_specialist_model_20260311_204206_714941_0` |
| 2 | 0.768315 | 0.850842 | `auto_neural_feature_interactions_20260311_020246_279282_1` |
| 3 | 0.768259 | 0.848722 | `auto_adaptive_per_target_loss_weighting_20260311_050739_042141_0` |
| 4 | 0.767861 | 0.850413 | `auto_tabnet_attention_feature_selection_20260312_134748_340898_0` |
| 5 | 0.766962 | 0.850351 | `auto_target_specific_feature_aug_neural_v2_20260312_054502_913453_0` |

### target_6_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.755821 | 0.849604 | `auto_binary_indicator_ultra_sparse_hard_targets_20260312_072741_104831_0` |
| 2 | 0.754458 | 0.849313 | `auto_value_based_ultra_sparse_neural_20260312_130909_553732_0` |
| 3 | 0.753541 | 0.848898 | `auto_hard_target_specialist_neural_v4_20260312_173126_069303_1` |
| 4 | 0.753425 | 0.845982 | `auto_target_specific_feature_gating_20260310_165648_587946_0` |
| 5 | 0.752718 | 0.849261 | `auto_target_specific_architecture_20260311_233651_494117_2` |

### target_6_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.758262 | 0.847603 | `auto_neural_label_smoothing_mixup_20260311_044346_471195_1` |
| 2 | 0.757782 | 0.850148 | `auto_swa_generalization_improvement_20260310_114815_152227_1` |
| 3 | 0.757713 | 0.850122 | `auto_proven_techniques_combined_neural_20260311_081743_479178_0` |
| 4 | 0.757686 | 0.849294 | `auto_cosine_warmup_restarts_tuning_20260310_130725_412526_2` |
| 5 | 0.757547 | 0.848119 | `auto_hard_target_feature_interactions_20260309_172715_619047_3` |

### target_6_3

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.798492 | 0.850351 | `auto_target_specific_feature_aug_neural_v2_20260312_054502_913453_0` |
| 2 | 0.798288 | 0.848921 | `auto_ultra_sparse_feature_embedding_neural_20260312_072219_250497_0` |
| 3 | 0.798079 | 0.849042 | `auto_target_3_1_specialist_model_20260311_204206_714941_0` |
| 4 | 0.797449 | 0.850842 | `auto_neural_feature_interactions_20260311_020246_279282_1` |
| 5 | 0.797449 | 0.850464 | `auto_per_target_best_model_14_models_20260312_023713_968132_0` |

### target_6_4

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.880903 | 0.849823 | `auto_ultra_sparse_feature_attention_v3_20260312_054800_928381_0` |
| 2 | 0.880131 | 0.850064 | `auto_nan_pattern_target_encoding_features_20260310_120527_857482_4` |
| 3 | 0.880047 | 0.849360 | `auto_mlp_label_smoothing_mixup_20260310_014926_561827_1` |
| 4 | 0.880008 | 0.849378 | `auto_transformer_teacher_distillation_20260310_141617_820348_0` |
| 5 | 0.879648 | 0.850218 | `auto_temperature_scaling_calibration_20260309_193517_837986_0` |

### target_6_5

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.967261 | 0.847746 | `auto_cascade_target_5_2_from_5_1_20260311_204206_717615_1` |
| 2 | 0.966849 | 0.849338 | `auto_soft_cascade_multi_task_neural_20260312_171354_006911_1` |
| 3 | 0.966847 | 0.849377 | `auto_neural_postproc_cascade_hard_targets_20260312_134321_754389_0` |
| 4 | 0.966540 | 0.848960 | `auto_ultra_sparse_feature_embeddings_v2_20260312_011811_672984_0` |
| 5 | 0.966438 | 0.847368 | `auto_optimizer_hyperparams_tuning_20260310_141141_818639_0` |

### target_7_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.827898 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 2 | 0.827742 | 0.849566 | `auto_cascade_exploit_target_5_2_v2_20260312_111830_187181_0` |
| 3 | 0.827690 | 0.849338 | `auto_soft_cascade_multi_task_neural_20260312_171354_006911_1` |
| 4 | 0.827580 | 0.849859 | `auto_multi_task_cascade_target_10_source_20260312_042333_858711_0` |
| 5 | 0.827534 | 0.849480 | `auto_distilled_feature_interaction_net_20260311_053308_443204_0` |

### target_7_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.884589 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 2 | 0.884453 | 0.849631 | `auto_asymmetric_focal_loss_hard_targets_20260310_194446_272637_0` |
| 3 | 0.884320 | 0.848119 | `auto_hard_target_feature_interactions_20260309_172715_619047_3` |
| 4 | 0.884244 | 0.849378 | `auto_target_specific_feature_selection_20260310_114815_158919_3` |
| 5 | 0.884093 | 0.850148 | `auto_swa_generalization_improvement_20260310_114815_152227_1` |

### target_7_3

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.844550 | 0.849647 | `auto_cascade_hard_target_improvement_20260311_182754_916306_4` |
| 2 | 0.844235 | 0.849100 | `auto_self_distill_progressive_temp_20260311_184754_054751_1` |
| 3 | 0.843115 | 0.850012 | `auto_target_adaptive_focal_loss_20260310_153553_599365_0` |
| 4 | 0.843103 | 0.849476 | `auto_cascade_post_processing_target_5_2_20260312_051830_536387_1` |
| 5 | 0.843072 | 0.848522 | `auto_direct_cascade_neural_architecture_20260312_032908_991808_0` |

### target_8_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.983009 | 0.848993 | `auto_lr_schedule_ablation_study_20260310_015911_005833_3` |
| 2 | 0.982906 | 0.850012 | `auto_target_adaptive_focal_loss_20260310_153553_599365_0` |
| 3 | 0.982901 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 4 | 0.982890 | 0.848820 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| 5 | 0.982876 | 0.849823 | `auto_ultra_sparse_feature_attention_v3_20260312_054800_928381_0` |

### target_8_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.872658 | 0.848818 | `auto_neural_hybrid_feature_selection_20260311_183407_858009_2` |
| 2 | 0.872294 | 0.849480 | `auto_distilled_feature_interaction_net_20260311_053308_443204_0` |
| 3 | 0.871985 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 4 | 0.871572 | 0.849647 | `auto_cascade_hard_target_improvement_20260311_182754_916306_4` |
| 5 | 0.871465 | 0.849566 | `auto_cascade_exploit_target_5_2_v2_20260312_111830_187181_0` |

### target_8_3

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.896227 | 0.849303 | `auto_two_stage_cascade_neural_v2_20260312_064028_614508_0` |
| 2 | 0.895965 | 0.849859 | `auto_multi_task_cascade_target_10_source_20260312_042333_858711_0` |
| 3 | 0.895848 | 0.849377 | `auto_neural_postproc_cascade_hard_targets_20260312_134321_754389_0` |
| 4 | 0.895749 | 0.849631 | `auto_asymmetric_focal_loss_hard_targets_20260310_194446_272637_0` |
| 5 | 0.895646 | 0.849042 | `auto_target_3_1_specialist_model_20260311_204206_714941_0` |

### target_9_1

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.824446 | 0.849859 | `auto_multi_task_cascade_target_10_source_20260312_042333_858711_0` |
| 2 | 0.823904 | 0.849046 | `auto_cascade_exploitation_neural_v2_20260312_030730_667645_0` |
| 3 | 0.823638 | 0.850351 | `auto_target_specific_feature_aug_neural_v2_20260312_054502_913453_0` |
| 4 | 0.823420 | 0.850842 | `auto_neural_feature_interactions_20260311_020246_279282_1` |
| 5 | 0.822959 | 0.850413 | `auto_tabnet_attention_feature_selection_20260312_134748_340898_0` |

### target_9_2

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.845811 | 0.849480 | `auto_distilled_feature_interaction_net_20260311_053308_443204_0` |
| 2 | 0.845578 | 0.850064 | `auto_nan_pattern_target_encoding_features_20260310_120527_857482_4` |
| 3 | 0.845541 | 0.850028 | `auto_mc_dropout_tta_inference_20260310_235403_438253_0` |
| 4 | 0.845431 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 5 | 0.845408 | 0.850220 | `auto_feature_pruning_clean_baseline_20260310_015911_003517_1` |

### target_9_3

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.705613 | 0.848898 | `auto_hard_target_specialist_neural_v4_20260312_173126_069303_1` |
| 2 | 0.704926 | 0.850148 | `auto_swa_generalization_improvement_20260310_114815_152227_1` |
| 3 | 0.704738 | 0.850227 | `auto_pseudo_labeling_iterative_20260310_164521_084406_0` |
| 4 | 0.704729 | 0.850464 | `auto_per_target_best_model_14_models_20260312_023713_968132_0` |
| 5 | 0.704729 | 0.850232 | `auto_feature_attention_transformer_20260309_172525_907595_1` |

### target_9_4

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.934175 | 0.850280 | `auto_cascade_aware_value_neural_v4_20260312_142410_477248_0` |
| 2 | 0.934074 | 0.849989 | `auto_ultra_sparse_feature_specialist_v3_20260312_062104_289883_0` |
| 3 | 0.933628 | 0.848820 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| 4 | 0.933094 | 0.849261 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| 5 | 0.932957 | 0.849492 | `auto_value_based_ultra_sparse_neural_v2_20260312_171354_004606_0` |

### target_9_5

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.864469 | 0.848902 | `auto_neural_target_specific_gating_20260311_182754_911265_2` |
| 2 | 0.864182 | 0.850280 | `auto_cascade_aware_value_neural_v4_20260312_142410_477248_0` |
| 3 | 0.863913 | 0.850064 | `auto_nan_pattern_target_encoding_features_20260310_120527_857482_4` |
| 4 | 0.863565 | 0.848921 | `auto_ultra_sparse_feature_embedding_neural_20260312_072219_250497_0` |
| 5 | 0.863475 | 0.848941 | `auto_target_3_1_specialist_neural_20260312_005152_879530_0` |

### target_9_6

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.702065 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 2 | 0.701896 | 0.849989 | `auto_ultra_sparse_feature_specialist_v3_20260312_062104_289883_0` |
| 3 | 0.701833 | 0.849604 | `auto_binary_indicator_ultra_sparse_hard_targets_20260312_072741_104831_0` |
| 4 | 0.701806 | 0.849480 | `auto_distilled_feature_interaction_net_20260311_053308_443204_0` |
| 5 | 0.701800 | 0.849265 | `auto_hard_target_specialist_heads_20260309_172525_914217_4` |

### target_9_7

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.782954 | 0.849598 | `auto_two_stage_cascade_source_specialist_20260312_040502_195982_0` |
| 2 | 0.782781 | 0.849360 | `auto_mlp_label_smoothing_mixup_20260310_014926_561827_1` |
| 3 | 0.782638 | 0.849378 | `auto_transformer_teacher_distillation_20260310_141617_820348_0` |
| 4 | 0.782529 | 0.850196 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| 5 | 0.782392 | 0.850126 | `auto_hyperparameter_optimization_bayesian_20260310_175506_439115_2` |

### target_9_8

| # | Target AUC | Macro AUC | Experiment |
|---|-----------|-----------|------------|
| 1 | 0.937802 | 0.848820 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| 2 | 0.937556 | 0.849046 | `auto_cascade_exploitation_neural_v2_20260312_030730_667645_0` |
| 3 | 0.937378 | 0.849303 | `auto_two_stage_cascade_neural_v2_20260312_064028_614508_0` |
| 4 | 0.937299 | 0.849261 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| 5 | 0.937229 | 0.849467 | `auto_target_specific_ultra_sparse_binary_v2_20260312_072741_108649_2` |

## Hardest Targets — Base Models (sorted by best AUC)

| Target | Best AUC | Best Experiment |
|--------|----------|-----------------|
| target_9_6 | 0.702065 ⚠️ | `smart_lightgbm_meta_top6_models_20260311_222252` |
| target_9_3 | 0.705613 ⚠️ | `auto_hard_target_specialist_neural_v4_20260312_173126_069303_1` |
| target_3_1 | 0.716465 ⚠️ | `auto_hard_target_specialist_heads_20260309_172525_914217_4` |
| target_6_1 | 0.755821 ⚠️ | `auto_binary_indicator_ultra_sparse_hard_targets_20260312_072741_104831_0` |
| target_6_2 | 0.758262 ⚠️ | `auto_neural_label_smoothing_mixup_20260311_044346_471195_1` |
| target_5_2 | 0.769494 ⚠️ | `auto_target_3_1_specialist_model_20260311_204206_714941_0` |
| target_10_1 | 0.775289 ⚠️ | `smart_lightgbm_meta_top6_models_20260311_222252` |
| target_5_1 | 0.777879 ⚠️ | `auto_distilled_feature_interaction_net_20260311_053308_443204_0` |
| target_2_6 | 0.779945 ⚠️ | `auto_tabnet_attention_feature_selection_20260312_134748_340898_0` |
| target_9_7 | 0.782954 ⚠️ | `auto_two_stage_cascade_source_specialist_20260312_040502_195982_0` |
| target_2_4 | 0.797991 ⚠️ | `auto_swa_generalization_improvement_20260310_114815_152227_1` |
| target_6_3 | 0.798492 ⚠️ | `auto_target_specific_feature_aug_neural_v2_20260312_054502_913453_0` |
| target_3_3 | 0.806093 | `auto_hard_target_specialist_heads_20260309_172525_914217_4` |
| target_2_3 | 0.812375 | `auto_neural_label_smoothing_mixup_20260311_044346_471195_1` |
| target_2_5 | 0.820000 | `auto_ultra_sparse_feature_attention_neural_20260312_041212_415921_0` |
| target_9_1 | 0.824446 | `auto_multi_task_cascade_target_10_source_20260312_042333_858711_0` |
| target_7_1 | 0.827898 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| target_7_3 | 0.844550 | `auto_cascade_hard_target_improvement_20260311_182754_916306_4` |
| target_2_1 | 0.845187 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| target_9_2 | 0.845811 | `auto_distilled_feature_interaction_net_20260311_053308_443204_0` |
| target_1_4 | 0.849605 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| target_1_2 | 0.860886 | `auto_target_adaptive_focal_loss_20260310_153553_599365_0` |
| target_9_5 | 0.864469 | `auto_neural_target_specific_gating_20260311_182754_911265_2` |
| target_4_1 | 0.867798 | `smart_lgbm_cross_target_meta_masking_20260311_214200` |
| target_8_2 | 0.872658 | `auto_neural_hybrid_feature_selection_20260311_183407_858009_2` |
| target_6_4 | 0.880903 | `auto_ultra_sparse_feature_attention_v3_20260312_054800_928381_0` |
| target_1_3 | 0.883793 | `auto_mc_dropout_tta_inference_20260310_235403_438253_0` |
| target_7_2 | 0.884589 | `smart_lightgbm_meta_top6_models_20260311_222252` |
| target_8_3 | 0.896227 | `auto_two_stage_cascade_neural_v2_20260312_064028_614508_0` |
| target_3_2 | 0.916578 | `smart_lgbm_cross_target_meta_masking_20260311_214200` |
| target_1_5 | 0.921385 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| target_1_1 | 0.922010 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| target_2_7 | 0.932221 | `auto_ultra_sparse_feature_specialist_v3_20260312_062104_289883_0` |
| target_9_4 | 0.934175 | `auto_cascade_aware_value_neural_v4_20260312_142410_477248_0` |
| target_9_8 | 0.937802 | `auto_cascade_aware_neural_v5_20260312_170708_976808_0` |
| target_2_2 | 0.945373 | `auto_target_specific_architecture_20260311_233651_494117_2` |
| target_3_4 | 0.966495 | `auto_mlp_label_smoothing_mixup_20260310_014926_561827_1` |
| target_6_5 | 0.967261 | `auto_cascade_target_5_2_from_5_1_20260311_204206_717615_1` |
| target_3_5 | 0.978547 | `auto_autoint_feature_interactions_20260310_150559_261086_0` |
| target_8_1 | 0.983009 | `auto_lr_schedule_ablation_study_20260310_015911_005833_3` |
| target_2_8 | 0.995357 | `auto_catboost_optimized_with_target_encoding_20260310_181907_913024_0` |
