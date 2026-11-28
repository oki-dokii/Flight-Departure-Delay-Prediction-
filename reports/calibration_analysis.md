# Validation probability calibration

Evidence label: **Pilot (rehearsal defaults)**. **Held-out test** is closed; score cells remain blank.

This run reuses the unchanged classical/boosting factories, manifest readers, shared training_sample and get_folds. Settings are exactly results/rehearsal_config.json (default pilot configurations); no model or feature was tuned by P2. The paired 20,000-flight Phase A pilot supplies four validation months, three seeds and both checkpoints. All 240 model/checkpoint/fold/seed evaluations produced 480,000 validation probability rows, saved under ignored data/processed/p2_calibration/. No test input or final scorer was used.

| Model | Brier 6h | Brier skill 6h | ECE 6h | Brier 1h | Brier skill 1h | ECE 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.2184 | -0.2623 | 0.2184 | 0.2184 | -0.2623 | 0.2184 |
| decision_tree | 0.1609 | 0.0551 | 0.0515 | 0.1571 | 0.0789 | 0.0419 |
| random_forest | 0.1600 | 0.0627 | 0.0601 | 0.1563 | 0.0839 | 0.0572 |
| logistic | 0.1624 | 0.0417 | 0.0676 | 0.1569 | 0.0772 | 0.0505 |
| pca_logistic | 0.1640 | 0.0405 | 0.0652 | 0.1573 | 0.0777 | 0.0492 |
| adaboost | 0.1703 | -0.0072 | 0.1060 | 0.1694 | -0.0021 | 0.1180 |
| gradient_boosting | 0.1577 | 0.0743 | 0.0451 | 0.1537 | 0.0989 | 0.0377 |
| xgboost | 0.1560 | 0.0832 | 0.0400 | 0.1523 | 0.1068 | 0.0360 |
| lightgbm | 0.1556 | 0.0862 | 0.0407 | 0.1520 | 0.1085 | 0.0367 |
| catboost | 0.1580 | 0.0718 | 0.0426 | 0.1534 | 0.0996 | 0.0358 |

Each entry averages the fold/seed metric. Brier skill = 1 − Brier(model)/Brier(training-draw base-rate predictor), computed separately for each validation fold and seed before averaging. The mean reference Brier is 0.1711. Majority uses the harness-compatible most-frequent classifier, giving zero late probability; it is a hard-class baseline, so it is distinct from the probability base-rate reference.

Calibration-in-the-large is recorded as mean prediction, observed late fraction and their signed difference for every fold/seed. ECE is the count-weighted absolute bin gap. Quantile bins are formed on validation probabilities for evaluation only, with tied boundaries collapsed (constant predictions yield one bin). Every bin exports n and its prediction range. Calibration is not fitted using these bins.

Among the default pilot models, LightGBM has the lowest mean Brier at each checkpoint. Its mean prediction minus observed rate is +0.0204 at 6h and +0.0140 at 1h. In September it overpredicts: the mean 6h prediction is 0.2434, versus observed 0.1605 in this pilot. AdaBoost has negative Brier skill despite useful ranking, illustrating that AP does not establish calibrated risk.

Reliability figures calibration_360.png and calibration_60.png show each model and fold for seed 42; metrics/bins cover all seeds. Shading is a 95% percentile interval from 200 resamples of UTC scheduled-departure days with replacement, preserving all flights in a sampled day. Both observed-rate and mean-prediction interval bounds and the number of supported bootstrap replicates are exported; the plot shades observed-rate bounds. This conditions on the fitted model and fixed evaluation bins, and does not include refitting uncertainty or all cross-day dependence. Intervals are not binomial independence intervals.

Class weighting and scale_pos_weight can distort probabilities away from deployment prevalence; a selected weighted configuration must be checked afresh. Logistic regression, coarse tree leaves, averaged forests and boosted margins have different probability behavior; ranking comparisons cannot establish calibration. The mean calibration gap also changes with month drift. No optional Platt/isotonic recalibration was attempted: fitting one would require a trailing inner-training block and a label-availability gap, never validation/test. Full matrices and teammates’ selected configurations remain pending; rerun with --manifest and --config when delivered.

## Full-development validation: selected-model draft

Selected P4/P5 configurations, both checkpoints: Brier, Brier skill, ECE, bias, support and reliability figure slots remain blank pending selected probability exports and frozen provenance. The table in report.tex keeps these numeric cells empty. src.selected_calibration checks exact validation IDs, labels, UTC days, fit identities and agreement with exported scores without fitting. Summarize metrics over the same folds/seeds; provide fold-level reliability and support, with day-block uncertainty conditional on the fixed predictions/bins. Do not copy rehearsal values into selected slots.

**Held-out test:** no support count, score, reliability figure or recalibration is authorized here.
