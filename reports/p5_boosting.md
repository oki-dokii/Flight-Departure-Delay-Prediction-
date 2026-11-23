# Phase A boosting

The schedule smoke run uses P4's deterministic 2,000-flight sample per UTC month from the Phase A development release. Every classifier uses the same fold-local median imputation and capped one-hot encoding as P4. LightGBM and CatBoost also use one-hot features in this comparison. CatBoost therefore does not compute categorical target statistics. Holdout records are excluded. The runner uses the P1 evaluator with four rolling folds and seeds 42, 73 and 101.

## Phase B tuning plan

| Family | Candidate values |
| --- | --- |
| AdaBoost | 50 or 150 estimators, learning rate 0.1 or 1, base tree depth 1 or 2 |
| Gradient Boosting | 50 or 150 estimators, learning rate 0.03 or 0.1, depth 2 or 3, subsample 0.8 or 1 |
| XGBoost | 100 or 300 rounds, depth 3 or 6, eta 0.03 or 0.1, row and column sampling 0.8 or 1, positive weight 1 or the training fold's negative-to-positive ratio |
| LightGBM | 15 or 31 leaves, minimum child samples 20 or 100, learning rate 0.03 or 0.1, 100 or 300 rounds |
| CatBoost | Depth 4 or 6, learning rate 0.03 or 0.1, L2 penalty 3 or 10, 100 or 300 iterations |

Run a bounded search on common time-stratified development IDs at both checkpoints. Rank configurations by mean average precision and report ROC-AUC alongside it. For early stopping, split a trailing time block from each training fold with the same temporal gap and label-availability rules. Choose the iteration count there and refit on the outer training fold before scoring its validation month. The smoke run uses fixed iteration counts throughout.

For extra analysis, compute precision among the highest-risk 10% on each validation fold, with flight ID breaking ties. Compare it with the fold's prevalence to quantify how many late flights an operations team can identify with a fixed alert budget. For the selected booster, summarize SHAP values on a sample of validation flights, using training rows as background and grouping encoded columns by original feature.

## Schedule-only smoke results

The table uses mean ± population standard deviation across 12 fold–seed evaluations per model. Evaluation time includes the baseline. Peak memory covers the full process, including projected input reads and sampling. The approximate full-data evaluation time assumes the same encoded width and linear scaling by P4's 287× row-work ratio. Measure another pilot with P3's features before committing to a full search, especially for sequential AdaBoost and sklearn Gradient Boosting. Use a common smaller training subsample across candidates when the pilot exceeds the compute budget.

| Model | ROC-AUC | PR-AUC | Evaluation (s) | Peak RSS (MiB) | Full-data estimate (min) |
| --- | --- | --- | ---: | ---: | ---: |
| adaboost | 0.670 ± 0.018 | 0.322 ± 0.075 | 2.09 | 1311 | 10.0 |
| catboost | 0.684 ± 0.009 | 0.355 ± 0.066 | 1.81 | 1356 | 8.7 |
| gradient_boosting | 0.688 ± 0.013 | 0.362 ± 0.066 | 2.96 | 1127 | 14.2 |
| lightgbm | 0.692 ± 0.011 | 0.369 ± 0.075 | 0.84 | 1319 | 4.0 |
| xgboost | 0.687 ± 0.013 | 0.356 ± 0.075 | 0.72 | 1389 | 3.5 |

The majority baseline scored ROC-AUC 0.500 ± 0.000 and PR-AUC 0.218 ± 0.051. The [per-run scores](tables/p5_schedule_runs.csv) include every model, seed and fold. Both checkpoint paths were also exercised with a synthetic paired manifest. Real checkpoint comparisons require the approved P3 artifacts.
