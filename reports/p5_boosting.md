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

## Phase B paired development pilot

The bounded search uses the 20,000 flights and twelve audited predictors from the paired Phase A manifest, including the congestion measures available at each checkpoint. At each horizon we compare two predefined parameter settings per family. LightGBM and CatBoost run both settings with one-hot and native categorical input. The native LightGBM path learns category levels on training rows and treats unknown levels as missing. Native CatBoost receives category strings and uses ordered boosting. This compares the native CatBoost setup as a whole, including its boosting mode.

For each outer fold and seed, the same P1 training draw is used across candidates. The last fourteen calendar dates in that draw supply inner validation. The inner fit excludes the preceding day and requires every training label to be observable before the gap. Refitting at 10, 30 and 60 rounds allows each library to use the same stopping rule: stop when inner average precision first fails to improve, then refit the best round count on the whole outer training draw. Use the outer validation month only for scoring. Rank candidates by mean outer average precision across four folds and three seeds. Selecting and reporting on those same folds can favor the best candidate. The final test will provide an independent estimate.

At each alert budget we rank flights by predicted probability, breaking ties by flight ID, and take the ceiling of the budget times the fold size. Precision measures the share of alerted flights that actually leave late. Lift divides that precision by the fold's observed late rate. The illustrative cost is one unit per alert plus five units per missed late flight, divided by the number of flights. This measures prioritization under stated costs and makes no claim that an alert prevents a delay. Classical comparators use P4's fixed settings. The plotted points are averages across folds and seeds.

Permutation SHAP is calculated on up to 128 October validation flights for the winning model at each checkpoint, using the first seed and 32 random feature orderings. Each ordering starts from actual outer-training rows. Permutations replace original columns before encoding, so contributions are grouped by input feature and expressed in probability units. Marginal replacements can break correlations among route and schedule variables. Interpret the resulting rankings as model explanations.

The [candidate scores](p5_phase_b_pilot/candidates.csv), [native encoding comparison](p5_phase_b_pilot/encoding.csv), [operations summary](p5_phase_b_pilot/summary.csv), and [selected boosting rows](p5_phase_b_pilot/selected_boosting_runs.csv) are reproducible from the README commands. Repeat the search on P3's frozen full development matrices for the final model selection.

One-hot LightGBM ranked first at both horizons. Its mean average precision increased from 0.387 at six hours to 0.420 at one hour. At one hour, 50.3% of flights in its highest-risk decile departed late, for a mean fold-level lift of 2.34. The [pilot master results](../results/p5_phase_b_pilot/master_results.csv) combine the selected boosters with the existing matched P4 pilot. The [table by checkpoint](../results/p5_phase_b_pilot/results_summary.md) includes fold–seed spread.

### Month ablation and calibration handoff

On the 20,000-flight G1 pilot, four of five boosting families had lower mean average precision without calendar month at both checkpoints. LightGBM lost 0.0071 at six hours and 0.0054 at one hour. CatBoost gained 0.0028 and 0.0009 respectively. Each difference averages 12 paired runs using the same training draw, validation flights and selected hyperparameters. Each variant repeated chronological round selection. These differences describe the pilot and do not establish a full-development feature choice. See the [model averages](p5_month_pilot/mean_deltas.csv) and [individual differences](p5_month_pilot/paired_deltas.csv).

Person 2 can read the 120 with-month validation probability files using the following command from a checkout containing the P2 calibration reader, this handoff and the Phase A development and paired pilot files.

```sh
python -m src.selected_calibration --handoff reports/p5_month_pilot/handoff.json
```

To reproduce the fits and package the prediction files from the repository root:

```sh
python -m src.p5_month_ablation --manifest data/sample/feature_manifest.json --selected reports/p5_phase_b_pilot/selected_boosting_runs.csv --output data/processed/p5_month_pilot
python -m src.p5_calibration_handoff --manifest data/sample/feature_manifest.json --source data/processed/p5_month_pilot --output reports/p5_month_pilot
```

The full G0 and G1 checkpoint matrices are available in the Phase B feature release. The joint full-scale comparison still needs the corresponding P4 experiment identity and selected runs on the same revision and cohort. G2 additionally needs clearance of the weather availability audit.
