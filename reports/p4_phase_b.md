# Phase B Classical Benchmark and Handoff

**Full-development validation — full-fold refits.** The G0 decision tree and non-PCA logistic control completed refits at both checkpoints. P1 has 48 verified fold score rows with typed model settings. P2 has a calibration manifest for 48 validation Parquets covering 28,293,216 prediction rows across the two models, two horizons, four folds and three seeds. The Parquets live under `data/processed/p4/full_scale/` in the benchmark workspace. [P1 fold scores](p4_phase_b_full/G0/verified_full_fold_runs.csv) · [P1 recipes](p4_phase_b_full/G0/verified_full_fold_recipes.json) · [P2 validation manifest](p4_phase_b_full/G0/p2_handoff.json)

## Full-development validation

The full benchmark uses the `phase-b-data-v1` arrival-audited development Parquet and the six G0–G2 checkpoint files from `phase-b-features-v1`. All six files and the development Parquet match their release checksums. Both G0 checkpoint files have identical predictor values. Comparisons by flight ID confirmed that G1 preserves all G0 values and that G2 preserves all G1 values at both checkpoints. [Development release](https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-/releases/tag/phase-b-data-v1) · [Feature release](https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-/releases/tag/phase-b-features-v1)

The paired validation cohort has 2,357,768 flights across July–October. July has 616,011 validation flights and 28.44% late departures. September has 559,894 flights and 16.56% late departures. Each candidate sees the same validation flights, and P1’s stratified tuning draw has approximately 500,000 training flights per fold and seed. The selected configurations refit on each complete eligible training fold.

P3’s `c9f5560` and `55d01cc` revisions resolve the `weather_missing` name and document the METAR feed and zero-fill assumptions. The published G2 matrices use the agreed column name and match P3’s updated checksums. P1’s sign-off on receipt-time availability remains pending. P2 still needs to review the missing gust and precipitation handling and the unresolved PBI and PPG mappings. The registry at `75a52dd` marks all six weather predictors pending at both checkpoints, and the classical runner enforces that gate before G2 fitting. [P3 feature specification](https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-/blob/55d01cc/FEATURE_SPEC.md#L49-L59) · [P2 weather audit](https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-/blob/75a52ddce266651e7efbd98b4c330951780870ef/reports/weather_audit.md)

Finished group runs land in `data/processed/p4/full_scale/G0`, `G1`, and `G2`. The [full benchmark commands](../README.md#phase-b-classical-models-and-pca) verify and index the per-flight validation predictions, generate the pruning, forest, PCA, and importance figures, and write the results tables included in [the report](report.tex). P1 receives `runs.csv`, `selected_recipes.json`, and `experiment.json` per group. `selected_study.csv` supplies fold scores and typed configurations for both matched PCA families. P2 receives `p2_handoff.json`, `predictions/`, `prediction_index.csv`, `experiment.json`, and `selected_study.csv` per group. The index gives the Parquet row counts and SHA-256 checksums, and the JSON is ready for `src.selected_calibration` on P2’s branch. The November–December evaluation remains with P1.

### Full-development validation — G0 tree tuning at 6h

The completed search ranked 27 configurations across 12 fold–seed fits. The selected tree had unbounded depth, a minimum leaf size of 200, and `ccp_alpha = 6.663750018267775e-06`. On the approximately 500,000-flight training draws, its mean validation PR-AUC was 0.3540 and ROC-AUC was 0.6816. With the same depth and leaf setting, the unpruned tree scored 0.3530 and 0.6761. Pruning reduced the average fitted leaf count from 1,896 to 457. [Depth, pruning and tree size curves](p4_phase_b_full/G0/decision_tree_search_360m.pdf) · [Candidate fold scores](p4_phase_b_full/G0/decision_tree_search_360m.csv) · [Selected tree settings](p4_phase_b_full/G0/decision_tree_selection_360m.json)

**Full-development validation — full-fold refits.** The selected tree completed all 12 refits. The paired export reuses each fit at both checkpoints because the G0 predictor values match exactly. Mean PR-AUC was **0.3583**, mean ROC-AUC was **0.6855**, and mean Brier score was **0.1578** at both checkpoints. The 24 Parquets contain 14,146,608 verified validation predictions across both horizons and all seeds. P1’s tree recipe and fold scores and P2’s manifest and checksum index are under `data/processed/p4/full_scale/G0_tree/`.

| Validation fold | Training flights | Validation flights | PR-AUC | ROC-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| July, three seeds | 3,369,902 | 616,011 | 0.4545 | 0.7012 | 0.1835 |
| August, three seeds | 3,986,996 | 596,812 | 0.3838 | 0.6931 | 0.1637 |
| September, three seeds | 4,586,912 | 559,894 | 0.2713 | 0.6711 | 0.1370 |
| October, three seeds | 5,145,342 | 585,051 | 0.3237 | 0.6767 | 0.1469 |

The full-fold permutation analysis ranked scheduled UTC hour cosine and carrier first. Across folds, seeds, and repeats, their mean PR-AUC drops were 0.1102 and 0.0478. [Full-fold importance](p4_phase_b_full/G0/decision_tree_full_fold_importance.pdf) · [October full-fold rules](p4_phase_b_full/G0/decision_tree_full_fold_october_rules.csv) · [Full-fold scores by month](p4_phase_b_full/G0/decision_tree_full_fold_summary.csv)

**Full-development validation — tuning.** The October tuning refit at seed 42 reproduced the saved PR-AUC of 0.3212 and ROC-AUC of 0.6723. On 3,999 stratified October validation flights, five permutations per predictor put scheduled UTC hour cosine first, with a mean PR-AUC drop of 0.0777. Carrier ranked second at 0.0512. The root split on hour cosine at 0.05016 separated training node late rates of 13.6% and 28.9%. [Permutation importance](p4_phase_b_full/G0/october_tree_tuning_importance.pdf) · [Repeat scores](p4_phase_b_full/G0/october_tree_tuning_importance.csv) · [October tree rules](p4_phase_b_full/G0/october_tree_tuning_rules.csv)

### Full-development validation — G0 forest and matched logistic tuning at 6h

The leading tuning candidate used 256 trees, unbounded depth, a minimum leaf size of 10, and square-root feature sampling. It reached mean PR-AUC 0.3796 and ROC-AUC 0.7013 on the tuning draws. For the same remaining settings, 64 and 128 trees reached mean PR-AUC 0.3779 and 0.3791 and averaged 38.0 and 84.6 seconds per fit and score. The 256-tree fit averaged 173.1 seconds. [Forest tuning curve](p4_phase_b_full/G0/random_forest_search_360m.pdf) · [Fold scores](p4_phase_b_full/G0/random_forest_search_360m.csv) · [Selected forest settings](p4_phase_b_full/G0/random_forest_selection_360m.json)

The matched forest selected 90% variance, with mean PR-AUC 0.3657 compared with 0.3615 at 95% and 0.3795 for its scaled non-PCA control. The 90% paired changes were −0.0138 PR-AUC and −0.0104 ROC-AUC. The logistic pair selected 95% variance. It scored 0.3499 PR-AUC, compared with 0.3458 at 90% and 0.3540 for its non-PCA control. The paired 95% changes were −0.0041 PR-AUC and −0.0029 ROC-AUC. These scores use the tuning draws. [Forest paired fold scores](p4_phase_b_full/G0/pca_forest_search_360m.csv) · [Forest matched settings](p4_phase_b_full/G0/pca_forest_selection_360m.json) · [Logistic paired fold scores](p4_phase_b_full/G0/pca_logistic_search_360m.csv) · [Logistic matched settings](p4_phase_b_full/G0/pca_logistic_selection_360m.json)

**Full-development validation — full-fold refits.** The G0 non-PCA logistic control completed refits, with mean PR-AUC **0.3549**, ROC-AUC **0.6860**, and Brier **0.1620** at each horizon. The 24 per-flight Parquets and checksum index are in `data/processed/p4/full_scale/G0_logistic/`. In concurrent execution, the dense full-fold PCA attempt and a retry using a disk-backed projection exhausted local headroom. The temporary projection was removed and the verified exports were preserved. The paired full-fold PCA score remains pending. [Logistic control scores by month](p4_phase_b_full/G0/logistic_control_full_fold_summary.csv)

### Full-development validation — G1 screening and tuning

At 6h, the G1 tree search selected unbounded depth, a minimum leaf size of 200, and `ccp_alpha = 7.151055288574012e-06`. Pruning raised mean validation PR-AUC from 0.3649 to 0.3667 and ROC-AUC from 0.6807 to 0.6864 on the tuning draws. The average fitted leaf count fell from 1,901 to 470. [G1 pruning curves](p4_phase_b_full/G1/decision_tree_search_360m.pdf) · [Candidate fold scores](p4_phase_b_full/G1/decision_tree_search_360m.csv) · [Selected tree settings](p4_phase_b_full/G1/decision_tree_selection_360m.json)

**Full-development validation — screening only.** At 1h, eight tree structures completed the initial screen across all four validation folds and three seeds, yielding 96 fold scores. Depth 12 with a minimum leaf size of 50 led the completed trials at mean PR-AUC 0.3981. The remaining depth and pruning trials determine the selected 1h recipe. [Verified 1h screening scores](p4_phase_b_full/G1/decision_tree_screen_60m.csv)

The matched G1 logistic search completed at both checkpoints. The 95% policy ranked first among the PCA targets at both horizons. At 6h, it scored 0.3560 PR-AUC and 0.6826 ROC-AUC, with paired changes of −0.0034 and −0.0033 against the control. At 1h, it scored 0.3908 PR-AUC and 0.6990 ROC-AUC, with paired changes of −0.0023 and −0.0020. These comparisons use all four validation folds and all three sampled training seeds. [Fold scores](p4_phase_b_full/G1/pca_logistic_search_scores.csv) · [Paired deltas](p4_phase_b_full/G1/pca_logistic_search_deltas.csv) · [Matched settings](p4_phase_b_full/G1/pca_logistic_search_recipes.json)

### Full-development validation — PCA on tuning draws

The July and October seed-42 draws contained 499,998 and 499,996 training flights. Both G0 draws required 50 components to reach 90% explained variance and 91 to reach 95%. Both G1 draws required 41 and 81 components at each checkpoint. The encoded width was 277 for G0 and 279 for G1, with densities of 3.61% and 4.30%. The faster initial rise of the G1 spectrum describes the scaled training features. Matched classifier validation scores determine the selected variance target. [Training spectra](p4_phase_b_full/pca_tuning_variance.pdf) · [Component counts and retained variance](p4_phase_b_full/pca_tuning_variance.csv)

## Pilot

The P4 pilot uses the paired 20,000-flight development release. G0 uses a projection of the audited schedule/calendar predictors and G1 adds the two origin congestion predictors. Each search uses four chronological validation folds and P1’s common draws at seeds 42, 73, and 101. The selected recipes refit on every eligible training row in the pilot fold and score every eligible pilot validation row. The reserved test period stays with P1. [Implementation plan](../P4_PHASE_B_IMPLEMENTATION_PLAN.md) · [P1 protocol](p1_phase_b.md)

### Pilot model selection and performance reporting

- **Decision Tree** screens depth and minimum leaf size before comparing shared pruning alphas on the leading structures. Supplementary alphas derive from the first fold’s training draw. The published training pruning path and validation curves show where pruning trades score for fewer leaves.
- **Random Forest** screens depth and minimum leaf size at 64 trees, then feature sampling on the two strongest settings, and heavier forests at 128 and 256 trees on the best setting. The pilot used the default unweighted comparison. `--class-weights` adds matched balanced trials. Runtime includes fit and scoring.
- **PCA** fits a centered training-only covariance eigendecomposition on sparse encoded predictors and projects only the selected prefix. Both 90% and 95% variance policies are scored against matched Random Forest and logistic controls. The selected target is recomputed for each refit. The G1 pilot caps both RF study arms at 64 trees because the preliminary 256-tree PCA fit averaged 14.2 seconds on the pilot. The main RF search still compares 64, 128, and 256 trees. Encoded loadings use the feature basis from the representative fit. Fold–seed score spread describes repeated evaluation on the four months. Selection and refit evaluation share the same development folds.

### Pilot evidence

| Group and checkpoint | Tree PR-AUC / ROC-AUC | Forest PR-AUC / ROC-AUC | Matched PCA RF ΔPR-AUC | Matched PCA logistic ΔPR-AUC |
| --- | ---: | ---: | ---: | ---: |
| G0 at 6h and 1h | 0.334 / 0.671 | 0.379 / 0.694 | −0.023 | +0.011 |
| G1 at 6h | 0.352 / 0.671 | 0.394 / 0.699 | −0.024 | +0.016 |
| G1 at 1h | 0.382 / 0.685 | 0.424 / 0.706 | −0.013 | +0.011 |

G0 produced matching predictions at 6h and 1h. The selected tree used depth 12, a minimum leaf size of 200, and zero pruning. The selected forest used 128 trees, depth 24, minimum leaf size 10, and square-root feature sampling. The 90% PCA policy retained 50 components in every selected G0 pilot refit. [G0 summary and spread](p4_phase_b_pilot/G0/summary.csv) · [Paired deltas](p4_phase_b_pilot/G0/pca_selected_deltas.csv)

At 6h in G1, pruning at α ≈ 0.000148 raised sampled-search PR-AUC from 0.339 to 0.342 at the chosen depth and minimum leaf size, and cut the average fitted leaves from 183 to 67. The 1h winner used a minimum leaf size of 200 and mild pruning at α = 0.00001. The G1 benchmark forest used 256 trees at both horizons and averaged 3.39 seconds per fit and score at 6h and 2.73 seconds at 1h. [G1 summary and spread](p4_phase_b_pilot/G1/summary.csv) · [Search scores and timing](p4_phase_b_pilot/G1/search_summary.csv) · [Tree curves](p4_phase_b_pilot/G1/decision_tree.pdf) · [Forest curves](p4_phase_b_pilot/G1/random_forest.pdf)

The selected G1 PCA refits all used a 90% variance target and retained 41 components. The representative encoded input had 279 columns at 4.3% density. The first component loaded most on distance and scheduled duration. The next components included scheduled UTC hour and congestion. Numeric scaling gives those variables greater influence in leading components even though one-hot categories account for most columns. [Full spectrum and paired comparisons](p4_phase_b_pilot/G1/pca.pdf) · [Paired deltas](p4_phase_b_pilot/G1/pca_selected_deltas.csv) · [October loadings](p4_phase_b_pilot/G1/october_pca_loadings.csv)

At 6h, scheduled UTC hour cosine had the largest G1 validation permutation effect. At 1h the recent origin late-departure rate ranked first. The October 1h tree split at a recent late rate of 0.2235, with training late rates of 15.9% and 38.9% in the two child nodes. These are predictive relationships in the pilot. [Permutation importance](p4_phase_b_pilot/G1/permutation_importance.pdf) · [October rules](p4_phase_b_pilot/G1/october_tree_rules.csv)

The G0 matched P4/P5 pilot combined 240 validation rows. A shared October refit at seed 42 reproduced both scores for all 20 model/checkpoint pairs and the ten exported classical prediction files within `1e-12`. The G1 check recomputed ROC-AUC and PR-AUC from all 168 classical prediction exports and reproduced the ten selected October prediction files with the shared scorer. A synthetic G2 fixture exercised the weather registry gate, nested feature check, training and export path. [Matched G0 master](../results/p4_phase_b_pilot/master_results.csv) · [G0 rehearsal configuration](../results/p4_phase_b_pilot/rehearsal_config.json) · [G1 selected score rows](../results/p4_phase_b_pilot/g1_classical_runs.csv)

## Reproduction and team handoff

The [README](../README.md#phase-b-classical-models-and-pca) gives the pilot commands. Use `--forest-jobs 4 --pca-forest-trees 64` for the recorded G1 study and `--forest-jobs 2` for G0. Set numerical library threads to one for the reported CPU timing. Use the same output directory only for an exact matching experiment. `experiment.json` identifies the development and checkpoint file hashes, predictor order, code and dependencies, and full input cohort. The scores include exact training and validation ID digests. The runner rejects paired matrices with incomplete development coverage in full-scale mode. Pass `--parent-manifest` for G1 and G2 to verify that the previous group’s values remain identical by flight ID.

| Recipient | Files and checks |
| --- | --- |
| P1 | Give the selected `runs.csv`, `experiment.json`, and `selected_recipes.json` for the agreed headline group. P1 uses the matched P5 exports for the guarded master table. Freeze the execution config against the integrated revision and rehearse the selected full-fold recipes on October at seed 42. Confirm the full development refit fits the scoring machine. |
| P2 | Read `predictions/<minutes>/<model>/<fold>-<seed>.parquet` for the selected P4 probability exports. The latest runner provides validation `flight_id`, label, probability, UTC day and fit identifiers. Use the fold counts, Brier scores, and prevalences in `runs.csv` for calibration and drift reporting. |
| P3 | Supply frozen G0, G1, and G2 manifests tied to the Phase B arrival-audited development release. Both checkpoint files must cover the same full development IDs. G2 requires `weather_missing` and a P2 registry approval for each added feature. |
| P5 | Run the matched P5 option with P4’s output directory. It uses P4’s selected classical metrics and fits boosters on the same outer fold rows. Plotting writes aggregation-ready CSVs and copies the provenance file. |

For a matched **G1 pilot**, run the following after the P4 runner and with the boosting dependencies installed. Use an equivalent common manifest for the full-scale handoff. G2 also requires `--audit` in the P5 command.

```bash
python -m src.phase_b_boosting --manifest data/sample/feature_manifest.json \
  --classical-run data/processed/p4/pilot_g1 --output data/processed/p4/matched_p5_g1
python -m src.plot_boosting data/processed/p4/matched_p5_g1
python -m src.aggregate_results --classical data/processed/p4/pilot_g1 \
  --boosting data/processed/p4/matched_p5_g1/aggregator_inputs \
  --experiment p4_p5_g1_pilot --output data/processed/p4/matched_g1
```

Merge the top-level keys from `selected_recipes.json` and P5’s `boosting_recipes.json` into the rehearsal config. Copy the `manifest_sha256`, `development_sha256`, and `feature_sha256` keys from `experiment.json` into a `development_inputs` object in that config. Run `src.final_run` with the development manifest and `--rehearsal-output` to validate the combined scorer. The typed recipes preserve the selected PCA family, variance policy and preprocessing, and P5’s native encoding and inner stopping policy. Compare all October prediction IDs and scores with the matching exports. P1 owns final scoring and publication of `results/master_results.csv`.

For the full-scale capacity check, add `--capacity-rehearsal` and use a separate output directory on P1’s scoring machine. That mode requires the complete development matrices, fits on every finally eligible development row, and times scoring a two-month development workload. It writes timings and peak process memory without publishing performance metrics from those overlapping rows. A local G0 sizing probe fit the 95% PCA logistic recipe on 499,996 sampled training flights and scored 585,051 October development flights. It retained 91 components, took 6.0 seconds to fit and 1.1 seconds to score, and reached 1.37 GiB peak process memory. This measures the sampled path on the released development data.

The checked-in Phase A and P5 pilot tables represent their own experiments. P1’s integrated results table requires P5 scores with the matching full-scale manifests and fold digests.
