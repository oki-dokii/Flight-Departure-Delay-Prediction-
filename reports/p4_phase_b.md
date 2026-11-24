# Phase B Classical Benchmark and Handoff

The P4 pilot uses the paired 20,000-flight development release. G0 uses a projection of the audited schedule/calendar predictors and G1 adds the two origin congestion predictors. Each search uses four chronological validation folds and P1’s common draws at seeds 42, 73, and 101. The selected recipes refit on every eligible training row in the pilot fold and score every eligible pilot validation row. The reserved test period stays with P1. [Implementation plan](../P4_PHASE_B_IMPLEMENTATION_PLAN.md) · [P1 protocol](p1_phase_b.md)

## Model selection and performance reporting

- **Decision Tree** screens depth and minimum leaf size before comparing shared pruning alphas on the leading structures. Supplementary alphas derive from the first fold’s training draw. The published training pruning path and validation curves show where pruning trades score for fewer leaves.
- **Random Forest** screens depth and minimum leaf size at 64 trees, then feature sampling on the two strongest settings, and heavier forests at 128 and 256 trees on the best setting. The pilot used the default unweighted comparison. `--class-weights` adds matched balanced trials. Runtime includes fit and scoring.
- **PCA** fits a centered training-only covariance eigendecomposition on sparse encoded predictors and projects only the selected prefix. Both 90% and 95% variance policies are scored against matched Random Forest and logistic controls. The selected target is recomputed for each refit. The G1 pilot caps both RF study arms at 64 trees because the preliminary 256-tree PCA fit averaged 14.2 seconds on the pilot. The main RF search still compares 64, 128, and 256 trees. Encoded loadings use the feature basis from the representative fit. Fold–seed score spread describes repeated evaluation on the four months. Selection and refit evaluation share the same development folds.

## Pilot evidence

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

The checked-in Phase A and P5 pilot tables represent their own experiments. Publish the full G0–G2 validation metrics and the one-time test results after P3’s audited matrices and the integrated run are available.
