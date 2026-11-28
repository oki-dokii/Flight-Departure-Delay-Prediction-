# Phase B: Full-Scale Runs, Tuning and Extra Analysis

**Project:** Flight Departure Delay Prediction (LATE = departs 15+ min after schedule)
**Team:** 5 people (P1 to P5)
**Based on:** the repo `Flight-Departure-Delay-Prediction--main` as uploaded after Phase A

---

## 1. Where the repo stands after Phase A

### Done

| Area | Status | Evidence in repo |
|---|---|---|
| Data | BTS Reporting Carrier On-Time data, Jan to Oct 2025 development set: **5,749,041 flights**, 21.38% late | `reports/tables/quality.json`, `data/download_manifest.json`, release `phase-a-data-v1` (SHA-256 recorded) |
| Cleaning | UTC conversion, label check (0 disagreements with `DepDelay`), cleaning counts reconciled | `src/prepare.py`, `DATA_SOURCES.md` |
| Split | 4 rolling folds (validation Jul, Aug, Sep, Oct 2025), 24h boundary buffer, 48h gap, **test starts 2025-11-02** | `SPLIT_DESIGN.md`, `src/splits.py`, `reports/tables/folds.csv` |
| Harness | ROC-AUC, PR-AUC, accuracy, mean and std over folds and seeds, with majority baseline | `src/evaluate.py` |
| EDA | 12 figures, tables, observations, notebook | `reports/figures/`, `notebooks/01_eda.ipynb` |
| Column audit | Full audit with conditional entries blocked | `COLUMN_AUDIT.csv`, `COLUMN_AUDIT.md` |
| Features | Checkpoints 6h and 1h; schedule/calendar features; 3-hour origin congestion; tail-based inbound features quarantined; leak tests | `FEATURE_SPEC.md`, `src/features.py`, `tests/test_features.py` |
| Classical models | Baseline, Decision Tree, Random Forest, logistic, PCA + logistic. **Smoke run on 20,000 flights** with fixed settings | `reports/p4_classical_tuning.md` |
| Boosting | All five boosters. **Smoke run on 2,000 flights per month** with fixed settings | `reports/p5_boosting.md` |

### Smoke-run results so far (schedule-only features; not final numbers)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Majority | 0.500 | 0.218 |
| Decision Tree | 0.669 | 0.331 |
| Random Forest | 0.685 | 0.358 |
| LightGBM | 0.692 | 0.369 |
| Gradient Boosting | 0.688 | 0.362 |
| XGBoost | 0.687 | 0.356 |
| CatBoost | 0.684 | 0.355 |
| AdaBoost | 0.670 | 0.322 |

These are tiny samples, so use them only to confirm the pipeline runs. The signal is weak and the models are close, which is what the problem statement predicted.

### Gaps that Phase B must close

1. **Features exist only for a small sample.** `build_sample_features.py` covers the sample; no full 5.7M-row feature matrix exists for either checkpoint.
2. **No weather yet.** The spec lists it as a stub, and `limitations.md` says station mapping and licences are pending.
3. **Tail-based inbound features are blocked.** `Tail_Number` is conditional in the audit because BTS is retrospective. This is potentially the strongest signal, so a decision is needed (see Decision D1).
4. **No tuning has been done.** All settings are fixed. The Decision Tree needs a justified depth and cost-complexity pruning, and the Random Forest needs tuning.
5. **PCA is only explored with 2 components** (51.5% variance). The required explained-variance analysis needs a full variance curve and a component count chosen by variance, and the comparison should use the best classifier setup.
6. **LightGBM and CatBoost use one-hot input**, so CatBoost's native categorical handling is not exercised.
7. **Real checkpoint comparison (6h vs 1h) has not happened**, only a synthetic test.
8. **Extra analysis (6 marks) has not started**, apart from planned precision-at-top-10%.
9. **Report sections are placeholders** in `reports/report.tex`; the repo link, final results table and test run are pending.
10. **Small open data issues** in `data_quality_issues.csv`: arrival clock validation, one flight already departed at the 1h checkpoint, tail availability.

### Things to watch (found while reading)

- **Strong drift between folds.** Validation late rate falls from 28.4% (July) to 16.6% (September). Expect large PR-AUC spread across folds, and report it honestly. It is also a good extra-analysis topic.
- **Deterministic models show zero seed spread.** In the smoke table, AdaBoost scores are identical across seeds 42, 73 and 101. Reported "spread across seeds" is then only fold spread. State this in the report, or give seeds a real effect (e.g. different training subsamples or random-initialised models).
- **No winter in training.** Development is Jan to Oct, test is Nov to Dec (Thanksgiving and December holidays). Expect a regime shift and write it up as a limitation and a discussion point.
- **Test row count is unknown until the final run** (by design), but the report needs the exact count and start date.

---

## 2. Decisions the team must make on Day 1 of Phase B

| ID | Decision | Recommended option | Owner |
|---|---|---|---|
| D1 | Use tail-based inbound features? | Keep them out of the **primary** audited feature set, and run them as a clearly labelled **ablation** ("what if the aircraft rotation is known at T"). State the assumption: schedules and prior-leg arrivals are available from an operational feed. This gives both a defensible headline result and a rich extra analysis. | P2 + P3 |
| D2 | Weather source and station mapping | Hourly airport observations (e.g. from the Iowa Environmental Mesonet ASOS archive) for the 20 origin airports, using only the latest observation at or before T. Record the licence and download date. | P3 |
| D3 | Tuning data size | Tune on a time-stratified subsample (e.g. 500k to 1M training rows per fold), then refit the final config on full training folds. Document the subsample. | P4 + P5 |
| D4 | Primary metric for model selection | PR-AUC (late class is the minority), report ROC-AUC alongside | All |
| D5 | Source of seed variation | Pick one (e.g. random subsample seeds or model seeds with subsampling) so seed spread is meaningful | P1 |

---

## 3. Shared rules for Phase B

1. **The test period stays closed** until the single final run (P1 runs it, once).
2. All models consume the **same frozen feature matrices** and the same `get_folds()`.
3. Every tuned configuration is chosen using validation folds only, ranked by mean PR-AUC.
4. Every run logs: code revision, parameters, fold, seed, runtime, row counts.
5. Features and models are never changed after the test run.

---

## 4. Roles in Phase B

### P1: Data, protocol and final-run owner

**Goal:** make full-scale runs possible and protect the protocol.

- [ ] Download or build the **November and December 2025** archives with the same cleaning rules, kept separate from development. Verify they are never loaded in any notebook until the final run.
- [ ] Add a **protocol guard** to the harness: raise an error if any row with timestamp at or after the test start appears in development inputs.
- [ ] Resolve open issues from `data_quality_issues.csv`: arrival clock validation, the one already-departed flight at the 1h checkpoint, repeated schedule keys.
- [ ] Implement Decision D5 (meaningful seed variation) in `evaluate.py` and re-verify tests.
- [ ] Add a **results aggregator** that collects every model's per-fold, per-seed scores into one master table (`results/master_results.csv`) and produces the final table with mean ± std for both checkpoints.
- [ ] Prepare the **final-run script**: refit each model on all development data with the chosen config, score the test period once, record the exact test start date and test flight count, save predictions.
- [ ] Publish updated dataset assets (full feature matrices, if sizes allow) as a release, with SHA-256 digests.

**Deliverables:** protocol guard, test-period files held back, `master_results.csv` aggregator, final-run script (dry-run tested on a validation fold).

### P2: EDA, audit, report and drift analysis lead

**Goal:** keep the narrative and the audit consistent with the new features, and own the drift analysis.

- [ ] **Update the column audit** for every new input: weather columns, tail-derived columns, airport static data. Each gets a type and a one-line justification.
- [ ] Record the Decision D1 assumption in the audit and in `limitations.md`.
- [ ] Extend EDA to new features (still development data only): late rate vs weather conditions, vs inbound delay, vs origin congestion.
- [ ] **Drift analysis (extra analysis):** quantify the shift in late rate and feature distributions across folds Jul, Aug, Sep, Oct (e.g. population stability index per feature, performance per month). Explain the 28% to 17% fall in validation late rate.
- [ ] **Calibration analysis (extra analysis):** reliability diagrams and Brier score per model and fold.
- [ ] Fill in the report: Sections 1 to 5 (problem, dataset, cleaning, split, audit), Limitations.
- [ ] Put the GitHub repo link in the report.

**Deliverables:** updated audit, drift and calibration figures with written insight, report sections 1 to 5 drafted.

### P3: Features at full scale, weather and ablations

**Goal:** produce the final frozen feature matrices and the feature-group story.

- [ ] Run `build_features` on the **full development set** for both checkpoints (T = 6h and T = 1h), in chunks if memory requires it. Verify row alignment with the development parquet and run the leak tests on the full output.
- [ ] **Integrate weather (Decision D2):** download observations, map each origin airport to a station, join the latest observation at or before T, add `weather_missing` flags. Add leak tests for the weather join (shifting T must not use later observations).
- [ ] Extend the congestion features if time allows (e.g. 1h window, share of cancellations seen so far) while keeping the strict at-or-before-T rule.
- [ ] Build **feature-group matrices** so ablations are one flag:
  - G0: schedule and calendar only
  - G1: + origin congestion
  - G2: + weather
  - G3: + tail-based inbound chain (clearly labelled as the Decision D1 ablation)
- [ ] Update `feature_manifest.json` for all matrices, with feature roles and revision tags.
- [ ] **Feature-group ablation (extra analysis):** with a fixed fast model (e.g. LightGBM), measure how much each group adds at each checkpoint. This answers the project's core question ("how much can be predicted from which kind of information").
- [ ] Write the Feature Engineering report section.

**Deliverables:** frozen matrices at both checkpoints, updated manifest, leak tests passing on full data, ablation results table and figure.

### P4: Classical models (Decision Tree, Random Forest, PCA)

**Goal:** tuned, justified classical benchmark and a proper PCA study.

- [ ] **Decision Tree:** tune `max_depth`, `min_samples_leaf` and `ccp_alpha` on the rolling folds. Produce the **cost-complexity pruning path** (validation PR-AUC vs alpha) and a depth vs score curve. Write the justification for the final choice.
- [ ] **Random Forest:** tune `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` (and `class_weight` as an option) with a bounded search using Decision D3. Report runtime.
- [ ] **PCA study:**
  - Fit PCA on training folds only (inside the pipeline) on the preprocessed matrix
  - Plot the **full explained-variance curve** and choose component counts (e.g. 90% and 95% variance), not only 2
  - Compare **PCA + classifier vs the same classifier without PCA**, using a classifier that is not trivially linear (e.g. PCA + Random Forest or PCA + logistic regression, both reported)
  - Report the paired differences by fold and seed, as in the Phase A smoke study
  - Note that the PCA input is dominated by one-hot columns (273 encoded columns at 2.2% density in Phase A), and discuss what the components mean
- [ ] Run all three on **both checkpoints** and on the G0 to G2 feature sets as agreed with P3.
- [ ] **Extra analysis:** permutation importance and tree-structure inspection for the final Decision Tree (top splits as readable rules).
- [ ] Write the classical models report section.

**Deliverables:** tuned DT and RF with pruning and tuning figures, PCA study with variance plots and paired comparison, results rows in `master_results.csv`.

### P5: Boosting models and operational metrics

**Goal:** tuned boosting benchmark, native categorical handling, and the novel metric.

- [ ] Run the bounded tuning grids from `reports/p5_boosting.md` for **AdaBoost, Gradient Boosting, XGBoost, LightGBM and CatBoost** on both checkpoints. Rank by mean PR-AUC, report ROC-AUC alongside.
- [ ] Use **early stopping on a trailing time block** inside each training fold (with the same gap and label-availability rules), then refit on the full outer training fold.
- [ ] **Native categoricals:** run LightGBM and CatBoost with native categorical handling as well as one-hot, and report the difference. This is the first real use of CatBoost's ordered target statistics.
- [ ] Handle runtime: AdaBoost and sklearn Gradient Boosting scale poorly. Use a common smaller training subsample for all candidates if needed (Decision D3), and state it.
- [ ] **Novel metric (extra analysis):** precision at the top 10% (and 5% and 20%) of predicted risk per fold, versus the fold's prevalence ("lift"). Add a **cost-based curve**: expected cost when ops staff can act on a fixed number of alerts. Compare all models, not only boosters, by plugging into the harness.
- [ ] **SHAP analysis (extra analysis):** for the selected booster, on a sample of validation flights, grouped by original feature, per checkpoint. Compare what drives risk at 6h vs 1h.
- [ ] Write the boosting report section.

**Deliverables:** tuned boosters at both checkpoints, native-categorical comparison, lift and cost curves, SHAP figures, rows in `master_results.csv`.

---

## 5. Timeline

| Days | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| **B1** | Team decisions D1 to D5, protocol guard | Update audit for D1, start report | Full-data features, weather download | Plan grids, start DT pruning | Plan grids, set up early stopping |
| **B2 to B3** | Seed variation, issue fixes | Drift analysis | Weather join and leak tests, **G0 and G1 matrices out** | DT pruning, RF tuning on G0 | Boosting tuning on G0 |
| **B4 to B5** | Aggregator, release assets | Calibration, extend EDA | **G2 and G3 matrices out**, ablation runs | PCA study, RF on G1 and G2 | Boosting on G1 and G2, native categoricals |
| **B6 to B7** | Final-run dry run | Report sections 1 to 5 | Ablation figure, report section | PCA and permutation importance write-up | Lift, cost curve, SHAP |
| **Freeze** | **Everyone freezes features, configs and code** | | | | |
| **Final run** | **P1 runs the test period once** | | | | |
| **After** | Final results table | Report assembly | Review | Review | Review |

## 6. Freeze checklist (before the single test run)

- [ ] Feature matrices frozen for both checkpoints, manifest and revision tags committed
- [ ] Final hyperparameters for all 9 model families chosen from validation results and committed
- [ ] All leak tests pass on full data
- [ ] Protocol guard passes: no test-period rows in any development input
- [ ] Final-run script dry-run on the last validation fold gives the same results as the tuning run
- [ ] `master_results.csv` complete for validation folds
- [ ] Column audit updated for every input column
- [ ] Everyone has agreed that nothing changes after the test run

## 7. Final results table (target format)

One row per model and checkpoint, with ROC-AUC and PR-AUC as mean ± std across folds and seeds on the validation folds, plus the single test-period score and the test flight count.

| Model | Checkpoint | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|
| Majority | 6h / 1h | | | | |
| Decision Tree | 6h / 1h | | | | |
| Random Forest | 6h / 1h | | | | |
| Best classifier without PCA | 6h / 1h | | | | |
| Same classifier with PCA | 6h / 1h | | | | |
| AdaBoost | 6h / 1h | | | | |
| Gradient Boosting | 6h / 1h | | | | |
| XGBoost | 6h / 1h | | | | |
| LightGBM | 6h / 1h | | | | |
| CatBoost | 6h / 1h | | | | |

Also report: exact test start date (2025-11-02), the number of test flights, and any additional metrics (lift at top 10%, Brier score).

## 8. Extra-analysis map (6 marks)

| Analysis | Owner | Answers |
|---|---|---|
| Feature-group ablation and checkpoint comparison | P3 | How much can be predicted from which kind of information? |
| Drift across months | P2 | Why do scores and late rates change over time, and what does that imply for the winter test? |
| Calibration (reliability, Brier) | P2 | Can the probabilities be trusted? |
| Lift at top 5/10/20% and cost curve | P5 | What can an operations team actually gain from a fixed alert budget? |
| SHAP, 6h vs 1h | P5 | What drives risk, and how does it change as T gets closer? |
| Decision Tree rules, permutation importance | P4 | Which simple rules explain delays? |
| Tail-chain ablation (Decision D1) | P3 | How much of delay propagation does aircraft rotation explain? |
| Leakage demonstration (optional) | P1 | How inflated would scores be if forbidden columns were used? |

## 9. Viva preparation

Five of the 20 marks are individual. Before the viva, every member should be able to explain, in their own words:

- Why the split is time-based, the buffer and gap, and why test starts on 2025-11-02
- The checkpoint rule, and why `Tail_Number` was treated as conditional
- Why PR-AUC is the main selection metric, and what the spread across folds means
- One result from each other role (e.g. what PCA did to PR-AUC, which feature group added most)
- The limitations: Jan to Oct training only, retrospective BTS data, one year of data, reporting carriers only

Run a cross-questioning round where each person is asked about another person's section.
