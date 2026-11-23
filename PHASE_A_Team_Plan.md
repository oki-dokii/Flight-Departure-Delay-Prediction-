# Phase A: Foundations (Days 1-3)

**Project:** Flight Departure Delay Prediction (will this flight leave 15+ min late?)
**Team size:** 5
**Phase A goal:** By the end of Day 3 we have a cleaned, UTC-timestamped dataset, a frozen split design and evaluation harness, a first EDA and column audit, a feature specification, and every model script running on a small sample. No tuning and no test-set use in this phase.

> **Note:** The problem statement says teams of 3 to 4. Confirm with the instructor that 5 is allowed.

---

## Roles at a glance

| # | Role | Phase A focus |
|---|------|---------------|
| P1 | Data and protocol lead | Download, UTC conversion, cleaning, split design, evaluation harness, repo |
| P2 | EDA, audit and report lead | EDA, column audit table, report skeleton |
| P3 | Feature engineering | Feature specification, checkpoint choice, first leak-safe prototypes |
| P4 | Classical models | Baseline, Decision Tree, Random Forest, PCA pipeline scripts on a sample |
| P5 | Boosting models | AdaBoost, GradientBoosting, XGBoost, LightGBM, CatBoost scripts on a sample |

---

## Shared ground rules (everyone reads these first)

1. **Checkpoint rule:** a prediction may only use information available at checkpoint T. Treat any ambiguous column as not available unless you can justify it.
2. **Time-only splits.** Nobody creates their own split. All splits come from P1's `get_folds()`.
3. **The test period is untouched** until the final run. Nobody loads it for EDA, feature selection, encodings, scaling or imputation.
4. **Reproducibility:** fixed seeds, requirements file, no absolute paths.
5. **Git:** one branch per person (`p1-data`, `p2-eda`, ...), pull requests into `main`, at least one reviewer per PR.
6. **Daily 15-minute sync.** Post blockers in the group chat immediately.

---

## P1: Data and Protocol Lead

**Goal:** give the whole team one clean dataset and one split function.

### Tasks
- [ ] Create the GitHub repo and folder structure (`data/`, `src/`, `notebooks/`, `reports/`, `tests/`) plus `README.md`, `requirements.txt`, `.gitignore` (exclude raw data).
- [ ] Download BTS Reporting Carrier On-Time Performance monthly files for the chosen window (suggested: 1 to 2 recent full years; confirm the latest published month on the BTS site). Source: https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD
- [ ] **Record source, licence and download date** in `DATA_SOURCES.md` (BTS data is listed as public domain on data.gov; confirm on the BTS page).
- [ ] Download airport time-zone and location info (e.g. OurAirports) and note its licence.
- [ ] Agree the scope with the team: window, airport subset (e.g. top 20 to 30 origins).
- [ ] Convert local scheduled and actual times to **UTC timestamps**. Handle midnight rollover (a departure after midnight local belongs to the next calendar day) and daylight-saving transitions.
- [ ] Build the label: `LATE = 1 if DepDelay >= 15`, using actual gate departure minus scheduled departure computed from the UTC timestamps. Cross-check against BTS `DepDelay`.
- [ ] Cleaning decisions (document each with a reason): drop cancelled flights, decide diverted flights, handle missing `Tail_Number`, missing times, duplicate rows.
- [ ] Save the cleaned data as parquet in `data/processed/`.
- [ ] **Design the split** and write it in `SPLIT_DESIGN.md`: test period at the end of the timeline (suggest 1 to 2 months, justify), a gap of at least 1 day before it, rolling forward-chaining validation folds on the development data.
- [ ] Implement `get_folds()` returning train/validation indices for each rolling fold. Implement a separate `get_test_split()` and **do not run it** on real test data until the final run.
- [ ] Build the evaluation harness: takes a model with `fit` and `predict_proba`, runs all folds and seeds, returns ROC-AUC and PR-AUC as mean ± std, plus the majority-class baseline.

### Deliverables by end of Day 3
- Cleaned parquet file and `DATA_SOURCES.md`
- `SPLIT_DESIGN.md` with exact dates, fold boundaries and test start date
- `src/splits.py` and `src/evaluate.py`, with a short usage example in the README
- Repo link shared with the team

### Hand-offs
- To **everyone**: parquet file and `get_folds()` (needed by Day 2 end).
- To **P2**: data dictionary notes and the list of cleaning decisions.

### Definition of done
- Any teammate can run one command and get identical folds.
- Row counts before and after each cleaning step are logged.

---

## P2: EDA, Audit and Report Lead

**Goal:** understand the data and lock down which columns are allowed.

### Tasks
- [ ] Read the BTS data dictionary and the reporting rules (14 CFR 234, technical directive) to document how the data was collected and what it covers and misses (US domestic nonstop flights, reporting carriers only, local times, and so on).
- [ ] Wait for P1's parquet (or work on one downloaded month in the meantime), then run EDA on **development data only**:
  - Class balance of `LATE`
  - Delay distribution (histogram, long tail, early departures)
  - Late rate by hour of day, day of week, month, carrier, origin airport, route
  - Missingness map per column
  - Holiday and seasonal effects
  - Sanity checks: duplicates, impossible values, departure before schedule extremes
- [ ] **Build the column audit table** (required in the report). Columns: `column`, `type` (known-in-advance / observed-by-T / static / outcome-like), `allowed?`, `one-line justification`. Cover every BTS column.
  - Mark clearly as outcome-like: `DepTime`, `DepDelay`, `DepDel15`, `TaxiOut`, `WheelsOff`, `ArrTime`, `ArrDelay`, `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`, and similar fields for the target flight.
  - Note that other flights' columns are only usable if their observation time is at or before T.
- [ ] Create the report skeleton (sections per the PS: EDA, cleaning, split design, column audit, feature engineering, model setup, results, extra analysis, limitations) in LaTeX or Word, with placeholders for each owner.
- [ ] Start a running **Limitations** list.

### Deliverables by end of Day 3
- `notebooks/01_eda.ipynb` with 8 to 12 clear plots and written observations
- `COLUMN_AUDIT.md` (or CSV) covering every column
- Report skeleton with section owners assigned
- Written summary of data collection and coverage (half a page)

### Hand-offs
- To **P3**: the column audit, by end of Day 2 (P3 builds features only from allowed columns).
- To **P1**: any data-quality issues found.

### Definition of done
- Every column has a type and a justification.
- No EDA plot uses test-period data.

---

## P3: Feature Engineering

**Goal:** a written feature specification that cannot leak, plus first working prototypes.

### Tasks
- [ ] Decide and justify the **checkpoints** with the team. Suggested: **T = 6 h** and **T = 1 h (or 30 min)** before scheduled departure. Write the justification (what is realistically known at each moment).
- [ ] Write `FEATURE_SPEC.md`. For each feature: definition, source columns, the time it becomes available, and the leak risk.
  - **Schedule and calendar:** hour, day of week, month, holiday proximity, route, carrier, distance, scheduled block time
  - **Inbound aircraft chain** (by `Tail_Number`): the previous leg's scheduled and actual arrival, its arrival delay **only if its actual arrival time <= T**, scheduled turnaround buffer, number of legs already flown that day
  - **Airport congestion by T:** scheduled departures per hour at origin, share of flights at the origin that already departed late before T, average recent departure delay before T
  - **Weather by T:** latest observation at or before T (visibility, wind, precipitation, ceiling), with the data source and licence noted
- [ ] Implement `build_features(df, T)` as a prototype on one month of data. All aggregates must use a strict `timestamp <= T_flight` filter.
- [ ] Write **unit tests** that check no feature uses an event after T (e.g. shift T earlier and confirm features do not change when only later data changes).
- [ ] Download or prototype the weather source (e.g. hourly ASOS/METAR for the chosen airports) and test the "latest observation at or before T" join. Confirm the licence.
- [ ] Decide how missing inbound-aircraft info is handled (first flight of the day, unknown tail, previous leg not yet landed at T). Use explicit indicator features, not silent fills.

### Deliverables by end of Day 3
- `FEATURE_SPEC.md` approved by P2 (audit consistency) and P1 (timestamps)
- Prototype `src/features.py` with passing leak tests on a sample
- Checkpoint justification paragraph for the report

### Hand-offs
- To **P4 and P5**: sample feature matrix (one month, both checkpoints) so they can build model code, by end of Day 2 or early Day 3.
- To **P2**: feature descriptions for the report.

### Definition of done
- Every feature in the spec has an availability time at or before T.
- Leak tests pass.

---

## P4: Classical Models

**Goal:** model code ready to run on any feature matrix and any fold.

### Tasks
- [ ] Set up `src/models_classical.py` with a common interface (`fit`, `predict_proba`) compatible with P1's harness.
- [ ] **Majority-class baseline** wrapper (for reference in the results table).
- [ ] **Decision Tree:**
  - Define the tuning grid: `max_depth`, `min_samples_leaf`, `ccp_alpha`
  - Plan the pruning analysis (cost-complexity path: validation score vs alpha)
  - Plan a depth vs validation AUC curve
- [ ] **Random Forest:** define the grid (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`), consider `class_weight`.
- [ ] **PCA pipeline:**
  - Pipeline: scaler, then PCA, then classifier (choose one, e.g. logistic regression or Random Forest)
  - Compare against the **same classifier without PCA**
  - Plan the explained-variance plot (cumulative variance vs components) and how to pick the number of components
  - Fit scalers and PCA **only on training folds** (inside the pipeline)
- [ ] Decide categorical encoding for these models (one-hot vs target encoding). Target encoding must be fit on training data only, per fold.
- [ ] Smoke-test everything on P3's sample matrix and P1's harness. Confirm it runs end to end and report timing, so we can budget compute.

### Deliverables by end of Day 3
- Working scripts for baseline, Decision Tree, Random Forest and PCA pipeline, all run on the sample
- A short tuning plan document (grids, metrics, how tuning uses validation folds only)
- Runtime estimates for full-data runs

### Hand-offs
- To **P1**: confirm the interface works with the harness.
- To **P5**: share the common encoding and pipeline decisions so boosting uses the same protocol.

### Definition of done
- Each model returns ROC-AUC and PR-AUC through the harness on the sample.
- No code touches the test period.

---

## P5: Boosting Models

**Goal:** all five boosting variants running under the same protocol.

### Tasks
- [ ] Set up `src/models_boosting.py` with the same `fit` / `predict_proba` interface as P4.
- [ ] Install and verify: scikit-learn (AdaBoost, GradientBoostingClassifier), **XGBoost, LightGBM, CatBoost**. Pin versions in `requirements.txt` and tell P1.
- [ ] For each variant, write down the tuning grid and what matters most:
  - **AdaBoost:** `n_estimators`, `learning_rate`, base tree depth
  - **sklearn GradientBoosting:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`
  - **XGBoost:** `max_depth`, `eta`, `subsample`, `colsample_bytree`, `scale_pos_weight`
  - **LightGBM:** `num_leaves`, `learning_rate`, `min_child_samples`, native categorical handling
  - **CatBoost:** `depth`, `learning_rate`, `l2_leaf_reg`, native categorical handling and its ordered target statistics
- [ ] Decide how each model handles categoricals (native vs encoded) and document it. Keep the protocol identical to P4's models.
- [ ] Plan early stopping: use a **validation fold from the development data only**, never the test period.
- [ ] Smoke-test every variant on the sample matrix through P1's harness; record runtime and memory. Flag any that will be too slow at full scale (sklearn GradientBoosting and AdaBoost often are) and propose a fallback such as a subsample for tuning.
- [ ] Prepare the plan for extra analysis: SHAP on the best boosting model, and a novel metric proposal (e.g. precision among the top 10% of flights ranked by predicted risk, or a cost-weighted score).

### Deliverables by end of Day 3
- Working scripts for all five boosting models on the sample
- Tuning plan and runtime table
- One-paragraph proposal for the novel metric

### Hand-offs
- To **P1**: pinned library versions and any harness issues.
- To **P4**: agree on common encoding approach.

### Definition of done
- All five boosters produce ROC-AUC and PR-AUC through the harness on the sample.
- Runtime estimates are recorded.

---

## Dependency timeline for Phase A

| Day | P1 | P2 | P3 | P4 | P5 |
|-----|----|----|----|----|----|
| **1** | Repo, download data, record source and licence | Read data dictionary, start EDA on one month | Draft checkpoints and feature list | Set up model interfaces | Install libraries, set up interfaces |
| **2** | UTC conversion, cleaning, label | EDA on cleaned data, **column audit to P3** | Feature spec, prototype `build_features` | Baseline, DT, RF code | AdaBoost, GB, XGB code |
| **3** | Split design, `get_folds()`, harness | Finish audit, report skeleton | Leak tests, **sample matrix to P4/P5** | PCA pipeline, smoke tests | LightGBM, CatBoost, smoke tests |

## Phase A end-of-phase checklist (team review at end of Day 3)

- [ ] Cleaned parquet exists and row counts are documented
- [ ] `DATA_SOURCES.md` has source, licence and download date
- [ ] `SPLIT_DESIGN.md` has the exact test start date and fold boundaries
- [ ] `get_folds()` and the evaluation harness work for everyone
- [ ] Column audit covers every column
- [ ] Feature spec approved; leak tests pass
- [ ] All 9 required model families run on the sample through the harness
- [ ] Runtime estimates recorded; compute plan agreed
- [ ] Test period untouched
- [ ] Everyone can explain the split design and the checkpoint rule (individual viva)

## What comes next (Phase B preview)
Build full feature matrices for both checkpoints, run rolling-fold tuning for all models, and begin the extra analysis (feature-group ablation, predictability curve, drift, calibration, SHAP, novel metric).
