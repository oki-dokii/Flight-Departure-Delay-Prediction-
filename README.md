# Flight Departure Delay Prediction

Predict whether a scheduled flight will leave its gate at least 15 minutes late using information available at a pre-departure checkpoint.

## Phase A data and evaluation

Use Python 3.12 from the repository root. After installing dependencies in a virtual environment, this command downloads the public source files and builds the development parquet and cleaning counts.

```bash
python -m pip install -r requirements.txt
python -m src.download && python -m src.prepare
```

The parquet is written to `data/processed/flights_development.parquet`. The downloader preserves the November and December source archives for final evaluation. For the verified team handoff, download the development parquet and per-month cleaning counts from the [Phase A dataset release](https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-/releases/tag/phase-a-data-v1). Place them under `data/processed/`. The branch tracks code and source receipts. See [data sources and cleaning](DATA_SOURCES.md) and the [frozen split](SPLIT_DESIGN.md).

Run this example to reproduce the fold sizes and exercise the evaluation API using a schedule-only feature. Supply a model factory that returns a fresh classifier or sklearn pipeline for each seed. Keep outcome columns out of the feature matrix.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.evaluate import evaluate
from src.splits import OBSERVED, TIME, get_folds

development = pd.read_parquet(
    "data/processed/flights_development.parquet",
    columns=[TIME, OBSERVED, "LATE", "CRSDepTime"],
)
print([(len(train), len(valid)) for train, valid in get_folds(development)])

X = development[["CRSDepTime"]]
runs, summary = evaluate(
    lambda seed: DecisionTreeClassifier(max_depth=3, random_state=seed),
    X,
    development,
)
print(summary)
```

The harness reports ROC-AUC, PR-AUC (average precision) and accuracy for the model and majority-class baseline, with mean and population standard deviation across folds and seeds. The example is a smoke test. Use the same folds and seeds for both prediction checkpoints.

Run the protocol tests with `python -m unittest discover -s tests -v`.

## Person 2: development analysis

Install `requirements-eda.txt` in a Python 3.12 virtual environment. Place the release's development parquet and cleaning counts in `data/processed/`, then run:

```bash
python -m src.eda
python -m unittest discover -s tests -v
```

Open `notebooks/01_eda.ipynb` with that environment to regenerate and view the analysis. Outputs are written to `reports/figures/`, `reports/tables/`, and `COLUMN_AUDIT.csv`. No test dataset is read. Conditional audit entries remain blocked pending P1/P3 review; derived features need separate availability approval. The report skeleton is `reports/report.tex`.

## P5 boosting smoke runs

Install `requirements-boosting.txt` in the project environment. The five model names are `adaboost`, `gradient_boosting`, `xgboost`, `lightgbm`, and `catboost`. On macOS, XGBoost and LightGBM require the OpenMP runtime (`libomp`).

```bash
python -m src.run_boosting --schedule-only --model xgboost
```

To evaluate both P3 checkpoints, replace `--schedule-only` with `--input-manifest PATH` using the same manifest as P4. See [boosting results and tuning plan](reports/p5_boosting.md).

## Phase B data and protocol

Use the [Phase B dataset release](https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-/releases/tag/phase-b-data-v1) and [P1 protocol handoff](reports/p1_phase_b.md). The [paired pilot results](results/results_summary.md) exercise the revised training protocol at both checkpoints.

## Phase B classical models and PCA

Use the paired development manifest to tune Decision Tree and Random Forest and compare both classifiers with their PCA controls. The runner selects settings on common sampled training draws, then refits the selected settings on the full training portion of each validation fold.

```bash
python -m src.phase_b_classical --manifest data/sample/feature_manifest.json \
  --group G1 --pilot --forest-jobs 4 --pca-forest-trees 64 --output data/processed/p4/pilot_g1
python -m src.plot_classical data/processed/p4/pilot_g1 --output reports/p4_phase_b_pilot/G1
```

Use `--group G0 --pilot` with the same released manifest for a schedule-only projection. For full-scale work, download the [full feature release](https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-/releases/tag/phase-b-features-v1) and put the G0–G2 manifests and Parquets in `data/processed/phase_b/`. Use the audited development Parquet from `phase-b-data-v1`. Verify both release checksum lists before fitting, and pass the preceding group’s manifest with `--parent-manifest` for G1 and G2. G2 requires P2’s approved feature registry via `--audit`.

```bash
python -m src.phase_b_classical --manifest data/processed/phase_b/manifest_G0.json \
  --group G0 --forest-jobs 6 --output data/processed/p4/full_scale/G0
python -m src.p4_handoff --manifest data/processed/phase_b/manifest_G0.json \
  --run data/processed/p4/full_scale/G0
python -m src.plot_classical data/processed/p4/full_scale/G0 --output reports/p4_phase_b_full/G0
```

Use the corresponding group in each command for G1 and G2. The handoff checker verifies the scored flight IDs, labels, UTC days, probabilities, and metrics, then writes `prediction_index.csv` with one checksum per Parquet and `p2_handoff.json` for P2’s calibration reader. Keep the run and manifest paths inside the repository for that reader. The figure command also writes the full-scale table included by `reports/report.tex`. See the [P4 results and team handoff](reports/p4_phase_b.md) for the recipient files and scoring rehearsal.

## Phase B boosting and alert budgets

Run the bounded P5 search on a paired development manifest with the boosting dependencies installed:

```bash
python -m src.phase_b_boosting --manifest data/sample/feature_manifest.json --output reports/p5_phase_b_pilot
python -m src.plot_boosting reports/p5_phase_b_pilot
```

For a frozen full-scale feature group, pass its manifest and a separate output directory. Each outer fold and seed shares P1's month-stratified training draw across candidates. The summary step also writes `aggregator_inputs/` for the `--boosting` argument of P1's results aggregator. Pair it with classical results from the same experiment.

## P2 Phase B audit, drift and calibration

See [P2 checks, findings and handoffs](reports/p2_phase_b.md). Use a **new Python 3.12 environment** with the three requirements files; `requirements-p2-lock.txt` records the resolved pinned environment. From the repository root:

```sh
python -m src.build_sample_features
python -m src.p2_evidence
python -m src.p2_registry
python -m src.calibration --manifest data/sample/feature_manifest.json
python -m src.drift --manifest data/sample/feature_manifest.json
python -m unittest discover -s tests -v
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf reports/report.tex
```

Both analysis CLIs use the reviewed derived-feature gate; `--manifest` supports later full-scale matrices and calibration `--config` accepts the rehearsal configuration format. The supplied pilot uses the **unmasked Phase A** file; full P3 group matrices, weather/inbound EDA and selected-model calibration remain pending. No final scorer is invoked.

`python -m src.eda` regenerates the source audit, tables and figures. It writes computed issues to `reports/data_quality_generated.csv` and observations to `reports/eda_generated.md`; curated audit prose, limitations, reviews, issue rows and prior observations are preserved. New source audit decisions belong in `src.eda.audit_columns`, derived feature definitions in `src.p2_registry`, and written scoped approvals in `reports/audit_assumptions.json`.

### P2 weather and selected-prediction handoff

The user-selected missing indicator is `weather_missing`; P3 must rename its delivered `wx_weather_missing` and regenerate matrices/manifests/hashes. The [weather audit](reports/weather_audit.md) records the six pending candidates, timing/missingness findings and PBI/PPG mapping review. `WEATHER_SOURCE_AUDIT.csv` inventories external lineage separately from the source-Parquet audit. None of these rows grants checkpoint approval while A-WEATHER is pending.

After P4/P5 deliver their actual selected validation exports and matching provenance, copy and fill [the handoff template](reports/templates/selected_calibration_handoff.json), then run:

```sh
python -m src.selected_calibration --handoff PATH_TO_FILLED_HANDOFF.json
```

This reads the selected probabilities without refitting, checks exact validation identities and reproduced scores, and writes calibration metrics/figures separately under `reports/selected_calibration/`. Full G2/G3 EDA/drift and selected-model conclusions remain pending actual audited matrices and prediction exports; the earlier default-config pilot remains labelled as such. Do not run the mixed-period weather loader for P2 analysis.

## Flight Explorer interface

Open [ui/index.html](ui/index.html) in a browser for the offline Flight Explorer. The Prediction page replays paired six-hour and one-hour LightGBM pilot predictions for 8,000 validation flights (four folds, seed 42). Choose a UTC date and flight, select **Predict Delay**, then **Reveal outcome** to compare the saved forecast with the recorded departure. Routes links each city pair to a flight. Results shows the separate full-development feature ablation, and About explains the data and links the report.

The browser dataset joins the seed-42 predictions in `reports/p5_month_pilot/predictions/` to the Phase A development Parquet by flight ID and embeds the comparison metrics from `reports/tables/p3_ablation_results.csv`. No backend, credentials, or live training process is needed.
