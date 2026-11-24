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
