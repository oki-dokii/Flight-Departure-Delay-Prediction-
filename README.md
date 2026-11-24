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

Use `--group G0 --pilot` with the same released manifest for a schedule-only projection. For full-scale work, pass P3’s complete manifest for each group, omit `--pilot`, and pass P2’s approved feature registry with `--audit` for G2. See the [P4 results and team handoff](reports/p4_phase_b.md) for the exports and shared rehearsal procedure.

## Phase B boosting and alert budgets

Run the bounded P5 search on a paired development manifest with the boosting dependencies installed:

```bash
python -m src.phase_b_boosting --manifest data/sample/feature_manifest.json --output reports/p5_phase_b_pilot
python -m src.plot_boosting reports/p5_phase_b_pilot
```

For a frozen full-scale feature group, pass its manifest and a separate output directory. Each outer fold and seed shares P1's month-stratified training draw across candidates. The summary step also writes `aggregator_inputs/` for the `--boosting` argument of P1's results aggregator. Pair it with classical results from the same experiment.
