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
