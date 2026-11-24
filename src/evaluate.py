"""Evaluate binary classifiers across the frozen development folds and seeds."""

import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from src.alert_metrics import alert_metrics
from src.splits import TIME, TEST_START, get_folds, utc_column

SEEDS = (42, 73, 101)


def training_sample(train, development, seed):
    """D5: stable, common, month-stratified draws for every candidate."""
    rng = np.random.default_rng(seed)
    months = development[TIME].iloc[train].dt.month.to_numpy()
    fraction = min(0.9, 500_000 / len(train))
    samples = []
    for month in np.unique(months):
        rows = train[months == month]
        ordered = rows[np.argsort(development.index.to_numpy()[rows], kind="stable")]
        size = min(len(rows), max(2, int(len(rows) * fraction)))
        samples.append(rng.choice(ordered, size=size, replace=False))
    return np.sort(np.concatenate(samples))


def revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                            capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def evaluate(model_factory, X, development, seeds=SEEDS):
    """Fit a fresh model per fold/seed and return per-run and summary DataFrames.

    model_factory(seed) must return an unfitted estimator with fit, predict_proba
    and classes_. Put learned preprocessing inside that estimator's pipeline.
    X and development must have matching row indices in matching order.
    """
    if (utc_column(development, TIME) >= TEST_START).any():
        raise ValueError("Test-period rows in development input")
    if not X.index.equals(development.index):
        raise ValueError("Features and development metadata must have identical indices")
    if not development["LATE"].isin([0, 1]).all():
        raise ValueError("LATE must contain only 0 and 1")
    if "LATE" in X:
        raise ValueError("LATE is a target, not a feature")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least one distinct seed")

    commit = revision()
    y = development["LATE"]
    runs = []
    for fold, (train, validate) in enumerate(get_folds(development), start=1):
        y_train, y_validate = y.iloc[train], y.iloc[validate]
        if y_train.nunique() != 2 or y_validate.nunique() != 2:
            raise ValueError(f"Fold {fold} requires both classes in train and validation")
        X_validate = X.iloc[validate]
        for seed in seeds:
            sample = training_sample(train, development, seed)
            X_seed, y_seed = X.iloc[sample], y.iloc[sample]
            estimators = {
                "model": model_factory(seed),
                "majority": DummyClassifier(strategy="most_frequent"),
            }
            for name, estimator in estimators.items():
                start = perf_counter()
                estimator.fit(X_seed, y_seed)
                classes = np.asarray(estimator.classes_)
                positive_column = np.flatnonzero(classes == 1).item()
                probabilities = estimator.predict_proba(X_validate)
                scores = probabilities[:, positive_column]
                runs.append({
                    "estimator": name,
                    **alert_metrics(y_validate, scores),
                    "revision": commit,
                    "train_rows": len(sample), "validation_rows": len(validate),
                    "runtime_seconds": perf_counter() - start,
                    "parameters": json.dumps(estimator.get_params(deep=True), default=str, sort_keys=True),
                    "fold": fold,
                    "seed": seed,
                    "roc_auc": roc_auc_score(y_validate, scores),
                    "pr_auc": average_precision_score(y_validate, scores),
                    "accuracy": accuracy_score(y_validate, classes[probabilities.argmax(axis=1)]),
                })
    per_run = pd.DataFrame(runs)
    metrics = per_run.groupby("estimator", sort=False)[["roc_auc", "pr_auc", "accuracy"]]
    summary = pd.concat({"mean": metrics.mean(), "std": metrics.std(ddof=0)}, axis=1)
    return per_run, summary
