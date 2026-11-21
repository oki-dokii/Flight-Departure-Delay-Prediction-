"""Evaluate binary classifiers across the frozen development folds and seeds."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from src.splits import get_folds

SEEDS = (42, 73, 101)


def evaluate(model_factory, X, development, seeds=SEEDS):
    """Fit a fresh model per fold/seed and return per-run and summary DataFrames.

    model_factory(seed) must return an unfitted estimator with fit, predict_proba
    and classes_. Put learned preprocessing inside that estimator's pipeline.
    X and development must have matching row indices in matching order.
    """
    if not X.index.equals(development.index):
        raise ValueError("Features and development metadata must have identical indices")
    if not development["LATE"].isin([0, 1]).all():
        raise ValueError("LATE must contain only 0 and 1")
    if "LATE" in X:
        raise ValueError("LATE is a target, not a feature")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least one distinct seed")

    y = development["LATE"]
    runs = []
    for fold, (train, validate) in enumerate(get_folds(development), start=1):
        y_train, y_validate = y.iloc[train], y.iloc[validate]
        if y_train.nunique() != 2 or y_validate.nunique() != 2:
            raise ValueError(f"Fold {fold} requires both classes in train and validation")
        X_train, X_validate = X.iloc[train], X.iloc[validate]
        for seed in seeds:
            estimators = {
                "model": model_factory(seed),
                "majority": DummyClassifier(strategy="most_frequent"),
            }
            for name, estimator in estimators.items():
                estimator.fit(X_train, y_train)
                classes = np.asarray(estimator.classes_)
                positive_column = np.flatnonzero(classes == 1).item()
                probabilities = estimator.predict_proba(X_validate)
                scores = probabilities[:, positive_column]
                runs.append({
                    "estimator": name,
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
