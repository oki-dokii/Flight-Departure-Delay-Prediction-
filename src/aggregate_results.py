"""Combine one classical run and five boosting runs for the same declared experiment."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import SEEDS
from src.models_boosting import MODELS
from src.models_classical import classical_factories
from src.run_classical import SCHEDULE_SCHEMA

KEY = ["model", "checkpoint_minutes", "fold", "seed"]
METRICS = ["roc_auc", "pr_auc"]


def aggregate(classical, boosting, experiment, output, test_scores=None):
    frames = [pd.read_csv(classical / "runs.csv")]
    for minutes, model in product((360, 60), MODELS):
        rows = pd.read_csv(boosting / str(minutes) / model / "runs.csv")
        frames.append(rows.loc[rows.estimator.eq("model")].drop(columns="estimator").assign(
            model=model, checkpoint_minutes=minutes))
    master = pd.concat(frames, ignore_index=True).sort_values(KEY)
    families = list(classical_factories(SCHEDULE_SCHEMA)) + list(MODELS)
    expected = set(product(families, (360, 60), range(1, 5), SEEDS))
    if master.duplicated(KEY).any() or set(master[KEY].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete or duplicate model/checkpoint/fold/seed results")
    if not master[METRICS].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("Invalid validation scores")
    master.insert(0, "experiment", experiment)
    groups = master.groupby(KEY[:2])[METRICS]
    summary = groups.mean().add_suffix("_mean").join(groups.std(ddof=0).add_suffix("_std")).reset_index()
    if test_scores:
        test = pd.read_csv(test_scores)
        summary = summary.merge(test, on=KEY[:2], how="left", validate="one_to_one")
        if summary.test_roc_auc.isna().any() or summary.test_pr_auc.isna().any():
            raise ValueError("Incomplete test results")
    output.mkdir(parents=True, exist_ok=True)
    master.to_csv(output / "master_results.csv", index=False)
    summary.assign(experiment=experiment).to_csv(output / "results_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("classical", "boosting"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--test-scores", type=Path)
    aggregate(**vars(parser.parse_args()))
