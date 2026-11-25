"""Combine one classical run and five boosting runs for the same declared experiment."""
import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd

from src.evaluate import SEEDS
from src.models_boosting import MODELS
from src.models_classical import classical_factories, phase_b_roster
from src.run_classical import SCHEDULE_SCHEMA

KEY = ["model", "checkpoint_minutes", "fold", "seed"]
METRICS = ["roc_auc", "pr_auc"]
PROVENANCE = ["revision", "fit_mode", "feature_group", "train_rows", "validation_rows",
              "train_ids_sha256", "validation_ids_sha256"]
INPUT_IDENTITY = ["feature_group", "feature_revision", "development_release", "development_sha256",
                  "manifest_sha256", "feature_sha256", "audit_sha256", "cohort_ids_sha256", "pilot", "fit_mode",
                  "revision"]


def check_provenance(classical, boosting, master):
    left = json.loads((classical / "experiment.json").read_text())
    right = json.loads((boosting / "experiment.json").read_text())
    mismatch = [key for key in INPUT_IDENTITY if key not in left or key not in right or left[key] != right[key]]
    if mismatch:
        raise ValueError(f"Incompatible benchmark inputs: {', '.join(mismatch)}")
    if any(key not in master or master[key].isna().any() for key in PROVENANCE):
        raise ValueError("Missing per-fit benchmark provenance")
    for key in ("revision", "feature_group", "fit_mode"):
        if not master[key].eq(left[key]).all():
            raise ValueError(f"Score rows disagree with the experiment {key}")
    matched = master.groupby(KEY[1:])[PROVENANCE].nunique()
    if not matched.eq(1).all().all():
        raise ValueError("Benchmark models used different train or validation cohorts")


def aggregate(classical, boosting, experiment, output, test_scores=None):
    frames = [pd.read_csv(classical / "runs.csv")]
    for minutes, model in product((360, 60), MODELS):
        rows = pd.read_csv(boosting / str(minutes) / model / "runs.csv")
        frames.append(rows.loc[rows.estimator.eq("model")].drop(columns="estimator").assign(
            model=model, checkpoint_minutes=minutes))
    master = pd.concat(frames, ignore_index=True).sort_values(KEY)
    recipe_file = classical / "selected_recipes.json"
    if recipe_file.exists():
        pair = json.loads(recipe_file.read_text())["classical_pair"]
        roster = phase_b_roster(pair)
        check_provenance(classical, boosting, master)
    else:
        roster = tuple(classical_factories(SCHEDULE_SCHEMA))
    families = (*roster, *MODELS)
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
        if (set(test[KEY[:2]].itertuples(index=False, name=None)) != set(product(families, (360, 60)))
                or not test[["test_roc_auc", "test_pr_auc"]].apply(lambda s: s.between(0, 1)).all().all()):
            raise ValueError("Invalid test score keys or metrics")
        summary = summary.merge(test, on=KEY[:2], how="left", validate="one_to_one")
        if summary.test_roc_auc.isna().any() or summary.test_pr_auc.isna().any():
            raise ValueError("Incomplete test results")
    output.mkdir(parents=True, exist_ok=True)
    master.to_csv(output / "master_results.csv", index=False)
    summary.assign(experiment=experiment).to_csv(output / "results_summary.csv", index=False)

    lines = [f"# {experiment}", "", "| Model | Checkpoint | Validation ROC-AUC | Validation PR-AUC | Test ROC-AUC | Test PR-AUC |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        scores = [f"{getattr(row, m + '_mean'):.3f} ± {getattr(row, m + '_std'):.3f}" for m in METRICS]
        final = [f"{getattr(row, 'test_' + m):.3f}" if test_scores else "Pending" for m in METRICS]
        lines.append("| " + " | ".join([row.model, f"{row.checkpoint_minutes // 60} h", *scores, *final]) + " |")
    (output / "results_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("classical", "boosting"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--test-scores", type=Path)
    aggregate(**vars(parser.parse_args()))
