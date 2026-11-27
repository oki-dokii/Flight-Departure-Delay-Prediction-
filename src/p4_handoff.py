"""Verify selected P4 validation exports and index the files for P2."""

import argparse
import json
from itertools import product
from os.path import abspath
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluate import SEEDS
from src.models_classical import FeatureSchema, classical_from_recipe, phase_b_roster
from src.phase_b_classical import CHECKPOINTS, FIT_KEYS, build_fits, file_digest, id_digest, write_json
from src.run_classical import ROOT, paired_metadata, read_manifest
from src.splits import TIME

MODELS = (*phase_b_roster("random_forest"), "logistic", "pca_logistic")
KEYS = ["model", "checkpoint_minutes", "fold", "seed", "fit_mode"]
METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score, "brier": brier_score_loss}


def verify(manifest, folder):
    manifest_path = Path(abspath(manifest)).relative_to(ROOT)
    run_path = Path(abspath(folder)).relative_to(ROOT)
    identity = json.loads((folder / "experiment.json").read_text())
    settings = json.loads((folder / "selected_recipes.json").read_text())
    inputs = read_manifest(manifest)
    mode = identity["fit_mode"]
    recipe_predictors = settings.get("predictors", identity["predictors"] if identity["pilot"] else None)
    if (inputs.fingerprint != identity["manifest_sha256"]
            or file_digest(inputs.development) != identity["development_sha256"]
            or {str(cp.minutes): file_digest(cp.path) for cp in inputs.checkpoints} != identity["feature_sha256"]
            or settings["rehearsal_fit_mode"] != mode
            or recipe_predictors != identity["predictors"]):
        raise ValueError("Manifest, feature files, or recipe metadata differ from the recorded experiment")

    metadata = paired_metadata(inputs)
    if (len(metadata) != identity["cohort_rows"] or id_digest(metadata.index) != identity["cohort_ids_sha256"]):
        raise ValueError("Loaded development cohort differs from the recorded experiment")
    fits = build_fits(metadata, (mode,))[mode]
    by_fold_seed = {(keys["fold"], keys["seed"]): (valid, keys) for _, valid, keys in fits}
    cohort_days = metadata[TIME].dt.floor("D")

    study = pd.read_csv(folder / "selected_study.csv").set_index(KEYS, verify_integrity=True).sort_index()
    checkpoints = tuple(identity.get("checkpoint_minutes", CHECKPOINTS))
    if checkpoints not in (CHECKPOINTS, (60,)):
        raise ValueError("Unexpected selected-study checkpoints")
    expected = set(product(MODELS, checkpoints, range(1, 5), SEEDS, (mode,)))
    if set(study.index) != expected:
        raise ValueError("Incomplete or unexpected selected-study score keys")
    if not study.groupby(level=["model", "checkpoint_minutes"]).parameters.nunique().eq(1).all():
        raise ValueError("Selected recipes must be constant across each checkpoint's folds and seeds")
    roster = phase_b_roster(settings["classical_pair"])
    runs = pd.read_csv(folder / "runs.csv").set_index(KEYS, verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(runs, study.loc[study.index.isin(roster, level="model")])

    schema = FeatureSchema(tuple(map(tuple, identity["predictors"])))
    records = []
    for (model, minutes, fold, seed, fit_mode), score in study.iterrows():
        valid, fit = by_fold_seed[fold, seed]
        expected_y = metadata.LATE.iloc[valid]
        recipe = json.loads(score.parameters)
        classical_from_recipe(schema, seed, recipe)
        if model in roster and recipe != settings["classical_recipes"][str(minutes)][model]:
            raise ValueError(f"Selected recipe differs from the score row for {model}/{minutes}")
        for key in ("train_rows", "validation_rows", "train_ids_sha256", "validation_ids_sha256"):
            if score[key] != fit[key]:
                raise ValueError(f"{key} differs from the protocol for {model}/{minutes}/{fold}/{seed}")
        if score.feature_group != identity["feature_group"] or score.revision != identity["revision"]:
            raise ValueError("Selected scores have inconsistent experiment metadata")

        path = Path("predictions", str(minutes), model, f"{fold}-{seed}.parquet")
        predictions = pd.read_parquet(folder / path).set_index("flight_id", verify_integrity=True)
        if not predictions.index.equals(expected_y.index):
            raise ValueError(f"Unexpected validation flight IDs in {path}")
        np.testing.assert_array_equal(predictions.LATE, expected_y, err_msg=str(path))
        pd.testing.assert_series_equal(predictions.day, cohort_days.iloc[valid], check_names=False)
        keys = dict(zip(FIT_KEYS, (minutes, fold, seed, fit_mode), strict=True))
        keys.update(model=model, run_id=identity["run_id"], feature_group=identity["feature_group"])
        for field, value in keys.items():
            if not predictions[field].eq(value).all():
                raise ValueError(f"Unexpected {field} in {path}")
        probabilities = predictions.probability.to_numpy()
        if not np.isfinite(probabilities).all() or not np.logical_and(0 <= probabilities, probabilities <= 1).all():
            raise ValueError(f"Invalid probabilities in {path}")
        for metric, compute in METRICS.items():
            np.testing.assert_allclose(compute(expected_y, probabilities), score[metric],
                                       rtol=0, atol=1e-12, err_msg=f"{path}: {metric}")
        records.append(dict(path=path.as_posix(), rows=len(predictions), sha256=file_digest(folder / path), **keys))

    index = pd.DataFrame(records)
    temporary = folder / "prediction_index.tmp"
    index.to_csv(temporary, index=False)
    temporary.replace(folder / "prediction_index.csv")
    # The G0 pilot projected G1 columns from its shared pilot manifest. P2's
    # calibration reader audits the entire declared predictor schema for the group.
    if inputs.schema == schema:
        write_json(folder / "p2_handoff.json", {
            "feature_manifest": manifest_path.as_posix(),
            "sources": [{
                "owner": "P4",
                "experiment": (run_path / "experiment.json").as_posix(),
                "runs": (run_path / "selected_study.csv").as_posix(),
                "predictions": (run_path / "predictions").as_posix(),
            }],
        })
    print(f"{identity['feature_group']}: verified {len(index)} files and {index.rows.sum():,} validation probabilities")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    verify(args.manifest, args.run)
