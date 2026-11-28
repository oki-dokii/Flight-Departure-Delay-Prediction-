"""Rehearse on fold four, or score frozen paired holdout features once."""
import argparse
import hashlib
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import fit_indices, positive_scores, revision
from src.models_boosting import MODELS, make_booster
from src.models_classical import FeatureSchema, classical_factories, classical_from_recipe, phase_b_roster
from src.phase_b_boosting import fit_with_stopping, pipeline
from src.phase_b_classical import file_digest
from src.run_classical import METADATA, _read_parquet, checkpoint_features, paired_metadata, read_manifest
from src.splits import DEVELOPMENT_END, OBSERVED, TEST_END, TEST_START, TIME, VALIDATION_WINDOWS, get_folds

FINAL = Path("data/processed/final_once")


def configured_booster(name, schema, seed, params, rounds=None):
    if "rounds" not in params:
        return make_booster(name, schema, seed).set_params(**params)
    return pipeline(name, schema, seed, params["parameters"], params["native"], rounds or params["rounds"])


def run(manifest, config, holdout_manifest=None, rehearsal_output=None, capacity_rehearsal=False):
    if rehearsal_output and (holdout_manifest or rehearsal_output.resolve() == FINAL.resolve()):
        raise ValueError("Choose a separate output directory for development rehearsal")
    if capacity_rehearsal and holdout_manifest:
        raise ValueError("Capacity rehearsal uses development data only")
    dev = read_manifest(manifest)
    metadata = paired_metadata(dev)
    settings = json.loads(config.read_text())
    schema = FeatureSchema(tuple(tuple(item) for item in settings.get("predictors", dev.schema.predictors)))
    if (any(predictor not in dev.schema.predictors for predictor in schema.predictors)
            or (holdout_manifest and schema != dev.schema)):
        raise ValueError("Configured predictors differ from the approved manifest")
    factories = classical_factories(schema)
    recipes = settings.get("classical_recipes")
    expected_inputs = settings.get("development_inputs")
    if recipes is not None and holdout_manifest and expected_inputs is None:
        raise ValueError("Phase B final scoring requires the frozen development input hashes")
    if expected_inputs is not None:
        actual_inputs = dict(manifest_sha256=dev.fingerprint, development_sha256=file_digest(dev.development),
                             feature_sha256={str(cp.minutes): file_digest(cp.path) for cp in dev.checkpoints})
        if expected_inputs != actual_inputs:
            raise ValueError("Development inputs differ from the frozen configuration")
    roster = phase_b_roster(settings["classical_pair"]) if recipes is not None else tuple(factories)
    families = set(MODELS) if recipes is not None else set(roster) | set(MODELS)
    for cp in dev.checkpoints:
        minutes = str(cp.minutes)
        if set(settings["models"][minutes]) != families or (
                recipes is not None and set(recipes[minutes]) != set(roster)):
            raise ValueError("Declare one configuration for every model and checkpoint")
        if recipes is not None:
            for recipe in recipes[minutes].values():
                classical_from_recipe(schema, settings["seed"], recipe)
        else:
            for name in roster:
                factories[name](settings["seed"]).set_params(**settings["models"][minutes][name])
        for name in MODELS:
            params = settings["models"][minutes][name]
            configured_booster(name, schema, settings["seed"], params,
                               rounds=10 if params.get("rounds") == "training_selected" else None)
    seed = settings["seed"]
    if holdout_manifest or capacity_rehearsal:
        all_ids = _read_parquet(dev.development, ["flight_id"]).flight_id
        if len(all_ids) != len(metadata) or not metadata.index.isin(all_ids).all():
            raise ValueError("Final refit requires complete development matrices")
        del all_ids
        idx = np.flatnonzero((metadata[OBSERVED] < DEVELOPMENT_END) &
                             (metadata[OBSERVED] > metadata[TIME] - pd.Timedelta(hours=1)))
    if holdout_manifest:
        if settings.get("frozen_revision") != revision():
            raise ValueError("Final scoring requires the frozen code revision")
        holdout = read_manifest(holdout_manifest)
        if (holdout.schema != dev.schema or holdout.revision != dev.revision
                or {c.minutes for c in holdout.checkpoints} != {c.minutes for c in dev.checkpoints}):
            raise ValueError("Holdout feature definition differs from development")
        out = FINAL
        out.mkdir(parents=True, exist_ok=False)
        test = _read_parquet(holdout.development, ["flight_id", *METADATA]).set_index(
            "flight_id", verify_integrity=True)
        if not test[TIME].between(TEST_START, TEST_END, inclusive="left").all():
            raise ValueError("Holdout metadata outside frozen test window")
        scored = test.loc[test[OBSERVED] > test[TIME] - pd.Timedelta(hours=1)]
    elif capacity_rehearsal:
        out = rehearsal_output or Path("data/processed/final_capacity_rehearsal")
        out.mkdir(parents=True, exist_ok=True)
        batches = [valid for _, valid in get_folds(metadata)[-2:]]
        scored = metadata.iloc[np.concatenate(batches)]
    else:
        out = rehearsal_output or Path("data/processed/final_rehearsal")
        out.mkdir(parents=True, exist_ok=True)
        train, val = get_folds(metadata)[-1]
        idx = fit_indices(train, metadata, seed, settings.get("rehearsal_fit_mode", "sampled"))
        scored = metadata.iloc[val]

    results = []
    for cp in dev.checkpoints:
        features = checkpoint_features(cp, dev.schema, metadata.index).loc[:, schema.columns]
        if holdout_manifest:
            target = next(c for c in holdout.checkpoints if c.minutes == cp.minutes)
            x_test = checkpoint_features(target, dev.schema, test.index).loc[scored.index]
        else:
            x_test = features.loc[scored.index]
        minutes = str(cp.minutes)
        for name in (*roster, *MODELS):
            start = perf_counter()
            params = settings["models"][minutes].get(name, {})
            if name in roster:
                model = (classical_from_recipe(schema, seed, recipes[minutes][name]) if recipes is not None
                         else factories[name](seed).set_params(**params))
                model.fit(features.iloc[idx], metadata.LATE.iloc[idx])
            elif params.get("rounds") == "training_selected":
                def make(rounds):
                    return configured_booster(name, schema, seed, params, rounds)

                model, _ = fit_with_stopping(features, metadata, idx, make)
            else:
                model = configured_booster(name, schema, seed, params)
                model.fit(features.iloc[idx], metadata.LATE.iloc[idx])
            prob = positive_scores(model, x_test)
            if capacity_rehearsal:
                if not np.isfinite(prob).all():
                    raise ValueError(f"Nonfinite rehearsal probabilities for {name}")
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                results.append(dict(model=name, checkpoint_minutes=cp.minutes, rows=len(scored),
                                    seconds=perf_counter() - start,
                                    process_peak_bytes=peak if platform.system() == "Darwin" else peak * 1024))
                del model
                continue
            predictions = scored[["LATE"]].assign(probability=prob)
            predictions.to_parquet(out / f"{name}_{cp.minutes}.parquet")
            results.append(dict(model=name, checkpoint_minutes=cp.minutes, rows=len(scored),
                                test_roc_auc=roc_auc_score(scored.LATE, prob),
                                test_pr_auc=average_precision_score(scored.LATE, prob)))
            del model
        del features, x_test
    pd.DataFrame(results).to_csv(out / ("capacity.csv" if capacity_rehearsal else "scores.csv"), index=False)
    period_start = TEST_START if holdout_manifest else VALIDATION_WINDOWS[-2 if capacity_rehearsal else -1][0]
    (out / "run.json").write_text(json.dumps({
        "mode": "capacity_rehearsal" if capacity_rehearsal else "final" if holdout_manifest else "fold_four_rehearsal",
        "start_utc": str(period_start),
        "eligible_flights": len(scored), "training_rows": len(idx), "revision": revision(),
        "configuration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, help="One-time final scoring after freeze")
    parser.add_argument("--rehearsal-output", type=Path, help="Output directory for development rehearsal")
    parser.add_argument("--capacity-rehearsal", action="store_true",
                        help="Fit all development rows and time a development-only scoring workload")
    run(**vars(parser.parse_args()))
