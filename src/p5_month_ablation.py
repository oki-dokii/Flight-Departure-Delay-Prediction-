"""Paired month ablation and calibration handoff for selected boosting recipes."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.evaluate import SEEDS, revision
from src.models_boosting import MODELS
from src.models_classical import FeatureSchema
from src.phase_b_boosting import pipeline, scores, tune_run
from src.run_classical import checkpoint_features, paired_metadata, read_manifest
from src.splits import TIME, get_folds


def ids_digest(index):
    return hashlib.sha256("\n".join(index.astype(str)).encode()).hexdigest()


def run(manifest, selected, output):
    inputs = read_manifest(manifest)
    meta = paired_metadata(inputs)
    recipes = pd.read_csv(selected)
    key = ["model", "checkpoint_minutes", "fold", "seed"]
    recipes = recipes.set_index(key, verify_integrity=True)
    if "month" not in inputs.schema.columns:
        raise ValueError("Month ablation requires the numeric month predictor")
    small = FeatureSchema(tuple(p for p in inputs.schema.predictors if p[0] != "month"))
    out, revision_id = [], revision()
    for cp in inputs.checkpoints:
        features = checkpoint_features(cp, inputs.schema, meta.index)
        for fold, (train, valid) in enumerate(get_folds(meta), 1):
            validation = meta.iloc[valid]
            for seed in SEEDS:
                for name in MODELS:
                    ref = recipes.loc[(name, cp.minutes, fold, seed)]
                    for variant, schema in [("with_month", inputs.schema), ("without_month", small)]:
                        X = features.loc[:, schema.columns]
                        start = perf_counter()
                        def make(rounds):
                            return pipeline(name, schema, seed, json.loads(ref.parameters), ref.native, rounds)
                        model, rounds, sampled = tune_run(X, meta, train, valid, seed, make)
                        proba = model.predict_proba(X.iloc[valid])[:, 1]
                        metrics = scores(validation.LATE, proba)
                        if variant == "with_month" and not np.allclose(
                            [metrics["pr_auc"], metrics["roc_auc"]], [ref.pr_auc, ref.roc_auc], atol=1e-10, rtol=0
                        ):
                            raise ValueError(f"Reference fit differs: {name}, {cp.minutes}, {fold}, {seed}")
                        row = dict(model=name, checkpoint_minutes=cp.minutes, fold=fold, seed=seed,
                                   variant=variant, rounds=rounds, parameters=ref.parameters, native=ref.native,
                                   revision=revision_id, train_rows=len(sampled), validation_rows=len(valid),
                                   train_id_sha256=ids_digest(meta.index[sampled]),
                                   validation_id_sha256=ids_digest(validation.index),
                                   runtime_seconds=perf_counter()-start, **metrics)
                        out.append(row)
                        if variant == "with_month":
                            directory = output / "predictions" / str(cp.minutes) / name
                            directory.mkdir(parents=True, exist_ok=True)
                            validation.LATE.to_frame().assign(probability=proba,
                                day=validation[TIME].dt.floor("D"), model=name, checkpoint_minutes=cp.minutes,
                                fold=fold, seed=seed, revision=revision_id).to_parquet(directory/f"{fold}-{seed}.parquet")
            print(f"Completed {cp.minutes} minutes, fold {fold}", flush=True)
    results = pd.DataFrame(out)
    results.to_csv(output / "runs.csv", index=False)
    differences = []
    for metric in ("pr_auc", "roc_auc"):
        pair = results.pivot(index=key, columns="variant", values=metric)
        differences.append((pair.without_month - pair.with_month).rename(metric))
    delta = pd.concat(differences, axis=1)
    delta.to_csv(output / "paired_deltas.csv")
    delta.groupby(level=key[:2]).mean().to_csv(output / "mean_deltas.csv")
    (output / "provenance.json").write_text(json.dumps(dict(manifest_sha256=inputs.fingerprint,
        source_release=inputs.release, feature_revision=inputs.revision, cohort_rows=len(meta),
        cohort_ids_sha256=ids_digest(meta.index), code_revision=revision_id,
        selected_runs_sha256=hashlib.sha256(selected.read_bytes()).hexdigest()), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "selected", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    run(**vars(parser.parse_args()))
