"""Rehearse on fold four, or score frozen paired holdout features once."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import revision, training_sample
from src.models_boosting import MODELS, make_booster
from src.models_classical import classical_factories
from src.run_classical import METADATA, _read_parquet, checkpoint_features, paired_metadata, read_manifest
from src.splits import DEVELOPMENT_END, OBSERVED, TEST_END, TEST_START, TIME, get_folds

FINAL = Path("data/processed/final_once")


def run(manifest, config, holdout_manifest=None):
    dev = read_manifest(manifest)
    metadata = paired_metadata(dev)
    factories = classical_factories(dev.schema)
    settings = json.loads(config.read_text())
    families = set(factories) | set(MODELS)
    for cp in dev.checkpoints:
        if set(settings["models"][str(cp.minutes)]) != families:
            raise ValueError("Declare one configuration for every model and checkpoint")
    seed = settings["seed"]
    if holdout_manifest:
        if settings.get("frozen_revision") != revision():
            raise ValueError("Final scoring requires the frozen code revision")
        all_ids = _read_parquet(dev.development, ["flight_id"]).flight_id
        if len(all_ids) != len(metadata) or not metadata.index.isin(all_ids).all():
            raise ValueError("Final refit requires complete development matrices")
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
        idx = np.flatnonzero((metadata[OBSERVED] < DEVELOPMENT_END) &
                             (metadata[OBSERVED] > metadata[TIME] - pd.Timedelta(hours=1)))
        scored = test.loc[test[OBSERVED] > test[TIME] - pd.Timedelta(hours=1)]
    else:
        out = Path("data/processed/final_rehearsal")
        out.mkdir(parents=True, exist_ok=True)
        train, val = get_folds(metadata)[-1]
        idx = training_sample(train, metadata, seed)
        scored = metadata.iloc[val]

    results = []
    for cp in dev.checkpoints:
        features = checkpoint_features(cp, dev.schema, metadata.index)
        if holdout_manifest:
            target = next(c for c in holdout.checkpoints if c.minutes == cp.minutes)
            x_test = checkpoint_features(target, dev.schema, test.index).loc[scored.index]
        else:
            x_test = features.loc[scored.index]
        for name, params in settings["models"][str(cp.minutes)].items():
            model = factories[name](seed) if name in factories else make_booster(name, dev.schema, seed)
            model.set_params(**params)
            model.fit(features.iloc[idx], metadata.LATE.iloc[idx])
            prob = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
            predictions = scored[["LATE"]].assign(probability=prob)
            predictions.to_parquet(out / f"{name}_{cp.minutes}.parquet")
            results.append(dict(model=name, checkpoint_minutes=cp.minutes, rows=len(scored),
                                test_roc_auc=roc_auc_score(scored.LATE, prob),
                                test_pr_auc=average_precision_score(scored.LATE, prob)))
    pd.DataFrame(results).to_csv(out / "scores.csv", index=False)
    (out / "run.json").write_text(json.dumps({
        "mode": "final" if holdout_manifest else "fold_four_rehearsal",
        "start_utc": str(TEST_START if holdout_manifest else pd.Timestamp("2025-10-01", tz="UTC")),
        "eligible_flights": len(scored), "training_rows": len(idx), "revision": revision(),
        "configuration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, help="One-time final scoring after freeze")
    run(**vars(parser.parse_args()))
