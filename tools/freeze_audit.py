"""P1 checks for the primary validation master and completed prediction exports.

Invoke from the repository root with python -m tools.freeze_audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.aggregate_results import aggregate
from src.models_boosting import MODELS
from src.models_classical import phase_b_roster
from src.phase_b_classical import file_digest
from src.run_classical import METADATA, _read_parquet, read_manifest
from src.splits import OBSERVED, TEST_END, TEST_START, TIME


def validation(manifest, classical, boosting, output, test_scores=None):
    inputs = read_manifest(manifest)
    if inputs.sampled_ids is not None:
        raise ValueError("Primary master requires a full-development manifest")
    for source in (classical, boosting):
        identity = json.loads((source / "experiment.json").read_text())
        required = dict(pilot=False, feature_group="G1", fit_mode="full_fold",
                        development_sha256=file_digest(inputs.development),
                        manifest_sha256=inputs.fingerprint,
                        feature_sha256={str(cp.minutes): file_digest(cp.path) for cp in inputs.checkpoints})
        if any(identity.get(k) != v for k, v in required.items()):
            raise ValueError(f"Primary experiment identity mismatch in {source}")
    if not (classical / "selected_recipes.json").is_file():
        raise ValueError("Final master requires selected recipes and per-fit provenance")
    if output.exists():
        raise FileExistsError(output)
    aggregate(classical, boosting, "full_G1", output, test_scores=test_scores)


def rebuild(folder, config, holdout_manifest, output):
    settings = json.loads(config.read_text())
    record = json.loads((folder / "run.json").read_text())
    if (record["mode"] != "final" or record["revision"] != settings["frozen_revision"]
            or record["configuration_sha256"] != hashlib.sha256(config.read_bytes()).hexdigest()):
        raise ValueError("Final run identity differs from frozen configuration")
    holdout = read_manifest(holdout_manifest)
    meta = _read_parquet(holdout.development, ["flight_id", *METADATA]).set_index("flight_id", verify_integrity=True)
    if (meta.index.hasnans or meta[OBSERVED].isna().any()
            or not meta[TIME].between(TEST_START, TEST_END, inclusive="left").all()):
        raise ValueError("Invalid holdout metadata")
    expected = meta.loc[meta[OBSERVED] > meta[TIME] - pd.Timedelta(hours=1), "LATE"].sort_index()
    if len(expected) != record["eligible_flights"] or set(expected.unique()) != {0, 1}:
        raise ValueError("Unexpected eligible test population")
    pairs = [(name, cp) for name in (*phase_b_roster(settings["classical_pair"]), *MODELS) for cp in (360,60)]
    if {p.name for p in folder.glob("*.parquet")} != {f"{m}_{cp}.parquet" for m, cp in pairs}:
        raise ValueError("Missing or unexpected prediction files")
    results = []
    for model, cp in pairs:
        frame = pd.read_parquet(folder / f"{model}_{cp}.parquet").sort_index()
        if frame.index.name != "flight_id" or not frame.index.is_unique or not frame.index.equals(expected.index):
            raise ValueError(f"Eligible flight IDs or N differ for {model} at {cp}")
        np.testing.assert_array_equal(frame.LATE, expected)
        p = frame.probability.to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probability")
        results.append(dict(model=model, checkpoint_minutes=cp, rows=len(expected),
                            test_pr_auc=average_precision_score(expected, p), test_roc_auc=roc_auc_score(expected, p)))
    result = pd.DataFrame(results).set_index(["model","checkpoint_minutes"]).sort_index()
    if (folder / "scores.csv").exists():
        original = pd.read_csv(folder / "scores.csv").set_index(["model","checkpoint_minutes"], verify_integrity=True).sort_index()
        pd.testing.assert_frame_equal(result, original[result.columns], atol=1e-12, rtol=0)
    result.to_csv(output, mode="x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, fields in [("validation", ("manifest","classical","boosting","output")),
                            ("rebuild", ("folder","config","holdout_manifest","output"))]:
        sub = commands.add_parser(command)
        for field in fields:
            sub.add_argument("--"+field.replace("_","-"), dest=field, required=True, type=Path)
        if command == "validation":
            sub.add_argument("--test-scores", type=Path)
    args = vars(parser.parse_args())
    action = args.pop("command")
    (validation if action == "validation" else rebuild)(**args)
