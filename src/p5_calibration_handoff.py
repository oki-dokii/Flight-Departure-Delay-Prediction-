"""Package selected with-month pilot validation predictions for Person 2."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.run_classical import read_manifest


def package(source, manifest, output):
    inputs = read_manifest(manifest)
    experiment = json.loads((source / "provenance.json").read_text())
    if inputs.sampled_ids is None or experiment["manifest_sha256"] != inputs.fingerprint:
        raise ValueError("Expected the pilot manifest used for the ablation")
    if set(inputs.schema.columns) != {
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month", "crs_elapsed_min",
        "distance", "carrier", "origin", "dest", "origin_departures_3h", "origin_late_rate_3h",
    }:
        raise ValueError("Expected the paired G1 pilot schema")
    with inputs.development.open("rb") as stream:
        development_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    experiment.update(run_id="p5-g1-pilot-" + experiment["code_revision"][:12],
                      fit_mode="sampled", feature_group="G1", pilot=True,
                      development_sha256=development_sha)
    runs = pd.read_csv(source / "runs.csv").query('variant == "with_month"').copy()
    runs["fit_mode"] = experiment["fit_mode"]
    for row in runs.itertuples():
        path = Path("predictions", str(row.checkpoint_minutes), row.model, f"{row.fold}-{row.seed}.parquet")
        frame = pd.read_parquet(source / path).reset_index()
        for field in ("run_id", "fit_mode", "feature_group"):
            frame[field] = experiment[field]
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output / path, index=False)
    runs.to_csv(output / "selected_runs.csv", index=False)
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    handoff = dict(feature_manifest=str(manifest), sources=[dict(owner="P5",
                   experiment=str(output / "experiment.json"), runs=str(output / "selected_runs.csv"),
                   predictions=str(output / "predictions"))])
    (output / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    package(**vars(parser.parse_args()))
