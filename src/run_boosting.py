"""Evaluate one boosting family per process for comparable peak-memory measurements."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from src.evaluate import evaluate
from src.models_boosting import MODELS, make_booster
from src.run_classical import (
    DEVELOPMENT_PATH, SCHEDULE_SCHEMA, checkpoint_features, paired_metadata,
    read_manifest, schedule_sample,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--schedule-only", action="store_true")
    source.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/p5"))
    args = parser.parse_args()
    if args.schedule_only:
        features, metadata = schedule_sample(DEVELOPMENT_PATH)
        schema = SCHEDULE_SCHEMA
        datasets = [("schedule", features)]
    else:
        inputs = read_manifest(args.input_manifest)
        schema = inputs.schema
        metadata = paired_metadata(inputs)
        datasets = ((str(cp.minutes), checkpoint_features(cp, schema, metadata.index))
                    for cp in inputs.checkpoints)

    for checkpoint, features in datasets:
        start = perf_counter()
        runs, summary = evaluate(
            lambda seed: make_booster(args.model, schema, seed), features, metadata)
        elapsed = perf_counter() - start
        dest = args.output / checkpoint / args.model
        dest.mkdir(parents=True, exist_ok=True)
        runs.to_csv(dest / "runs.csv", index=False)
        summary.to_csv(dest / "summary.csv")
        timing = {"rows": len(features), "evaluation_seconds": elapsed}
        if sys.platform in ("darwin", "linux"):
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            timing["process_peak_mib"] = peak / (2**20 if sys.platform == "darwin" else 1024)
        (dest / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")


if __name__ == "__main__":
    main()
