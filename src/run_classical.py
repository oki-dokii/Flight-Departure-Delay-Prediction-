"""Run the fixed Phase A configurations on paired P3 data or P1's schedule sample."""

import argparse
import hashlib
import json
import platform
import subprocess
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning

from src.evaluate import SEEDS, evaluate
from src.models_classical import PCA_COMPONENTS, FeatureSchema, classical_factories
from src.prepare import hhmm_minutes
from src.splits import DEVELOPMENT_END, GAP, OBSERVED, START, TIME, get_folds, utc_column

ROOT = Path(__file__).resolve().parents[1]
FLIGHT_ID = "flight_id"
METADATA = [TIME, OBSERVED, "LATE"]
METRICS = ["roc_auc", "pr_auc", "accuracy"]
PCA_DIAGNOSTICS = [
    "checkpoint_minutes", "fold", "seed", "train_rows", "encoded_width", "encoded_density",
    "component", "explained_variance_ratio", "cumulative_variance_ratio",
]
DEVELOPMENT_PATH = Path("data/processed/flights_development.parquet")
DEVELOPMENT_RELEASE = "phase-a-data-v1"
SAMPLE_PER_MONTH = 2_000
SAMPLE_SEED = 42
SCHEDULE_SCHEMA = FeatureSchema((
    ("scheduled_local_minutes", "numeric"), ("CRSElapsedTime", "numeric"),
    ("Distance", "numeric"), ("Reporting_Airline", "categorical"),
    ("Origin", "categorical"), ("Dest", "categorical"),
))


@dataclass(frozen=True)
class Checkpoint:
    minutes: int
    path: Path


@dataclass(frozen=True)
class PairedInputs:
    development: Path
    release: str
    revision: str
    schema: FeatureSchema
    checkpoints: tuple[Checkpoint, ...]
    sampled_ids: Path | None
    fingerprint: str


def _required_text(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty string without surrounding whitespace")
    return value


def _input_path(value, field):
    return (ROOT / _required_text(value, field)).resolve()


def read_manifest(path):
    contents = path.read_bytes()
    document = json.loads(contents)
    try:
        development = document["development"]
        release = _required_text(development["release"], "development.release")
        revision = _required_text(document["feature_spec_revision"], "feature_spec_revision")
        schema = FeatureSchema(tuple(
            (predictor["name"], predictor["role"]) for predictor in document["predictors"]
        ))
        checkpoints = []
        for entry in document["checkpoints"]:
            offset = entry["offset_minutes"]
            if type(offset) is not int or offset <= 0:
                raise ValueError("Each checkpoint requires a positive integer offset_minutes")
            if offset > GAP // pd.Timedelta(minutes=1):
                raise ValueError(f"Checkpoint {offset} min exceeds P1's {GAP} fold gap. "
                                 "Reconcile training-label availability with P1 before running.")
            if entry["source_release"] != release:
                raise ValueError(f"Checkpoint {offset} min source release differs from development")
            checkpoints.append(Checkpoint(offset, _input_path(entry["path"], "checkpoint path")))
        metadata_path = _input_path(development["path"], "development.path")
        sample_path = document.get("sampled_ids")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid checkpoint manifest structure: {exc}") from exc
    if (len(checkpoints) != 2 or len({cp.minutes for cp in checkpoints}) != 2
            or len({cp.path for cp in checkpoints}) != 2):
        raise ValueError("Declare exactly two distinct checkpoint offsets and file paths")
    return PairedInputs(
        metadata_path, release, revision, schema, tuple(checkpoints),
        _input_path(sample_path, "sampled_ids") if sample_path is not None else None,
        hashlib.sha256(contents).hexdigest(),
    )


def _read_parquet(path, columns, **kwargs):
    # Treat a physical flight_id field the same whether pandas saved it as an index or a column.
    return pq.read_table(path, columns=columns, **kwargs).to_pandas(ignore_metadata=True)


class TimedParquetReads:
    """Measure projected Parquet reads, decoding, and conversion to pandas."""

    def __init__(self):
        self.seconds = 0.0

    def __call__(self, path, columns, **kwargs):
        start = perf_counter()
        frame = _read_parquet(path, columns, **kwargs)
        self.seconds += perf_counter() - start
        return frame


def _flight_index(values, source):
    if values.empty or values.isna().any() or pd.api.types.infer_dtype(values) != "string":
        raise ValueError(f"{source} requires nonempty, nonmissing string flight IDs")
    if values.str.strip().ne(values).any() or values.eq("").any():
        raise ValueError(f"{source} contains blank or whitespace-padded flight IDs")
    ids = pd.Index(values, name=FLIGHT_ID)
    if ids.has_duplicates:
        raise ValueError(f"{source} contains duplicate flight IDs")
    return ids


def _check_file(path, predictors, declarations=None):
    schema = pq.read_schema(path)
    if schema.names.count(FLIGHT_ID) != 1:
        raise ValueError(f"{path} must contain exactly one {FLIGHT_ID} field")
    try:
        predictors.validate_columns(name for name in schema.names if name != FLIGHT_ID)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    metadata = schema.metadata or {}
    for key, value in (declarations or {}).items():
        if key.encode() in metadata and metadata[key.encode()] != str(value).encode():
            raise ValueError(f"{path}: Parquet {key} does not match the manifest declaration {value!r}")


def _match_ids(expected, actual, source):
    missing = expected.difference(actual, sort=False)
    extra = actual.difference(expected, sort=False)
    if len(missing) or len(extra):
        raise ValueError(f"{source} has different flight IDs "
                         f"(missing {len(missing)}, examples={missing[:3].tolist()}; "
                         f"extra {len(extra)}, examples={extra[:3].tolist()})")


def _development_dates(frame):
    scheduled = utc_column(frame, TIME)
    if not scheduled.between(START, DEVELOPMENT_END, inclusive="left").all():
        raise ValueError("Input must contain only the frozen development period")
    return scheduled


def paired_metadata(inputs, *, read=_read_parquet):
    """Validate both files and their common cohort before any configuration is fitted."""
    cohort = None
    for checkpoint in inputs.checkpoints:
        _check_file(checkpoint.path, inputs.schema, {
            "checkpoint_minutes": checkpoint.minutes, "source_release": inputs.release,
            "feature_spec_revision": inputs.revision,
        })
        ids = _flight_index(read(checkpoint.path, [FLIGHT_ID])[FLIGHT_ID], checkpoint.path)
        if cohort is None:
            cohort = ids.sort_values()
        else:
            _match_ids(cohort, ids, checkpoint.path)
    if inputs.sampled_ids is not None:
        if pq.read_schema(inputs.sampled_ids).names != [FLIGHT_ID]:
            raise ValueError("The sampled_ids Parquet must contain only flight_id")
        sampled = _flight_index(
            read(inputs.sampled_ids, [FLIGHT_ID])[FLIGHT_ID], inputs.sampled_ids,
        )
        _match_ids(sampled, cohort, "Paired checkpoints versus sampled_ids")

    development = read(inputs.development, [FLIGHT_ID, *METADATA])
    development.index = _flight_index(development.pop(FLIGHT_ID), inputs.development)
    _development_dates(development)
    missing = cohort.difference(development.index, sort=False)
    if len(missing):
        raise ValueError(f"{len(missing)} checkpoint flight IDs absent from P1 development "
                         f"(examples={missing[:3].tolist()})")
    return development.loc[cohort]


def checkpoint_features(checkpoint, schema, cohort, *, read=_read_parquet):
    frame = read(checkpoint.path, [FLIGHT_ID, *schema.columns])
    frame.index = _flight_index(frame.pop(FLIGHT_ID), checkpoint.path)
    _match_ids(cohort, frame.index, checkpoint.path)
    return schema.normalize(frame.loc[cohort])


def _sample_ids(development_path, read):
    targets = read(development_path, [FLIGHT_ID, TIME])
    targets.index = _flight_index(targets.pop(FLIGHT_ID), development_path)
    months = _development_dates(targets).dt.month
    generator = np.random.default_rng(SAMPLE_SEED)
    sampled_ids = []
    for _, month in targets.groupby(months, sort=True):
        ordered = month.index.sort_values()
        positions = generator.choice(len(ordered), size=min(SAMPLE_PER_MONTH, len(ordered)),
                                     replace=False)
        sampled_ids.extend(ordered.take(positions))
    return pd.Index(sampled_ids, name=FLIGHT_ID).sort_values()


def schedule_sample(development_path, *, read=_read_parquet):
    """Select up to 2,000 targets per scheduled UTC month without using labels."""
    cohort = _sample_ids(development_path, read)
    raw_columns = ["CRSDepTime", *SCHEDULE_SCHEMA.columns[1:]]
    sample = read(
        development_path, [FLIGHT_ID, *METADATA, *raw_columns],
        filters=[(FLIGHT_ID, "in", cohort.tolist())],
    )
    sample.index = _flight_index(sample.pop(FLIGHT_ID), development_path)
    _match_ids(cohort, sample.index, "Schedule sample")
    sample = sample.loc[cohort]
    sample["scheduled_local_minutes"] = hhmm_minutes(sample.pop("CRSDepTime")).mod(1440)
    return SCHEDULE_SCHEMA.normalize(sample[SCHEDULE_SCHEMA.columns]), sample[METADATA]


def fold_details(development):
    if not development["LATE"].isin([0, 1]).all():
        raise ValueError("P1's LATE must contain only 0 and 1")
    details = []
    for fold, (train, validate) in enumerate(get_folds(development), start=1):
        y_train = development["LATE"].iloc[train]
        y_validate = development["LATE"].iloc[validate]
        if y_train.nunique() != 2 or y_validate.nunique() != 2:
            raise ValueError(f"Fold {fold} requires both classes in train and validation")
        details.append({
            "fold": fold, "train_rows": len(train), "validation_rows": len(validate),
            "validation_prevalence": float(y_validate.mean()),
        })
    return details


def evaluate_configuration(name, factory, X, development, checkpoint, folds):
    """Use the evaluator's fold-major, seed-minor invocation order for fit diagnostics."""
    keys = [(fold["fold"], seed) for fold in folds for seed in SEEDS]
    diagnostics = []
    context = None
    estimator = None
    key_iterator = iter(keys)

    def record_pca():
        if name == "pca_logistic" and estimator is not None:
            pca = estimator.named_steps["pca"]
            train_rows, width = pca.training_shape_
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            diagnostics.extend({
                "checkpoint_minutes": checkpoint, "fold": context[0], "seed": context[1],
                "train_rows": train_rows, "encoded_width": width,
                "encoded_density": pca.training_density_, "component": component,
                "explained_variance_ratio": ratio, "cumulative_variance_ratio": total,
            } for component, (ratio, total) in enumerate(
                zip(pca.explained_variance_ratio_, cumulative), start=1,
            ))

    def tracked_factory(seed):
        nonlocal context, estimator
        record_pca()
        context = next(key_iterator, None)
        if context is None or context[1] != seed:
            raise RuntimeError("P1 evaluator factory order differs from the frozen fold/seed order")
        estimator = factory(seed)
        return estimator

    start = perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            per_run, summary = evaluate(tracked_factory, X, development, seeds=SEEDS)
    except (ValueError, ConvergenceWarning) as exc:
        horizon = "schedule-only" if checkpoint is None else f"{checkpoint} min"
        run = f", fold {context[0]}, seed {context[1]}" if context is not None else ""
        components = f", n_components={PCA_COMPONENTS}" if name == "pca_logistic" else ""
        raise ValueError(f"{name} at {horizon}{run}{components}: {exc}") from exc
    elapsed = perf_counter() - start
    record_pca()

    runs = per_run.loc[per_run["estimator"].eq("model"), ["fold", "seed", *METRICS]].reset_index(drop=True)
    baseline = per_run.loc[
        per_run["estimator"].eq("majority"), ["fold", "seed", *METRICS]
    ].reset_index(drop=True)
    if list(runs[["fold", "seed"]].itertuples(index=False, name=None)) != keys:
        raise ValueError(f"{name} has missing, duplicate, or unexpected P1 fold/seed results")
    if name == "majority" and not runs.equals(baseline):
        raise ValueError("Explicit majority results differ from P1's majority results")

    statistics = {
        f"{metric}_{stat}": summary.loc["model", (stat, metric)]
        for metric in METRICS for stat in ("mean", "std")
    }
    if not (np.isfinite(runs[METRICS].to_numpy()).all()
            and np.isfinite(list(statistics.values())).all()):
        raise ValueError(f"{name} returned nonfinite metrics")
    runs.insert(0, "checkpoint_minutes", checkpoint)
    runs.insert(0, "model", name)
    return runs, {"model": name, "checkpoint_minutes": checkpoint, **statistics}, elapsed, diagnostics


def _model_settings(factory):
    model = factory(SEEDS[0])
    if not hasattr(model, "named_steps"):
        return model.get_params(deep=False)
    settings = {"classifier": model.named_steps["classifier"].get_params(deep=False)}
    pca = model.named_steps.get("pca")
    if pca is not None:
        settings["pca"] = pca if pca == "passthrough" else pca.get_params(deep=False)
    return settings


def _code_revision():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() or None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-manifest", type=Path)
    source.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--development-parquet", type=Path, help="override the schedule-only source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(classical_factories(SCHEDULE_SCHEMA)))
    parser.add_argument("--checkpoint-minutes", type=int)
    args = parser.parse_args(argv)
    if args.schedule_only and args.checkpoint_minutes is not None:
        parser.error("--checkpoint-minutes requires --input-manifest")
    if args.input_manifest and args.development_parquet is not None:
        parser.error("--development-parquet requires --schedule-only")

    start = perf_counter()
    read = TimedParquetReads()
    try:
        if args.schedule_only:
            schema = SCHEDULE_SCHEMA
            path = ROOT / (args.development_parquet or DEVELOPMENT_PATH)
            schedule_X, development = schedule_sample(path, read=read)
            checkpoints = (None,)
            provenance = {
                "input_kind": "schedule_only", "development_path": str(path),
                "development_release": DEVELOPMENT_RELEASE if args.development_parquet is None else None,
                "sample_seed": SAMPLE_SEED, "sample_per_utc_month": SAMPLE_PER_MONTH,
            }
        else:
            inputs = read_manifest(ROOT / args.input_manifest)
            schema = inputs.schema
            checkpoints = tuple(cp for cp in inputs.checkpoints
                                if args.checkpoint_minutes in (None, cp.minutes))
            if not checkpoints:
                raise ValueError("--checkpoint-minutes must match a declared checkpoint")
            development = paired_metadata(inputs, read=read)
            provenance = {
                "input_kind": "paired_p3", "development_path": str(inputs.development),
                "development_release": inputs.release, "feature_spec_revision": inputs.revision,
                "manifest_sha256": inputs.fingerprint,
                "checkpoints": {cp.minutes: str(cp.path) for cp in inputs.checkpoints},
            }

        folds = fold_details(development)
        input_seconds = perf_counter() - start
        input_parquet_seconds = read.seconds
        factories = classical_factories(schema)
        if args.model:
            factories = {args.model: factories[args.model]}
        runs, summaries, timings, diagnostics = [], [], [], []
        for checkpoint in checkpoints:
            start = perf_counter()
            parquet_start = read.seconds
            X = schedule_X if checkpoint is None else checkpoint_features(
                checkpoint, schema, development.index, read=read,
            )
            feature_seconds = 0.0 if checkpoint is None else perf_counter() - start
            feature_parquet_seconds = read.seconds - parquet_start
            minutes = checkpoint.minutes if checkpoint is not None else None
            for name, factory in factories.items():
                rows, summary, elapsed, pca = evaluate_configuration(
                    name, factory, X, development, minutes, folds,
                )
                runs.append(rows)
                summaries.append(summary)
                diagnostics.extend(pca)
                timings.append({
                    "model": name, "checkpoint_minutes": minutes,
                    "feature_input_seconds": feature_seconds,
                    "feature_parquet_seconds": feature_parquet_seconds,
                    "evaluate_seconds": elapsed,
                })
            del X
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(runs, ignore_index=True).to_csv(output / "runs.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(diagnostics, columns=PCA_DIAGNOSTICS).to_csv(
        output / "pca_diagnostics.csv", index=False,
    )
    if args.schedule_only:
        development.index.to_frame(index=False).to_parquet(output / "sampled_ids.parquet", index=False)
    provenance.update({
        "cohort_rows": len(development), "predictors": dict(schema.predictors),
        "folds": folds, "seeds": SEEDS, "input_and_cohort_seconds": input_seconds,
        "input_parquet_seconds": input_parquet_seconds,
        "timings": timings, "code_revision": _code_revision(), "python": platform.python_version(),
        "p4_source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in ("src/models_classical.py", "src/run_classical.py")},
        "dependencies": {name: version(name) for name in (
            "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "timezonefinder", "tzdata",
        )},
        "model_settings_at_first_seed": {name: _model_settings(factory)
                                         for name, factory in factories.items()},
    })
    (output / "metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {sum(len(frame) for frame in runs)} model runs to {output}")


if __name__ == "__main__":
    main()
