"""Tune classical models and export matched rolling-fold validation results."""

import argparse
import hashlib
import json
import platform
import resource
import warnings
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.alert_metrics import alert_metrics
from src.evaluate import SEEDS, fit_indices, positive_scores, revision
from src.features import AUDITED_PREDICTORS, EXPERIMENTAL_TAIL_PREDICTORS
from src.models_classical import FeatureSchema, VariancePCA, classical_from_recipe, phase_b_roster
from src.run_classical import checkpoint_features, paired_metadata, read_manifest
from src.splits import TIME, get_folds

CHECKPOINTS = (360, 60)
DEPTHS = (3, 6, 12, None)
LEAVES = (10, 50, 200)
ALPHAS = (0, 1e-6, 1e-5, 1e-4, 1e-3)
VARIANCES = (0.90, 0.95)
FIT_KEYS = ["checkpoint_minutes", "fold", "seed", "fit_mode"]


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def id_digest(index):
    # Canonical flight IDs are sorted by the paired loader, independent of Parquet order.
    hashes = pd.util.hash_pandas_object(index, index=False).to_numpy(dtype="<u8")
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_recipe(family, *, variance=None, scaled=False, **params):
    spec = {"family": family, "parameters": {f"classifier__{k}": v for k, v in params.items()}}
    if scaled:
        spec["scaled"] = True
    if variance is not None:
        spec["variance"] = variance
    return spec


def feature_schema(inputs, group, pilot, audit):
    schedule = tuple((item["name"], item["role"]) for item in AUDITED_PREDICTORS[:-2])
    congestion = tuple((item["name"], item["role"]) for item in AUDITED_PREDICTORS)
    declared = inputs.schema.predictors
    expected = schedule if group == "G0" else congestion
    if group == "G0" and pilot and declared == congestion:
        return FeatureSchema(schedule)
    if declared[:len(expected)] != expected:
        raise ValueError(f"{group} manifest does not match the audited {group} predictor roles")
    if group != "G2" and declared == expected:
        return inputs.schema
    if group == "G2" and ("weather_missing", "indicator") not in declared:
        raise ValueError("G2 must extend G1 and declare a binary weather_missing flag")
    if not audit:
        raise ValueError(f"{group} extensions require the reviewed --audit CSV")
    columns = pd.read_csv(audit).set_index("name", verify_integrity=True)
    experimental = {item["name"] for item in EXPERIMENTAL_TAIL_PREDICTORS}
    allowed = {"G1", "G2"} if group == "G2" else {group}
    for name, _ in declared[len(expected):]:
        if (name in experimental or "tail" in name.casefold() or name not in columns.index
                or columns.at[name, "approval_status"] != "approved" or columns.at[name, "group"] not in allowed
                or not columns.loc[name, ["allowed_6h", "allowed_1h"]].eq("yes").all()):
            raise ValueError(f"Predictor {name!r} is not approved at both checkpoints")
    return inputs.schema


def load_experiment(manifest, group, pilot, audit):
    inputs = read_manifest(manifest)
    if {cp.minutes for cp in inputs.checkpoints} != set(CHECKPOINTS):
        raise ValueError("Phase B requires 360- and 60-minute checkpoints")
    if pilot != (inputs.sampled_ids is not None):
        raise ValueError("Use --pilot only with an explicit sampled_ids manifest")
    schema = feature_schema(inputs, group, pilot, audit)
    meta = paired_metadata(inputs)
    if not pilot and len(meta) != pq.ParquetFile(inputs.development).metadata.num_rows:
        raise ValueError("Full-scale matrices must contain all development flight IDs")
    if not meta.LATE.isin([0, 1]).all():
        raise ValueError("LATE must contain only 0 and 1")
    return inputs, schema, meta


def build_fits(meta, modes):
    fits = {}
    for mode in modes:
        fits[mode] = []
        for fold, (train, valid) in enumerate(get_folds(meta), 1):
            for seed in SEEDS:
                selected = fit_indices(train, meta, seed, mode)
                y_train, y_valid = meta.LATE.iloc[selected], meta.LATE.iloc[valid]
                if y_train.nunique() != 2 or y_valid.nunique() != 2:
                    raise ValueError(f"Both classes are required in fold {fold}, seed {seed}")
                keys = dict(fold=fold, seed=seed, fit_mode=mode, train_rows=len(selected),
                            validation_rows=len(valid), validation_prevalence=float(y_valid.mean()),
                            train_ids_sha256=id_digest(y_train.index),
                            validation_ids_sha256=id_digest(y_valid.index))
                fits[mode].append((selected, valid, keys))
    return fits


class CheckpointRun:
    def __init__(self, X, meta, schema, minutes, output, identity, fits):
        self.X, self.y, self.schema = X, meta.LATE, schema
        self.days = meta[TIME].dt.floor("D")
        self.minutes, self.output, self.identity, self.fits = minutes, output, identity, fits
        self.candidates = []
        self.importance, self.spectrum, self.loadings, self.paths = [], [], [], []
        self.rules = []

    def fit(self, name, recipe, mode, artifacts=False):
        records = []
        for train, valid, keys in self.fits[mode]:
            model = classical_from_recipe(self.schema, keys["seed"], recipe)
            first_full_forest = artifacts and name == "random_forest" and mode == "full_fold" and not records
            if first_full_forest:
                peak_before = peak_rss_bytes()
            start = perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(self.X.iloc[train], self.y.iloc[train])
            fit_seconds = perf_counter() - start
            if first_full_forest:
                print(f'{name} {self.minutes}m fold={keys["fold"]} seed={keys["seed"]} '
                      f'fit_seconds={fit_seconds:.3f} process_peak_rss_bytes={peak_rss_bytes()} '
                      f'process_peak_before_fit_bytes={peak_before}', flush=True)
            X_valid, y_valid = self.X.iloc[valid], self.y.iloc[valid]
            start = perf_counter()
            scores = positive_scores(model, X_valid)
            score_seconds = perf_counter() - start
            if not (np.isfinite(scores).all() and np.logical_and(scores >= 0, scores <= 1).all()):
                raise ValueError(f"Invalid prediction probabilities for {name}, {keys}")
            row = dict(model=name, checkpoint_minutes=self.minutes,
                       feature_group=self.identity["feature_group"], revision=self.identity["revision"],
                       **keys, parameters=json.dumps(recipe, sort_keys=True),
                       fit_seconds=fit_seconds, score_seconds=score_seconds,
                       runtime_seconds=fit_seconds + score_seconds,
                       pr_auc=average_precision_score(y_valid, scores), roc_auc=roc_auc_score(y_valid, scores))
            classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
            if recipe["family"] == "decision_tree":
                row.update(fitted_depth=int(classifier.get_depth()), leaves=int(classifier.get_n_leaves()))
            if recipe["family"] == "random_forest":
                row.update(nodes=sum(t.tree_.node_count for t in classifier.estimators_))
            if "variance" in recipe:
                pca = model.named_steps["pca"]
                row.update(components=pca.n_components_,
                           retained_variance=float(pca.explained_variance_ratio_[:pca.n_components_].sum()))
                self.record_pca(model, row, name)
            if artifacts:
                row.update(**alert_metrics(y_valid, scores), brier=brier_score_loss(y_valid, scores))
                directory = self.output / "predictions" / str(self.minutes) / name
                directory.mkdir(parents=True, exist_ok=True)
                predictions = y_valid.to_frame().assign(
                    probability=scores, day=self.days.iloc[valid], model=name, feature_group=row["feature_group"],
                    run_id=self.identity["run_id"], **{k: row[k] for k in FIT_KEYS})
                prediction_path = directory / f'{keys["fold"]}-{keys["seed"]}.parquet'
                predictions.reset_index().to_parquet(prediction_path, index=False)
                if name == "decision_tree":
                    self.interpret(model, X_valid, y_valid, keys)
                print(f'{name} {self.minutes}m fold={keys["fold"]} seed={keys["seed"]} '
                      f'elapsed_seconds={row["runtime_seconds"]:.3f} output={prediction_path.resolve()}', flush=True)
            records.append(row)
            del model, classifier
        return records

    def record_pca(self, model, row, name):
        pca = model.named_steps["pca"]
        context = {k: row[k] for k in FIT_KEYS}
        if name in {"logistic_90", "pca_logistic"}:
            self.spectrum.extend(dict(**context, component=i, cumulative_variance=value,
                                      encoded_width=pca.training_shape_[1], encoded_density=pca.training_density_)
                                 for i, value in enumerate(pca.explained_variance_ratio_.cumsum(), 1))
            if name == "pca_logistic" and row["fold"] == 4 and row["seed"] == SEEDS[0]:
                names = model.named_steps["preprocess"].get_feature_names_out()
                self.loadings.extend(dict(**context, component=i, feature=feature, loading=value)
                                     for i, vector in enumerate(pca.components_[:3], 1)
                                     for feature, value in zip(names, vector, strict=True))

    def interpret(self, model, X, y, keys):
        rng = np.random.default_rng(4000 + keys["fold"])
        if len(y) > 4_000:
            selected = []
            for label in (0, 1):
                rows = np.flatnonzero(y.to_numpy() == label)
                size = max(1, int(4_000 * len(rows) / len(y)))
                selected.extend(rng.choice(rows, size=size, replace=False))
            sample = np.sort(selected)
            X, y = X.iloc[sample], y.iloc[sample]
        baseline = average_precision_score(y, positive_scores(model, X))
        result = permutation_importance(model, X, y, scoring="average_precision", n_repeats=5,
                                        random_state=4000 + keys["fold"])
        self.importance.extend(dict(checkpoint_minutes=self.minutes, fold=keys["fold"], seed=keys["seed"],
                                    feature=feature, repeat=i, pr_auc_drop=drop, baseline=baseline,
                                    sample_rows=len(y), sample_ids_sha256=id_digest(y.index))
                               for feature, values in zip(X.columns, result.importances, strict=True)
                               for i, drop in enumerate(values, 1))
        classifier = model.named_steps["classifier"]
        names = model.named_steps["preprocess"].get_feature_names_out()
        self.rules.extend(tree_rules(classifier.tree_, names, self.minutes, keys))

    def score_candidate(self, name, recipe):
        if any(row["model"] == name for row in self.candidates):
            return
        digest = hashlib.sha256(json.dumps([name, recipe], sort_keys=True).encode()).hexdigest()[:16]
        path = self.output / "candidates" / f"{self.minutes}-{digest}.json"
        # PCA diagnostics are regenerated with the fitted spectrum.
        if path.exists() and "variance" not in recipe:
            rows = json.loads(path.read_text())
        else:
            rows = self.fit(name, recipe, "sampled")
            write_json(path, rows)
        if len(rows) != len(self.fits["sampled"]):
            raise ValueError(f"Incomplete candidate {name}")
        self.candidates.extend(rows)

    def rank(self, names):
        subset = pd.DataFrame(self.candidates).loc[lambda frame: frame.model.isin(names)]
        means = subset.groupby("model")[["pr_auc", "roc_auc"]].mean()
        return sorted(names, key=lambda name: (-means.at[name, "pr_auc"], -means.at[name, "roc_auc"],
                                              names.index(name)))

    def pruning_path(self, recipe, train, seed):
        model = classical_from_recipe(self.schema, seed, recipe)
        y = self.y.iloc[train]
        encoded = model[:-1].fit_transform(self.X.iloc[train], y)
        return model.named_steps["classifier"].cost_complexity_pruning_path(encoded, y)


def tree_rules(tree, names, minutes, keys, node=0, depth=0, condition="All flights"):
    value = tree.value[node, 0]
    row = dict(checkpoint_minutes=minutes, fold=keys["fold"], seed=keys["seed"], node=node, depth=depth,
               condition=condition, train_support=int(tree.n_node_samples[node]),
               train_late_rate=float(value[1] / value.sum()))
    yield row
    if depth == 2 or tree.feature[node] < 0:
        return
    name = names[tree.feature[node]]
    role, name = name.split("__", 1)
    threshold = tree.threshold[node]
    if role == "categorical":
        feature, category = name.split("_", 1)
        left, right = f"{feature} is outside {category}", f"{feature} is in {category}"
    else:
        units = {"distance": " miles", "crs_elapsed_min": " minutes"}.get(name, "")
        left, right = f"{name} ≤ {threshold:.4g}{units}", f"{name} > {threshold:.4g}{units}"
    for child, predicate in [(tree.children_left[node], left), (tree.children_right[node], right)]:
        yield from tree_rules(tree, names, minutes, keys, child, depth + 1, f"{condition} AND {predicate}")


def tune_tree(run):
    grid = {f"tree_{d}_{l}": make_recipe("decision_tree", max_depth=d, min_samples_leaf=l, ccp_alpha=0)
            for d, l in product(DEPTHS, LEAVES)}
    for name, recipe in grid.items():
        run.score_candidate(name, recipe)
    fixed = "tree_phase_a"
    run.score_candidate(fixed, make_recipe("decision_tree"))
    finalists = run.rank(list(grid))[:2]
    for name in finalists:
        first_path = None
        for train, _, keys in run.fits["sampled"]:
            path = run.pruning_path(grid[name], train, keys["seed"])
            if first_path is None:
                first_path = path.ccp_alphas
            run.paths.extend(dict(checkpoint_minutes=run.minutes, candidate=name, fold=keys["fold"],
                                  seed=keys["seed"], alpha=a, impurity=i)
                             for a, i in zip(path.ccp_alphas, path.impurities, strict=True))
        nontrivial = first_path[(first_path > 0) & (first_path < first_path[-1])]
        extra = np.quantile(nontrivial, [0.25, 0.5, 0.75]) if len(nontrivial) else []
        for alpha in sorted(set((*ALPHAS[1:], *extra))):
            candidate = f"{name}_alpha_{float(alpha)!r}"
            params = {**grid[name]["parameters"], "classifier__ccp_alpha": float(alpha)}
            grid[candidate] = {"family": "decision_tree", "parameters": params}
            run.score_candidate(candidate, grid[candidate])
    winner = run.rank(list(grid))[0]
    return grid[winner]


def tune_forest(run, jobs, class_weights):
    grid = {}

    def add(depth, leaf, features="sqrt", weight=None, trees=64):
        name = f"forest_{depth}_{leaf}_{features}_{weight}_{trees}"
        grid[name] = make_recipe("random_forest", max_depth=depth, min_samples_leaf=leaf,
                                max_features=features, class_weight=weight, n_estimators=trees, n_jobs=jobs)
        run.score_candidate(name, grid[name])

    for depth, leaf in product((12, 24, None), (10, 50)):
        add(depth, leaf)
    for name in run.rank(list(grid))[:2]:
        p = grid[name]["parameters"]
        for features, weight in product(("sqrt", 0.3), (None, "balanced") if class_weights else (None,)):
            add(p["classifier__max_depth"], p["classifier__min_samples_leaf"], features, weight)
    p = grid[run.rank(list(grid))[0]]["parameters"]
    for trees in (128, 256):
        add(p["classifier__max_depth"], p["classifier__min_samples_leaf"], p["classifier__max_features"],
            p["classifier__class_weight"], trees)
    run.score_candidate("forest_phase_a", make_recipe("random_forest", n_jobs=jobs))
    return grid[run.rank(list(grid))[0]]


def fit_checkpoint(run, jobs, class_weights, mode, pca_forest_trees):
    tree = tune_tree(run)
    forest = tune_forest(run, jobs, class_weights)
    matched_forest = {**forest, "scaled": True, "parameters": forest["parameters"].copy()}
    if pca_forest_trees:
        trees = "classifier__n_estimators"
        matched_forest["parameters"][trees] = min(forest["parameters"][trees], pca_forest_trees)
    controls = {"logistic": make_recipe("logistic"), "random_forest": matched_forest}
    selected = {"majority": make_recipe("majority"), "decision_tree": tree, "random_forest": forest,
                "logistic": controls["logistic"], "pca_rf_control": controls["random_forest"]}
    for family, control in controls.items():
        run.score_candidate(f"{family}_control", control)
        variants = {f"{family}_{int(v * 100)}": {**control, "variance": v} for v in VARIANCES}
        for name, recipe in variants.items():
            run.score_candidate(name, recipe)
        _, with_pca = phase_b_roster(family)[-2:]
        selected[with_pca] = variants[run.rank(list(variants))[0]]
    rows = []
    for name, recipe in selected.items():
        rows.extend(run.fit(name, recipe, mode, artifacts=True))
    return rows, selected


def paired_deltas(rows, pairs):
    frame = pd.DataFrame(rows)
    keep = [*FIT_KEYS, "train_ids_sha256", "validation_ids_sha256"]
    columns = [*keep, "pr_auc", "roc_auc"]
    outputs = []
    for with_pca, without_pca in pairs:
        left = frame.loc[frame.model.eq(with_pca), columns]
        right = frame.loc[frame.model.eq(without_pca), columns]
        joined = left.merge(right, on=keep, validate="one_to_one", suffixes=("_pca", "_control"))
        if not len(joined) or len(joined) != len(left) or len(joined) != len(right):
            raise ValueError(f"PCA comparison has mismatched fit IDs: {with_pca}, {without_pca}")
        outputs.append(joined.assign(model=with_pca, control=without_pca,
                                     delta_pr_auc=joined.pr_auc_pca - joined.pr_auc_control,
                                     delta_roc_auc=joined.roc_auc_pca - joined.roc_auc_control))
    return pd.concat(outputs, ignore_index=True)


def run(manifest, group, output, mode, pilot=False, audit=None, forest_jobs=1, class_weights=False,
        parent_manifest=None, pca_forest_trees=None, one_hour_only=False):
    start = perf_counter()
    inputs, schema, meta = load_experiment(manifest, group, pilot, audit)
    parent = None
    if parent_manifest:
        if group == "G0":
            raise ValueError("G0 has no parent feature group")
        parent_group = "G0" if group == "G1" else "G1"
        parent, parent_schema, parent_meta = load_experiment(parent_manifest, parent_group, pilot, audit)
        if (parent.development != inputs.development or not meta.index.equals(parent_meta.index)
                or schema.predictors[:len(parent_schema.predictors)] != parent_schema.predictors):
            raise ValueError("Nested feature groups must preserve the development IDs and predictor roles")
        del parent_meta
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidates").mkdir(exist_ok=True)
    identity = dict(feature_group=group, feature_revision=inputs.revision, development_release=inputs.release,
                    development_sha256=file_digest(inputs.development), manifest_sha256=inputs.fingerprint,
                    feature_sha256={str(cp.minutes): file_digest(cp.path) for cp in inputs.checkpoints},
                    parent_manifest_sha256=parent.fingerprint if parent else None,
                    audit_sha256=file_digest(audit) if audit else None,
                    predictors=[list(item) for item in schema.predictors], revision=revision(),
                    source_sha256={name: file_digest(Path(__file__).with_name(name)) for name in
                                   ("phase_b_classical.py", "models_classical.py", "evaluate.py", "splits.py",
                                    "run_classical.py", "alert_metrics.py")},
                    versions={name: version(name) for name in ("scikit-learn", "numpy", "scipy", "pandas", "pyarrow")},
                    python=platform.python_version(), pilot=pilot, fit_mode=mode, forest_jobs=forest_jobs,
                    pca_forest_trees=pca_forest_trees,
                    class_weights=class_weights, cohort_rows=len(meta), cohort_ids_sha256=id_digest(meta.index))
    if one_hour_only:
        identity["checkpoint_minutes"] = [60]
    identity["run_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config = output / "experiment.json"
    if config.exists() and json.loads(config.read_text()) != identity:
        raise ValueError("Output already belongs to a different experiment")
    write_json(config, identity)
    fits = build_fits(meta, dict.fromkeys(("sampled", mode)))
    scores, candidates, diagnostics, recipes = [], [], {}, {}
    checkpoints = inputs.checkpoints
    if one_hour_only:
        # Check the six-hour parent predictors before fitting the one-hour study.
        checkpoints = sorted(checkpoints, key=lambda cp: cp.minutes, reverse=True)
    for cp in checkpoints:
        if one_hour_only and cp.minutes != 60 and parent is None:
            continue
        X = checkpoint_features(cp, inputs.schema, meta.index).loc[:, schema.columns]
        if parent:
            previous = next(item for item in parent.checkpoints if item.minutes == cp.minutes)
            expected = checkpoint_features(previous, parent.schema, meta.index).loc[:, parent_schema.columns]
            if not X.loc[:, parent_schema.columns].equals(expected):
                raise ValueError(f"{cp.minutes}m predictors changed from the parent feature group")
            del expected
        if one_hour_only and cp.minutes != 60:
            del X
            continue
        study = CheckpointRun(X, meta, schema, cp.minutes, output, identity, fits)
        rows, selected = fit_checkpoint(study, forest_jobs, class_weights, mode, pca_forest_trees)
        recipes[str(cp.minutes)] = selected
        scores.extend(rows)
        candidates.extend(study.candidates)
        for name in ("importance", "spectrum", "loadings", "paths", "rules"):
            diagnostics.setdefault(name, []).extend(getattr(study, name))
        print(f"{group} {cp.minutes}m: {len(study.candidates)} candidate and {len(rows)} selected scores", flush=True)
        del X, study
    selected = pd.DataFrame(scores)
    means = selected.loc[selected.model.isin(("logistic", "pca_rf_control"))].groupby("model").pr_auc.mean()
    pair = "random_forest" if means["pca_rf_control"] > means["logistic"] else "logistic"
    roster = phase_b_roster(pair)
    runs = selected.loc[selected.model.isin(roster)]
    runs.to_csv(output / "runs.csv", index=False)
    selected.to_csv(output / "selected_study.csv", index=False)
    pd.DataFrame(candidates).to_csv(output / "candidate_scores.csv", index=False)
    stats = runs.groupby(["model", "checkpoint_minutes"])[["pr_auc", "roc_auc", "runtime_seconds"]]
    stats.mean().add_suffix("_mean").join(stats.std(ddof=0).add_suffix("_std")).to_csv(output / "summary.csv")
    searched = [(f"{name}_{int(v * 100)}", f"{name}_control") for name, v in product(
        ("logistic", "random_forest"), VARIANCES)]
    paired_deltas(candidates, searched).to_csv(output / "pca_search_deltas.csv", index=False)
    paired_deltas(scores, [("pca_logistic", "logistic"), ("pca_rf", "pca_rf_control")]).to_csv(
        output / "pca_selected_deltas.csv", index=False)
    for name, rows in diagnostics.items():
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    write_json(output / "selected_recipes.json", {"seed": SEEDS[0], "rehearsal_fit_mode": mode,
                                                  "predictors": identity["predictors"],
                                                  "classical_pair": pair, "classical_recipes": {
        minutes: {name: values[name] for name in roster} for minutes, values in recipes.items()}})
    write_json(output / "runtime.json", dict(wall_seconds=perf_counter() - start,
                                             peak_rss_bytes=peak_rss_bytes()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--group", required=True, choices=("G0", "G1", "G2"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("sampled", "full_fold"), default="full_fold")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--parent-manifest", type=Path, help="Check unchanged predictor values from the preceding group")
    parser.add_argument("--forest-jobs", type=int, default=1)
    parser.add_argument("--pca-forest-trees", type=int, help="Apply a common tree-count cap to both matched RF arms")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--one-hour-only", action="store_true", help="Fit only the 60-minute checkpoint")
    args = parser.parse_args()
    if args.forest_jobs < 1 or (args.pca_forest_trees is not None and args.pca_forest_trees < 1):
        parser.error("Forest jobs and any matched-tree cap must be positive")
    run(**vars(args))
