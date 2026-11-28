"""Paired development-only tuning and operations analysis for P5."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.alert_metrics import alert_metrics
from src.evaluate import SEEDS, positive_scores, revision, training_sample
from src.models_boosting import MODELS, make_booster
from src.models_classical import MISSING_CATEGORY, classical_factories
from src.phase_b_classical import build_fits, file_digest, id_digest, load_experiment, write_json
from src.run_classical import checkpoint_features, paired_metadata, read_manifest
from src.splits import GAP, OBSERVED, TIME, get_folds


class NativeCategories(BaseEstimator, TransformerMixin):
    def __init__(self, columns, catboost):
        self.columns = columns
        self.catboost = catboost

    def fit(self, X, y=None):
        self.levels_ = {c: X[c].dropna().unique().tolist() for c in self.columns}
        return self

    def transform(self, X):
        frame = X.copy()
        for c in self.columns:
            frame[c] = (frame[c].fillna(MISSING_CATEGORY) if self.catboost
                        else pd.Categorical(frame[c], categories=self.levels_[c]))
        return frame


GRIDS = {
    "adaboost": [{"estimator__max_depth": d, "learning_rate": r} for d, r in [(1, 1.0), (2, 0.1)]],
    "gradient_boosting": [{"max_depth": d, "learning_rate": r} for d, r in [(2, 0.1), (3, 0.03)]],
    "xgboost": [{"max_depth": d, "learning_rate": r} for d, r in [(3, 0.1), (6, 0.03)]],
    "lightgbm": [{"num_leaves": n, "min_child_samples": m} for n, m in [(15, 20), (31, 100)]],
    "catboost": [{"depth": d, "learning_rate": r} for d, r in [(4, 0.1), (6, 0.03)]],
}


def pipeline(name, schema, seed, params, native, rounds):
    base = make_booster(name, schema, seed)
    estimator = base.steps[-1][1]
    estimator.set_params(**params, **{"iterations" if name == "catboost" else "n_estimators": rounds})
    if not native:
        return base
    cats = schema.by_role("categorical")
    if name == "catboost":
        estimator.set_params(cat_features=cats, boosting_type="Ordered")
    return make_pipeline(FunctionTransformer(schema.normalize), NativeCategories(cats, name == "catboost"), estimator)


def tune_run(X, meta, train, valid, seed, make):
    sampled = training_sample(train, meta, seed)
    model, rounds = fit_with_stopping(X, meta, sampled, make)
    return model, rounds, sampled


def fit_with_stopping(X, meta, rows, make):
    times = meta[TIME].iloc[rows]
    boundary = times.max().floor("D") - pd.Timedelta(days=13)
    inner = rows[((times < boundary - GAP) & (meta[OBSERVED].iloc[rows] < boundary - GAP)).to_numpy()]
    stop = rows[(times >= boundary).to_numpy()]
    y = meta.LATE
    if min(y.iloc[a].nunique() for a in (inner, stop)) < 2:
        raise ValueError("Inner training and trailing validation each require both classes")
    best, chosen = -1, None
    for rounds in (10, 30, 60):
        candidate = make(rounds).fit(X.iloc[inner], y.iloc[inner])
        score = average_precision_score(y.iloc[stop], candidate.predict_proba(X.iloc[stop])[:, 1])
        del candidate
        if score <= best:
            break
        best, chosen = score, rounds
    model = make(chosen).fit(X.iloc[rows], y.iloc[rows])
    return model, chosen


def scores(y, p):
    return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                **alert_metrics(y, p))


def shap_sample(model, background, cases, seed):
    """Monte Carlo permutation SHAP using individual training flights as backgrounds."""
    rng = np.random.default_rng(seed)
    x = cases.reset_index(drop=True)
    contributions = np.zeros(x.shape)
    for _ in range(32):
        hybrid = background.iloc[rng.integers(len(background), size=len(x))].reset_index(drop=True).copy()
        previous = model.predict_proba(hybrid)[:, 1]
        for j in rng.permutation(x.shape[1]):
            hybrid[x.columns[j]] = x.iloc[:, j]
            new = model.predict_proba(hybrid)[:, 1]
            contributions[:, j] += new - previous
            previous = new
    return pd.Series(np.abs(contributions / 32).mean(axis=0), index=x.columns)


def run(manifest, output):
    inputs = read_manifest(manifest)
    meta = paired_metadata(inputs)
    folds = get_folds(meta)
    rev = revision()
    output.mkdir(parents=True, exist_ok=True)
    all_rows, comparisons, explanations = [], [], []
    for cp in inputs.checkpoints:
        X = checkpoint_features(cp, inputs.schema, meta.index)
        for name in MODELS:
            for native in ([False, True] if name in ("catboost", "lightgbm") else [False]):
                for candidate, params in enumerate(GRIDS[name]):
                    for fold, (train, valid) in enumerate(folds, 1):
                        for seed in SEEDS:
                            start = perf_counter()
                            def make(rounds):
                                return pipeline(name, inputs.schema, seed, params, native, rounds)
                            model, rounds, sampled = tune_run(X, meta, train, valid, seed, make)
                            p = model.predict_proba(X.iloc[valid])[:, 1]
                            all_rows.append(dict(model=name, native=native, candidate=candidate,
                                parameters=json.dumps(params), rounds=rounds, fold=fold, seed=seed,
                                checkpoint=cp.minutes, train_rows=len(sampled), validation_rows=len(valid),
                                runtime_seconds=perf_counter()-start, revision=rev, **scores(meta.LATE.iloc[valid], p)))
                    print(cp.minutes, name, native, candidate, flush=True)
        cp_rows = pd.DataFrame(all_rows).query("checkpoint == @cp.minutes")
        ranked = cp_rows.groupby(["model","native","candidate"]).pr_auc.mean().sort_values(ascending=False)
        selected = set()
        for name, native, candidate in ranked.index:
            if name in selected:
                continue
            selected.add(name)
            comparisons.extend(cp_rows.query("model == @name and native == @native and candidate == @candidate").to_dict("records"))
        name, native, candidate = ranked.index[0]
        train, valid = folds[-1]
        seed = SEEDS[0]
        make = lambda rounds: pipeline(name, inputs.schema, seed, GRIDS[name][candidate], native, rounds)
        model, _, sampled = tune_run(X, meta, train, valid, seed, make)
        impact = shap_sample(model, X.iloc[sampled], X.iloc[valid].sample(n=min(128,len(valid)), random_state=seed),seed)
        explanations.extend(dict(checkpoint=cp.minutes, model=name, feature=f, mean_abs_shap=v) for f,v in impact.items())
        for name, factory in classical_factories(inputs.schema).items():
            for fold, (train, valid) in enumerate(folds, 1):
                for seed in SEEDS:
                    sampled = training_sample(train, meta, seed)
                    model = factory(seed).fit(X.iloc[sampled], meta.LATE.iloc[sampled])
                    comparisons.append(dict(checkpoint=cp.minutes, model=name, fold=fold, seed=seed,
                        **scores(meta.LATE.iloc[valid], model.predict_proba(X.iloc[valid])[:,1])))
    pd.DataFrame(all_rows).to_csv(output/"candidates.csv",index=False)
    pd.DataFrame(comparisons).to_csv(output/"operations.csv",index=False)
    pd.DataFrame(explanations).to_csv(output/"shap.csv",index=False)


def run_matched(manifest, output, classical, audit):
    """Evaluate P5 on the same feature data and fit rows as a P4 export."""
    identity = json.loads((classical / "experiment.json").read_text())
    group, mode = identity["feature_group"], identity["fit_mode"]
    inputs, schema, meta = load_experiment(manifest, group, identity["pilot"], audit)
    actual = dict(revision=revision(), development_release=inputs.release, feature_revision=inputs.revision,
                  development_sha256=file_digest(inputs.development), manifest_sha256=inputs.fingerprint,
                  feature_sha256={str(cp.minutes): file_digest(cp.path) for cp in inputs.checkpoints},
                  audit_sha256=file_digest(audit) if audit else None, cohort_ids_sha256=id_digest(meta.index))
    if any(identity[key] != value for key, value in actual.items()):
        raise ValueError("P4 export was produced on different inputs or a different revision")
    output.mkdir(parents=True, exist_ok=True)
    fits = build_fits(meta, dict.fromkeys(("sampled", mode)))
    ops = pd.read_csv(classical / "runs.csv").rename(columns={"checkpoint_minutes": "checkpoint"}).to_dict("records")
    candidates, explanations, recipes = [], [], {}
    selected_runs = []
    run_id = "p5-" + identity["run_id"]
    for cp in inputs.checkpoints:
        X = checkpoint_features(cp, inputs.schema, meta.index).loc[:, schema.columns]
        selected = {}
        best = (-1.0, None)
        for model in MODELS:
            records = []
            for native in (False, True) if model in ("lightgbm", "catboost") else (False,):
                for index, params in enumerate(GRIDS[model]):
                    recipe = {"parameters": params, "native": native, "rounds": "training_selected"}
                    for train, valid, keys in fits["sampled"]:
                        row, estimator, probabilities = matched_fit(model, recipe, schema, X, meta, train, valid, keys, cp.minutes, group)
                        records.append(dict(row, candidate=index, native=native))
                        del estimator
            candidates.extend(records)
            ranking = pd.DataFrame(records).groupby(["native", "candidate"]).pr_auc.mean()
            native, index = ranking.idxmax()
            selected[model] = dict(parameters=GRIDS[model][index], native=bool(native), rounds="training_selected")
            if ranking.max() > best[0]:
                best = (ranking.max(), model)
        recipes[str(cp.minutes)] = selected
        for model, recipe in selected.items():
            for train, valid, keys in fits[mode]:
                row, estimator, probabilities = matched_fit(model, recipe, schema, X, meta, train, valid, keys, cp.minutes, group)
                ops.append(row)
                selected_runs.append({**row, "checkpoint_minutes": cp.minutes})
                directory = output / "predictions" / str(cp.minutes) / model
                directory.mkdir(parents=True, exist_ok=True)
                validation = meta.iloc[valid]
                validation.LATE.to_frame().assign(
                    probability=probabilities, day=validation[TIME].dt.floor("D"),
                    model=model, checkpoint_minutes=cp.minutes, fold=keys["fold"],
                    seed=keys["seed"], fit_mode=mode, feature_group=group, run_id=run_id,
                ).reset_index().to_parquet(directory / f'{keys["fold"]}-{keys["seed"]}.parquet', index=False)
                if model == best[1] and keys["fold"] == 4 and keys["seed"] == SEEDS[0]:
                    sample = X.iloc[valid].sample(n=min(128, len(valid)), random_state=SEEDS[0])
                    impact = shap_sample(estimator, X.iloc[train], sample, SEEDS[0])
                    explanations.extend(dict(checkpoint=cp.minutes, model=model, feature=feature, mean_abs_shap=value)
                                        for feature, value in impact.items())
                del estimator
            print(f"Matched P5 {group} {cp.minutes}m {model}", flush=True)
        del X
    pd.DataFrame(candidates).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(ops).to_csv(output / "operations.csv", index=False)
    pd.DataFrame(explanations).to_csv(output / "shap.csv", index=False)
    write_json(output / "experiment.json", actual | {"run_id": run_id} |
               {key: identity[key] for key in ("feature_group", "pilot", "fit_mode")})
    pd.DataFrame(selected_runs).to_csv(output / "selected_runs.csv", index=False)
    write_json(output / "p2_handoff.json", dict(feature_manifest=str(manifest), sources=[dict(
        owner="P5", experiment=str(output / "experiment.json"), runs=str(output / "selected_runs.csv"),
        predictions=str(output / "predictions"))]))
    write_json(output / "boosting_recipes.json", {"models": recipes})


def matched_fit(name, recipe, schema, X, meta, train, valid, keys, minutes, group):
    start = perf_counter()

    def make(rounds):
        return pipeline(name, schema, keys["seed"], recipe["parameters"], recipe["native"], rounds)

    estimator, rounds = fit_with_stopping(X, meta, train, make)
    p = positive_scores(estimator, X.iloc[valid])
    row = dict(model=name, checkpoint=minutes, feature_group=group, revision=revision(), **keys,
               parameters=json.dumps(recipe, sort_keys=True), rounds=rounds, runtime_seconds=perf_counter() - start,
               **scores(meta.LATE.iloc[valid], p))
    return row, estimator, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classical-run", type=Path, help="Use the selected P4 export and its exact cohort")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.classical_run:
        run_matched(args.manifest, args.output, args.classical_run, args.audit)
    else:
        run(args.manifest, args.output)
