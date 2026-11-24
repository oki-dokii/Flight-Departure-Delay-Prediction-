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
from src.evaluate import SEEDS, revision, training_sample
from src.models_boosting import MODELS, make_booster
from src.models_classical import MISSING_CATEGORY, classical_factories
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
    times = meta[TIME].iloc[sampled]
    boundary = times.max().floor("D") - pd.Timedelta(days=13)
    inner = sampled[((times < boundary - GAP) & (meta[OBSERVED].iloc[sampled] < boundary - GAP)).to_numpy()]
    stop = sampled[(times >= boundary).to_numpy()]
    y = meta.LATE
    if min(y.iloc[a].nunique() for a in (inner, stop)) < 2:
        raise ValueError("Inner training and trailing validation each require both classes")
    best, chosen = -1, None
    for rounds in (10, 30, 60):
        candidate = make(rounds).fit(X.iloc[inner], y.iloc[inner])
        score = average_precision_score(y.iloc[stop], candidate.predict_proba(X.iloc[stop])[:, 1])
        if score <= best:
            break
        best, chosen = score, rounds
    model = make(chosen).fit(X.iloc[sampled], y.iloc[sampled])
    return model, chosen, sampled


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(**vars(parser.parse_args()))
