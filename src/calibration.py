"""Validation-only probabilities and calibration on a paired development manifest.

Run from root: python -m src.calibration --manifest data/sample/feature_manifest.json
No test-scoring option exists. Full-scale reruns use the same manifest argument.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.audit import approve_manifest
from src.evaluate import SEEDS, revision, training_sample
from src.models_classical import classical_factories
from src.run_classical import read_manifest, paired_metadata, checkpoint_features
from src.splits import TIME, get_folds


def reliability(y, p, days=None, bins=10, bootstrap=200, seed=42):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if (len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError('Nonempty binary labels and finite probabilities in [0,1] required')
    # Collapse tied quantiles; never split identical probabilities arbitrarily.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins+1)))
    inner = edges[1:-1]
    assigned = np.searchsorted(inner, p, side='left')
    rows = []
    for k in np.unique(assigned):
        mask = assigned == k
        rows.append(dict(bin=int(k), n=int(mask.sum()), predicted=float(p[mask].mean()),
                         observed=float(y[mask].mean()), lower=float(p[mask].min()),
                         upper=float(p[mask].max())))
    table = pd.DataFrame(rows)
    if days is not None and bootstrap:
        days = np.asarray(days)
        if len(days) != len(y) or pd.isna(days).any(): raise ValueError('Invalid day blocks')
        _, day = np.unique(days, return_inverse=True)
        n_days = int(day.max()+1)
        if n_days < 2: raise ValueError('At least two days required for bootstrap')
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(n_days, np.full(n_days, 1/n_days), size=bootstrap)
        for i, row in table.iterrows():
            mask = assigned == row['bin']
            counts = np.bincount(day[mask], minlength=n_days)
            obs = np.bincount(day[mask], weights=y[mask], minlength=n_days)
            pred = np.bincount(day[mask], weights=p[mask], minlength=n_days)
            denominator = weights @ counts
            supported = denominator > 0
            for name, values in [('observed', obs), ('predicted', pred)]:
                samples = (weights @ values)[supported] / denominator[supported]
                lo, hi = np.quantile(samples, [.025, .975])
                table.loc[i, name+'_lo'] = lo
                table.loc[i, name+'_hi'] = hi
            table.loc[i, 'bootstrap_supported'] = int(supported.sum())
        table['day_blocks'] = n_days
    return table


def calibration_metrics(y, p, base_rate, **kwargs):
    table = reliability(y, p, **kwargs)
    y, p = np.asarray(y), np.asarray(p)
    brier = float(np.mean((p-y)**2))
    reference = float(np.mean((base_rate-y)**2))
    return dict(brier=brier, base_rate_brier=reference,
        brier_skill=1-brier/reference if reference else np.nan,
        mean_prediction=float(p.mean()), observed_rate=float(y.mean()),
        calibration_bias=float(p.mean()-y.mean()),
        ece=float((table.n * (table.predicted-table.observed).abs()).sum()/len(y))), table


def run(manifest, config, group='G1', bootstrap=200, output=Path('reports'), models=None):
    # Validate manifest declarations and cohort bounds before opening predictors/fitting.
    inputs = read_manifest(manifest)
    import pyarrow.parquet as pq
    from src.eda import validate_audit
    audit = pd.read_csv('COLUMN_AUDIT.csv')
    validate_audit(pq.read_schema(inputs.development).names, audit)
    registry = pd.read_csv('FEATURE_REGISTRY.csv', keep_default_na=False)
    doc = json.loads(manifest.read_text())
    external = Path('WEATHER_SOURCE_AUDIT.csv')
    external_audits = (pd.read_csv(external),) if external.exists() else ()
    for cp in inputs.checkpoints:
        approve_manifest(doc, audit, registry, cp.minutes, group, source_audits=external_audits)
    metadata = paired_metadata(inputs)
    folds = get_folds(metadata)
    settings = json.loads(config.read_text())
    from src.models_boosting import make_booster
    factories = classical_factories(inputs.schema)
    tables, bins, predictions = [], [], []
    out = output / 'tables'; out.mkdir(parents=True, exist_ok=True)
    figs = output / 'figures'; figs.mkdir(parents=True, exist_ok=True)
    probabilities = Path('data/processed/p2_calibration'); probabilities.mkdir(parents=True, exist_ok=True)
    for cp in inputs.checkpoints:
        X = checkpoint_features(cp, inputs.schema, metadata.index)
        for fold, (train, valid) in enumerate(folds, 1):
            yv = metadata.LATE.iloc[valid]
            days = metadata[TIME].iloc[valid].dt.strftime('%Y-%m-%d').to_numpy()
            for seed in SEEDS:
                draw = training_sample(train, metadata, seed)
                yt = metadata.LATE.iloc[draw]
                base = float(yt.mean())
                for name, params in settings['models'][str(cp.minutes)].items():
                    if models and name not in models: continue
                    start = perf_counter()
                    model = factories[name](seed) if name in factories else make_booster(name, inputs.schema, seed)
                    model.set_params(**params)
                    model.fit(X.iloc[draw], yt)
                    p = model.predict_proba(X.iloc[valid])[:, list(model.classes_).index(1)]
                    key = dict(model=name, checkpoint_minutes=cp.minutes, fold=fold, seed=seed)
                    metrics, b = calibration_metrics(yv, p, base, days=days,
                                                    bootstrap=bootstrap, seed=seed)
                    ap = float(average_precision_score(yv, p)); prevalence = float(yv.mean())
                    tables.append(dict(**key, **metrics, train_rows=len(draw), validation_rows=len(valid),
                        training_prevalence=base, pr_auc=ap, roc_auc=roc_auc_score(yv,p),
                        normalized_ap=(ap-prevalence)/(1-prevalence), ap_lift=ap/prevalence,
                        runtime_seconds=perf_counter()-start,
                        parameters=json.dumps(model.get_params(deep=True), default=str, sort_keys=True)))
                    bins.append(b.assign(**key))
                    frame = pd.DataFrame(dict(flight_id=yv.index, LATE=yv.to_numpy(), probability=p,
                                              day=days, **key))
                    frame.to_parquet(probabilities/f'{name}_{cp.minutes}_{fold}_{seed}.parquet', index=False)
                    print(f'Calibration {name} {cp.minutes} fold {fold} seed {seed}', flush=True)
    summary = pd.DataFrame(tables)
    summary.to_csv(out/'calibration_summary.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(out/'calibration_bins.csv', index=False)
    record = dict(manifest=str(manifest), manifest_sha256=inputs.fingerprint,
        development=str(doc['development']['path']), release=inputs.release,
        arrival_masking=False if inputs.release == 'phase-a-data-v1' else 'verify release evidence', config=str(config),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), seeds=list(SEEDS),
        revision=revision(), bootstrap_replicates=bootstrap, bootstrap_seed='run seed',
        interval='95% percentile day-block bootstrap; seed 42 shown; conditional on fitted model',
        probability_directory=str(probabilities), group=group)
    (out/'calibration_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    plot(output)
    return summary


def plot(output, title='Development paired pilot'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = pd.read_csv(output/'tables/calibration_bins.csv')
    for cp, cohort in data[data.seed.eq(42)].groupby('checkpoint_minutes'):
        names = cohort.model.unique()
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), squeeze=False)
        for ax, name in zip(axes.flat, names):
            for fold, b in cohort[cohort.model.eq(name)].groupby('fold'):
                ax.plot(b.predicted, b.observed, 'o-', markersize=3, label=f'fold {fold}')
                if 'observed_lo' in b:
                    ax.fill_between(b.predicted, b.observed_lo, b.observed_hi, alpha=.12)
            ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax.set(title=name, xlabel='Mean predicted risk', ylabel='Observed late rate', xlim=(0,1), ylim=(0,1))
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(names):]: ax.axis('off')
        fig.suptitle(f'{title}: {cp} min; seed 42; 95% day-block intervals')
        fig.tight_layout()
        fig.savefig(output/f'figures/calibration_{cp}.png', dpi=140)
        plt.close(fig)
        # A readable report panel alongside the all-model diagnostic sheet.
        focal = cohort[cohort.model.eq('lightgbm')]
        if not focal.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for fold, b in focal.groupby('fold'):
                ax.plot(b.predicted, b.observed, 'o-', label=f'Fold {fold}')
                if 'observed_lo' in b:
                    ax.fill_between(b.predicted, b.observed_lo, b.observed_hi, alpha=.15)
            ax.plot([0,1], [0,1], '--', color='gray', linewidth=1)
            ax.set(xlabel='Mean predicted late probability', ylabel='Observed late fraction',
                   xlim=(0,1), ylim=(0,1), title=f'LightGBM, {title}: {cp} min, seed 42')
            ax.legend()
            fig.tight_layout()
            fig.savefig(output/f'figures/calibration_lightgbm_{cp}.png', dpi=160)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path('results/rehearsal_config.json'))
    parser.add_argument('--group', default='G1', choices=['G0','G1','G2','G3'])
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('reports'))
    parser.add_argument('--models', nargs='+')
    args = vars(parser.parse_args()); run(**args)
