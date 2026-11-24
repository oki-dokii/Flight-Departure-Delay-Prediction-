"""Calibrate reporting from teammates' selected validation exports; never refit.

A handoff JSON declares feature_manifest and sources (owner, experiment, runs,
predictions). Each source's runs CSV must contain only the selected rows to use.
Input prediction paths follow P4: <predictions>/<minutes>/<model>/<fold>-<seed>.parquet.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.audit import approve_manifest
from src.calibration import calibration_metrics, plot
from src.evaluate import SEEDS, training_sample
from src.run_classical import read_manifest, paired_metadata
from src.splits import TIME, get_folds


def safe_path(value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or any(
            part.lower() in ('raw','holdout','test','final_once') or 'holdout' in part.lower()
            for part in path.parts):
        raise ValueError('Only repository-relative development/validation handoff paths are allowed')
    return path


def verify_predictions(frame, expected, key, run_id, group=None):
    columns=['flight_id','LATE','probability','day','model','checkpoint_minutes','fold','seed','run_id','fit_mode']
    if not set(columns).issubset(frame): raise ValueError('Prediction export lacks identity fields')
    if frame.flight_id.isna().any() or frame.flight_id.duplicated().any():
        raise ValueError('Duplicate or missing validation IDs')
    if set(frame.flight_id)!=set(expected.index): raise ValueError('Validation ID cohort mismatch')
    frame=frame.set_index('flight_id').reindex(expected.index)
    for name,value in key.items():
        if not frame[name].eq(value).all(): raise ValueError(f'Prediction identity mismatch: {name}')
    if not frame.run_id.eq(run_id).all(): raise ValueError('Prediction run_id mismatch')
    if group is not None and ('feature_group' not in frame or not frame.feature_group.eq(group).all()):
        raise ValueError('Prediction feature group mismatch')
    if not np.array_equal(frame.LATE.to_numpy(),expected.LATE.to_numpy()):
        raise ValueError('Prediction labels differ from frozen development metadata')
    days=pd.to_datetime(frame.day,utc=True,errors='raise')
    if not days.eq(expected[TIME].dt.floor('D')).all():
        raise ValueError('Prediction day differs from scheduled validation day')
    p=frame.probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Invalid probability')
    return p


def run(handoff, output=Path('reports/selected_calibration'), bootstrap=200):
    specification=json.loads(handoff.read_text())
    manifest=safe_path(specification['feature_manifest']);inputs=read_manifest(manifest)
    # Read approved registry before consuming any prediction payload.
    audit=pd.read_csv('COLUMN_AUDIT.csv')
    import pyarrow.parquet as pq
    from src.eda import validate_audit
    validate_audit(pq.read_schema(inputs.development).names,audit)
    registry=pd.read_csv('FEATURE_REGISTRY.csv',keep_default_na=False)
    external=Path('WEATHER_SOURCE_AUDIT.csv')
    external_audits=(pd.read_csv(external),) if external.exists() else ()
    groups={json.loads(safe_path(source['experiment']).read_text())['feature_group']
            for source in specification['sources']}
    if len(groups)!=1: raise ValueError('Use one feature group per calibration handoff')
    group=groups.pop()
    for cp in inputs.checkpoints:
        approve_manifest(manifest,audit,registry,cp.minutes,group,source_audits=external_audits)
    metadata=paired_metadata(inputs);folds=get_folds(metadata)
    metrics=[];bins=[];source_records=[]
    keys=['model','checkpoint_minutes','fold','seed','fit_mode']
    seen=set()
    for source in specification['sources']:
        experiment=safe_path(source['experiment']);runs_path=safe_path(source['runs'])
        identity=json.loads(experiment.read_text())
        if identity['manifest_sha256']!=inputs.fingerprint:
            raise ValueError('Selected run manifest hash differs from declared inputs')
        with inputs.development.open('rb') as stream:
            development_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if identity['development_sha256']!=development_sha:
            raise ValueError('Selected run development hash mismatch')
        selected=pd.read_csv(runs_path)
        required = keys + ['pr_auc','roc_auc']
        if not set(required).issubset(selected) or selected.empty:
            raise ValueError('Selected runs need identity and score columns')
        if not selected.fit_mode.eq(identity['fit_mode']).all():
            raise ValueError('Mixed or mismatched selected fit modes')
        if selected.duplicated(keys).any(): raise ValueError('Duplicate selected score keys')
        for name,part in selected.groupby('model'):
            if set(part.checkpoint_minutes)!={cp.minutes for cp in inputs.checkpoints}:
                raise ValueError('Selected model needs both checkpoint exports')
        for (name,cp),part in selected.groupby(keys[:2]):
            if set(part[['fold','seed']].itertuples(index=False,name=None))!={(f,s) for f in range(1,5) for s in SEEDS}:
                raise ValueError('Selected model needs every frozen validation fold and seed')
        for row in selected.to_dict('records'):
            key={k:row[k] for k in keys};fold=int(row['fold']);seed=int(row['seed']);cp=int(row['checkpoint_minutes'])
            if cp not in (360,60) or fold not in range(1,5) or seed not in SEEDS:
                raise ValueError('Unknown checkpoint/fold/seed')
            signature=tuple(key[k] for k in keys)
            if signature in seen: raise ValueError('Duplicate model across selected sources')
            seen.add(signature)
            train,valid=folds[fold-1]
            if row['fit_mode']=='sampled': train=training_sample(train,metadata,seed)
            elif row['fit_mode']!='full_fold': raise ValueError('Unknown selected fit mode')
            path=safe_path(str(safe_path(source['predictions'])/str(cp)/row['model']/f'{fold}-{seed}.parquet'))
            frame=pd.read_parquet(path)
            expected=metadata.iloc[valid]
            p=verify_predictions(frame,expected,key,identity['run_id'],group)
            ap=float(average_precision_score(expected.LATE,p));roc=float(roc_auc_score(expected.LATE,p))
            if not np.isclose(ap,row['pr_auc'],atol=1e-12,rtol=0) or not np.isclose(roc,row['roc_auc'],atol=1e-12,rtol=0):
                raise ValueError('Selected scores do not reproduce exported probabilities')
            m,b=calibration_metrics(expected.LATE,p,float(metadata.LATE.iloc[train].mean()),
                days=expected[TIME].dt.strftime('%Y-%m-%d'),bootstrap=bootstrap,seed=seed)
            if 'brier' in row and not np.isclose(m['brier'],row['brier'],atol=1e-12,rtol=0):
                raise ValueError('Brier differs from teammate export')
            prevalence=float(expected.LATE.mean())
            metrics.append(dict(**key,**m,owner=source['owner'],run_id=identity['run_id'],
                feature_group=group,pr_auc=ap,roc_auc=roc,normalized_ap=(ap-prevalence)/(1-prevalence),
                ap_lift=ap/prevalence,validation_rows=len(valid),train_rows=len(train)))
            bins.append(b.assign(**key,owner=source['owner']))
            source_records.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    tables=output/'tables';figures=output/'figures';tables.mkdir(parents=True,exist_ok=True);figures.mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(tables/'calibration_summary.csv',index=False)
    pd.concat(bins,ignore_index=True).to_csv(tables/'calibration_bins.csv',index=False)
    (tables/'calibration_provenance.json').write_text(json.dumps(dict(handoff=str(handoff),
        handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
        mode='selected_validation_exports_no_refit',feature_manifest=str(manifest),
        manifest_sha256=inputs.fingerprint,bootstrap_replicates=bootstrap,files=source_records),indent=2)+'\n')
    scope='pilot' if inputs.sampled_ids is not None else 'full development'
    plot(output,title=f'Selected validation exports ({scope}, {group})')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reports/selected_calibration'))
    p.add_argument('--bootstrap',type=int,default=200)
    run(**vars(p.parse_args()))
