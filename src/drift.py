"""Descriptive development drift; training-only PSI definitions and paired AP."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.audit import approve_manifest
from src.run_classical import read_manifest, paired_metadata, checkpoint_features
from src.splits import TIME, OBSERVED, get_folds
from src.eda import validate_development

CALENDAR = {'month','hour_sin','hour_cos','dow_sin','dow_cos'}


def ks_effect(train, valid):
    a, b = np.sort(np.asarray(train, dtype=float)), np.sort(np.asarray(valid, dtype=float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b): return np.nan
    grid = np.unique(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a,grid,side='right')/len(a)
                               -np.searchsorted(b,grid,side='right')/len(b))))


def psi(train, valid, categorical=False):
    """All bin/category choices are fitted on train; rare threshold fixed at 1%."""
    train, valid = pd.Series(train), pd.Series(valid)
    if categorical:
        shares = train.dropna().astype(str).value_counts()/len(train)
        known = set(shares.index); frequent = set(shares[shares >= .01].index)
        def assign(s):
            return s.map(lambda x: '__MISSING__' if pd.isna(x) else
                         'value:'+str(x) if str(x) in frequent else '__RARE_UNSEEN__')
        labels = sorted(['__MISSING__', '__RARE_UNSEEN__']+['value:'+v for v in frequent])
        a,b = assign(train),assign(valid)
        definition = dict(categories=sorted(frequent), rare_share_threshold=.01,
                          other='explicit shared rare/unseen bucket', missing='separate')
    else:
        finite = pd.to_numeric(train, errors='raise').dropna().to_numpy()
        if np.isinf(finite).any(): raise ValueError('Infinite feature')
        cuts = np.unique(np.quantile(finite, np.linspace(0,1,11)))[1:-1] if len(finite) else np.array([])
        edges = np.r_[-np.inf, cuts, np.inf]
        def assign(s):
            result = pd.Series(np.searchsorted(cuts, s.fillna(0).astype(float), side='left'), index=s.index).astype(str)
            return result.where(s.notna(), '__MISSING__')
        labels = [str(i) for i in range(len(edges)-1)] + ['__MISSING__']
        a,b = assign(train),assign(valid)
        definition = dict(internal_edges=cuts.tolist(), outer_edges=['-inf','inf'], missing='separate')
    ac = a.value_counts().reindex(labels, fill_value=0)
    bc = b.value_counts().reindex(labels, fill_value=0)
    p,q = np.maximum(ac.to_numpy()/len(train),1e-6),np.maximum(bc.to_numpy()/len(valid),1e-6)
    p,q = p/p.sum(),q/q.sum()
    values = (q-p)*np.log(q/p)
    detail = pd.DataFrame(dict(bin=labels, train_n=ac.to_numpy(), validation_n=bc.to_numpy(),
                               train_share=p, validation_share=q, psi_component=values))
    return float(values.sum()), detail, definition


def kitagawa(train, valid):
    a = train.groupby('cell').LATE.agg(n='size', rate='mean')
    b = valid.groupby('cell').LATE.agg(n='size', rate='mean')
    cells = a.index.union(b.index)
    a,b = a.reindex(cells),b.reindex(cells)
    # For unsupported cells, carry the observed rate across populations. This
    # makes their contribution composition-only; disclose masses explicitly.
    ra,rb = a.rate.fillna(b.rate),b.rate.fillna(a.rate)
    wa,wb = a.n.fillna(0)/len(train),b.n.fillna(0)/len(valid)
    result = pd.DataFrame(dict(train_share=wa, validation_share=wb,
        train_rate=ra, validation_rate=rb, train_n=a.n.fillna(0), validation_n=b.n.fillna(0),
        composition=.5*(ra+rb)*(wb-wa), within=.5*(wa+wb)*(rb-ra)))
    result.index.name='cell'
    return result.reset_index()


def run(manifest, group='G1', performance=Path('reports/tables/calibration_summary.csv')):
    inputs=read_manifest(manifest)
    doc=json.loads(manifest.read_text())
    import pyarrow.parquet as pq
    from src.eda import validate_audit
    audit=pd.read_csv('COLUMN_AUDIT.csv')
    validate_audit(pq.read_schema(inputs.development).names,audit)
    registry=pd.read_csv('FEATURE_REGISTRY.csv',keep_default_na=False)
    for cp in inputs.checkpoints: approve_manifest(doc,audit,registry,cp.minutes,group)
    meta=paired_metadata(inputs); folds=get_folds(meta)
    rows,bins,definitions=[],[],[]
    for cp in inputs.checkpoints:
        X=checkpoint_features(cp,inputs.schema,meta.index)
        for fold,(train,valid) in enumerate(folds,1):
            for feature,role in inputs.schema.predictors:
                a,b=X[feature].iloc[train],X[feature].iloc[valid]
                score,detail,definition=psi(a,b,role=='categorical')
                rows.append(dict(checkpoint_minutes=cp.minutes,fold=fold,feature=feature,psi=score,
                    ks=ks_effect(a,b) if role!='categorical' else np.nan,
                    calendar_by_construction=feature in CALENDAR,train_rows=len(train),validation_rows=len(valid)))
                bins.append(detail.assign(checkpoint_minutes=cp.minutes,fold=fold,feature=feature))
                definitions.append(dict(checkpoint_minutes=cp.minutes,fold=fold,feature=feature,**definition))
    out=Path('reports/tables'); figs=Path('reports/figures'); figs.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'drift_features.csv',index=False)
    pd.concat(bins).to_csv(out/'drift_bins.csv',index=False)
    (out/'drift_bin_definitions.json').write_text(json.dumps(definitions,indent=2)+'\n')
    # Full paired eligible development population for prevalence/decomposition;
    # feature drift above uses the exact paired pilot cohort, not a substitute.
    full=pd.read_parquet(inputs.development,columns=['flight_id',TIME,OBSERVED,'LATE',
         'Reporting_Airline','CRSDepTime'])
    validate_development(full)
    full['cell']=full.Reporting_Airline.astype(str)+'|'+((full.CRSDepTime//100).astype(int)%24//6).astype(str)
    parts=[]; decomposition=[]
    for fold,(train,valid) in enumerate(get_folds(full),1):
        detail=kitagawa(full.iloc[train],full.iloc[valid]); parts.append(detail.assign(fold=fold))
        delta=full.LATE.iloc[valid].mean()-full.LATE.iloc[train].mean()
        composition=detail.composition.sum();within=detail.within.sum()
        if not np.isclose(delta, composition+within): raise ValueError('Decomposition failed')
        decomposition.append(dict(fold=fold,train_rate=full.LATE.iloc[train].mean(),
            validation_rate=full.LATE.iloc[valid].mean(),change=delta,composition=composition,within=within,
            entering_mass=detail.loc[detail.train_n.eq(0),'validation_share'].sum(),
            exiting_mass=detail.loc[detail.validation_n.eq(0),'train_share'].sum()))
    pd.concat(parts).to_csv(out/'drift_composition_cells.csv',index=False)
    decomp=pd.DataFrame(decomposition);decomp.to_csv(out/'drift_composition.csv',index=False)
    perf=pd.read_csv(performance)
    needed=['model','checkpoint_minutes','fold','seed','pr_auc','observed_rate']
    perf=perf[needed].copy()
    if perf.duplicated(needed[:4]).any(): raise ValueError('Duplicate performance keys')
    perf['normalized_ap']=(perf.pr_auc-perf.observed_rate)/(1-perf.observed_rate)
    perf['ap_lift']=perf.pr_auc/perf.observed_rate
    perf.to_csv(out/'drift_performance.csv',index=False)
    mean=perf.groupby(['model','checkpoint_minutes','fold'])[['pr_auc','observed_rate','normalized_ap','ap_lift']].mean().reset_index()
    mean.to_csv(out/'drift_performance_folds.csv',index=False)
    ref=perf[perf.model.eq('logistic')].drop(columns='model')
    paired=perf.merge(ref,on=['checkpoint_minutes','fold','seed'],suffixes=('','_logistic'),validate='many_to_one')
    for metric in ('pr_auc','normalized_ap','ap_lift'): paired[metric+'_difference']=paired[metric]-paired[metric+'_logistic']
    paired.to_csv(out/'drift_paired_differences.csv',index=False)
    spread=[]
    for (name,cp),g in perf.groupby(['model','checkpoint_minutes']):
        by=g.groupby('fold').pr_auc
        between=float(by.mean().var(ddof=0));within=float(by.var(ddof=0).mean());pooled=float(g.pr_auc.var(ddof=0))
        spread.append(dict(model=name,checkpoint_minutes=cp,pooled_std=np.sqrt(pooled),
            fold_mean_std=np.sqrt(between),within_fold_seed_rms_std=np.sqrt(within),
            fold_variance_share=between/pooled if pooled else np.nan))
    pd.DataFrame(spread).to_csv(out/'drift_seed_spread.csv',index=False)
    (out/'drift_provenance.json').write_text(json.dumps(dict(manifest=str(manifest),
        manifest_sha256=inputs.fingerprint,release=inputs.release,feature_cohort_rows=len(meta),
        composition_cohort='full paired eligible development',feature_cohort='paired eligible manifest population',
        performance=str(performance),cell='carrier x origin-local 6-hour scheduled band'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(decomp.fold,decomp.train_rate,'o-',label='Own training window')
    ax.plot(decomp.fold,decomp.validation_rate,'o-',label='Validation')
    ax.set(xticks=[1,2,3,4],xticklabels=['Jul','Aug','Sep','Oct'],ylabel='Late fraction',title='Full paired eligible development population')
    ax.legend();fig.tight_layout();fig.savefig(figs/'drift_prevalence.png',dpi=150);plt.close(fig)
    drift=pd.DataFrame(rows);rank=drift[~drift.calendar_by_construction].groupby(['feature','checkpoint_minutes']).psi.mean().unstack()
    rank.plot.barh(figsize=(8,5));plt.xlabel('Mean PSI across folds (training-only bins)');plt.tight_layout();plt.savefig(figs/'drift_psi.png',dpi=150);plt.close()
    for cp,g in mean.groupby('checkpoint_minutes'):
        fig,ax=plt.subplots(figsize=(9,5))
        for name,model in g.groupby('model'): ax.plot(model.fold,model.normalized_ap,'o-',label=name)
        ax.set(xlabel='Fold (Jul–Oct)',ylabel='(AP − prevalence) / (1 − prevalence)',title=f'Paired pilot validation: {cp} min',xticks=[1,2,3,4])
        ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(figs/f'drift_normalized_ap_{cp}.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--group',default='G1')
    p.add_argument('--performance',type=Path,default=Path('reports/tables/calibration_summary.csv'))
    run(**vars(p.parse_args()))
