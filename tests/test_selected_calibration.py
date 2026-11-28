import unittest
import tempfile
import os
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from src.selected_calibration import verify_predictions, safe_path, run
from src.splits import TIME, OBSERVED, get_folds
from src.eda import audit_columns
from src.p2_registry import build_registry
from src import run_classical


class SelectedPredictionsTests(unittest.TestCase):
    def test_complete_selected_export_handoff_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            previous=Path.cwd()
            self.addCleanup(os.chdir,previous)
            os.chdir(root)
            dates=pd.to_datetime([f'2025-{m:02d}-{day}' for m in [1,6,7,8,9,10]
                                  for day in [15,16]],utc=True).repeat(2)
            meta=pd.DataFrame({TIME:dates,'LATE':[0,1]*12})
            meta[OBSERVED]=dates+pd.to_timedelta(meta.LATE*15,unit='min')
            meta['Distance']=100.
            meta.index=pd.Index([f'id-{i}' for i in range(len(meta))],name='flight_id')
            meta.to_parquet('development.parquet')
            for cp in [360,60]:
                pd.DataFrame({'distance':100.},index=meta.index).to_parquet(f'features_{cp}.parquet')
            doc=dict(development=dict(path='development.parquet',release='synthetic'),
                feature_spec_revision='synthetic',predictors=[dict(name='distance',role='numeric')],
                checkpoints=[dict(offset_minutes=cp,path=f'features_{cp}.parquet',source_release='synthetic') for cp in [360,60]])
            Path('manifest.json').write_text(json.dumps(doc))
            audit_columns(pq.read_schema('development.parquet')).to_csv('COLUMN_AUDIT.csv',index=False)
            build_registry().to_csv('FEATURE_REGISTRY.csv',index=False)
            Path('reports').mkdir();Path('reports/audit_assumptions.json').write_text('{}')
            identity=dict(feature_group='G0',manifest_sha256=hashlib.sha256(Path('manifest.json').read_bytes()).hexdigest(),
                development_sha256=hashlib.sha256(Path('development.parquet').read_bytes()).hexdigest(),
                fit_mode='full_fold',run_id='selected')
            Path('experiment.json').write_text(json.dumps(identity))
            rows=[]
            for cp in [360,60]:
                for fold,(_,valid) in enumerate(get_folds(meta),1):
                    part=meta.iloc[valid]
                    for seed in [42,73,101]:
                        key=dict(model='selected_tree',checkpoint_minutes=cp,fold=fold,seed=seed,fit_mode='full_fold')
                        scores=.2+.6*part.LATE
                        target=Path('predictions')/str(cp)/'selected_tree'/f'{fold}-{seed}.parquet'
                        target.parent.mkdir(parents=True,exist_ok=True)
                        part[['LATE']].assign(probability=scores,day=part[TIME].dt.floor('D'),
                            run_id='selected',feature_group='G0',**key).reset_index().to_parquet(target,index=False)
                        rows.append(dict(**key,pr_auc=average_precision_score(part.LATE,scores),
                                         roc_auc=roc_auc_score(part.LATE,scores),brier=.04))
            pd.DataFrame(rows).to_csv('runs.csv',index=False)
            Path('handoff.json').write_text(json.dumps(dict(feature_manifest='manifest.json',sources=[
                dict(owner='P4',experiment='experiment.json',runs='runs.csv',predictions='predictions')])))
            with patch.object(run_classical,'ROOT',root), patch('src.selected_calibration.plot'):
                run(Path('handoff.json'),output=Path('results'),bootstrap=20)
            summary=pd.read_csv('results/tables/calibration_summary.csv')
            self.assertEqual(len(summary),24)
            self.assertTrue((summary.brier-.04).abs().lt(1e-12).all())
            self.assertTrue(summary.normalized_ap.eq(1).all())

    def setUp(self):
        self.expected=pd.DataFrame({'LATE':[0,1],TIME:pd.to_datetime(['2025-07-01','2025-07-02'],utc=True)},index=['a','b'])
        self.key=dict(model='tree',checkpoint_minutes=60,fold=1,seed=42,fit_mode='full_fold')
        self.frame=pd.DataFrame(dict(flight_id=['b','a'],LATE=[1,0],probability=[.8,.2],
            day=pd.to_datetime(['2025-07-02','2025-07-01'],utc=True),run_id='selected',**self.key))

    def test_align_and_verify_actual_selected_export(self):
        p=verify_predictions(self.frame,self.expected,self.key,'selected')
        self.assertEqual(list(p),[.2,.8])

    def test_wrong_labels_ids_days_or_identity_rejected(self):
        for field,value in [('LATE',0),('flight_id','a'),('day','2025-11-02'),('run_id','other'),('probability',1.2),('seed',73)]:
            frame=self.frame.copy();frame.loc[0,field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                verify_predictions(frame,self.expected,self.key,'selected')

    def test_holdout_and_raw_paths_blocked(self):
        for path in ['data/raw/file.parquet','data/processed/holdout/f.parquet','data/processed/flights_holdout.parquet','../outside','/absolute']:
            with self.assertRaises(ValueError): safe_path(path)
