"""Small synthetic checks; no models or flight inputs."""
import csv
from pathlib import Path
import tempfile
import unittest

from src.report_artifact_check import check_claims, expected_value, artifact_path, check_blank_cells, check_selected_placeholders


class ReportArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        folder = self.root / 'reports/tables'
        folder.mkdir(parents=True)
        with (folder / 'metrics.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([['model','checkpoint','score','n'], ['A','60','.12341','10'],
                              ['A','60','.12349','20'], ['B','60','.9876','30']])
        self.spec = dict(csv='reports/tables/metrics.csv', column='score',
                         where=dict(model='A', checkpoint='60'), op='mean', format='.4f')
        self.claim = dict(id='model_a', label='Pilot', pattern=r'Model A (?P<v1>\d+\.\d+)', values=[self.spec])

    def test_rounding_and_aggregation(self):
        self.assertEqual(expected_value(self.spec, self.root), '0.1235')
        result = check_claims('Model A\n0.1235', [self.claim], self.root)
        self.assertEqual(result[0]['status'], 'pass')

    def test_detect_changed_number_and_missing_claim(self):
        self.assertEqual(check_claims('Model A 0.9876', [self.claim], self.root)[0]['status'], 'mismatch')
        self.assertEqual(check_claims('Model B 0.1235', [self.claim], self.root)[0]['status'], 'missing_claim')

    def test_duplicate_claim_is_ambiguous(self):
        result = check_claims('Model A 0.1235 Model A 0.1235', [self.claim], self.root)
        self.assertEqual(result[0]['status'], 'ambiguous_claim')

    def test_identity_selector_fails_closed(self):
        with self.assertRaises(ValueError):
            expected_value(dict(self.spec, op='single'), self.root)
        with self.assertRaises(ValueError):
            expected_value(dict(self.spec, where={'model':'unknown'}), self.root)

    def test_only_approved_csv_paths(self):
        for path in ['data/processed/flights_test.parquet', 'reports/tables/../../secret.csv',
                     '/tmp/other.csv', 'reports/tables/metrics.json']:
            with self.assertRaises(ValueError):
                artifact_path(self.root, path)

    def test_no_held_out_numeric_claims(self):
        with self.assertRaises(ValueError):
            check_claims('Model A 0.1235', [dict(self.claim, label='Held-out test')], self.root)

    def test_blank_test_cells_in_source_and_pdf(self):
        names = ['Majority','Decision Tree','Random Forest','Logistic (no PCA)', 'PCA + logistic',
                 'AdaBoost','Gradient Boosting','XGBoost','LightGBM','CatBoost']
        rows = [f'{n} & {h} & pending & pending & & \\\\' for n in names for h in ['6h','1h']]
        tex = 'Model & Checkpoint & Val ROC-AUC\n' + '\n'.join(rows) + '\n\\bottomrule'
        text = 'Test PR-AUC ' + ' '.join(f'{n} {h} pending pending' for n in names for h in ['6h','1h']) + ' Validation cells'
        self.assertTrue(check_blank_cells(tex, text))
        self.assertFalse(check_blank_cells(tex.replace('& &', '& 0.9 &', 1), text))
        self.assertFalse(check_blank_cells(tex, text.replace('Majority 6h pending pending', 'Majority 6h pending pending 0.9')))

    def test_selected_calibration_stays_blank(self):
        tex = 'Selected owner & Checkpoint & Brier\n' + '\n'.join(
            f'{owner} & {h} & & & & ' + '\\\\' for owner in ['P4','P5'] for h in ['6h','1h']) + '\n\\bottomrule'
        text = 'Selected owner Checkpoint Brier Skill ECE Bias P4 6h P4 1h P5 6h P5 1h Pending artifacts:'
        self.assertTrue(check_selected_placeholders(tex, text))
        self.assertFalse(check_selected_placeholders(tex.replace('& &', '& .1 &', 1), text))
        self.assertFalse(check_selected_placeholders(tex, text.replace('P4 6h', 'P4 6h 0.1')))
        self.assertFalse(check_selected_placeholders('', ''))
        self.assertFalse(check_blank_cells('', ''))

    def test_scoped_table_avoids_other_numbers(self):
        claim = dict(self.claim, start='Table start', end='Table end')
        result = check_claims('Model A 0.9999 Table start Model A 0.1235 Table end', [claim], self.root)
        self.assertEqual(result[0]['status'], 'pass')


if __name__ == '__main__':
    unittest.main()
