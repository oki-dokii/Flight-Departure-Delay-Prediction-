import tempfile
import unittest
from pathlib import Path
import pyarrow as pa
import pandas as pd
import json
from unittest.mock import patch
from src import eda
from src.splits import TIME, OBSERVED
from src.eda import write_audit, write_issues


class GenerationTests(unittest.TestCase):
    def test_complete_generator_twice_preserves_curated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'reports/tables').mkdir(parents=True)
            dates = pd.date_range('2025-01-15', periods=10, freq='MS', tz='UTC') - pd.Timedelta(days=1)
            # Explicit January–October dates, with both labels in every slice.
            dates = pd.to_datetime([f'2025-{m:02d}-15' for m in range(1, 11)], utc=True).repeat(2)
            frame = pd.DataFrame({TIME: dates, 'LATE': [0,1]*10})
            frame[OBSERVED] = dates + pd.to_timedelta([0,15]*10, unit='min')
            frame['flight_id'] = [str(i) for i in range(20)]
            frame['FlightDate'] = frame[TIME].dt.strftime('%Y-%m-%d')
            for key, value in {'DepDelay':[0,15]*10, 'DepDel15':[0,1]*10,
                'CRSDepTime':1200, 'DayOfWeek':3, 'Reporting_Airline':'AA',
                'Origin':'JFK', 'Dest':'BOS', 'tail_number_missing':False,
                'Diverted':0, 'Flight_Number_Reporting_Airline':list(range(20))}.items():
                frame[key] = value
            frame.to_parquet(root/'data/processed/flights_development.parquet', index=False)
            counts = {f'2025-{m:02d}':{'outside_date_window':{'after':2}} for m in range(1,11)}
            (root/'data/processed/cleaning_counts.json').write_text(json.dumps(counts))
            (root/'reports/tables/arrival_exceptions.csv').write_text('flight_id\na\n')
            curated = ['COLUMN_AUDIT.md', 'reports/data_quality_issues.csv',
                       'reports/eda_observations.md', 'reports/limitations.md', 'reports/p2_phase_b.md']
            for name in curated: (root/name).write_text('curated sentinel\n')
            with patch.object(eda, 'ROOT', root):
                eda.run()
                audit = (root/'COLUMN_AUDIT.csv').read_bytes()
                eda.run()
            self.assertEqual(audit, (root/'COLUMN_AUDIT.csv').read_bytes())
            for name in curated: self.assertEqual((root/name).read_text(), 'curated sentinel\n')
            issues = pd.read_csv(root/'reports/data_quality_generated.csv').set_index('issue')
            self.assertEqual(issues.loc['arrival_clock_validation','status'], 'pending')
            self.assertEqual(issues.loc['repeated_schedule_keys','status'], 'clear')

    def test_repeatable_and_curated_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reports').mkdir()
            curated = ['COLUMN_AUDIT.md', 'reports/data_quality_issues.csv',
                       'reports/eda_observations.md', 'reports/limitations.md',
                       'reports/p2_phase_b.md']
            for name in curated:
                (root / name).write_text('reviewed content\n')
            schema = pa.schema([('DepDelay', pa.int32()), ('Tail_Number', pa.string())])
            write_audit(schema, root)
            original = (root / 'COLUMN_AUDIT.csv').read_bytes()
            write_audit(schema, root)
            write_issues([dict(issue='synthetic', status='pending')], root)
            self.assertEqual(original, (root / 'COLUMN_AUDIT.csv').read_bytes())
            for name in curated:
                self.assertEqual((root / name).read_text(), 'reviewed content\n')
