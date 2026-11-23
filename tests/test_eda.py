import unittest
import pandas as pd
from src.eda import validate_development, validate_audit, approved_features
from src.splits import TIME, OBSERVED

class EdaTests(unittest.TestCase):
    def test_labels_and_boundaries(self):
        d=pd.DataFrame({'flight_id':['a','b'],TIME:pd.to_datetime(['2025-01-02']*2,utc=True),'LATE':[0,1]})
        d[OBSERVED]=d[TIME]+pd.to_timedelta([14,15],unit='min')
        validate_development(d)
        d.loc[0,TIME]=pd.Timestamp('2025-11-02',tz='UTC')
        with self.assertRaises(ValueError): validate_development(d)
    def test_audit_enforcement(self):
        a=pd.DataFrame({'column':['CRSDepTime','Tail_Number'],'allowed':['yes','conditional']})
        validate_audit(['CRSDepTime','Tail_Number'],a)
        for c in ['DepDelay','Tail_Number']:
            with self.assertRaises(ValueError): approved_features(pd.DataFrame({c:[1]}),a)
        with self.assertRaises(ValueError): validate_audit(['CRSDepTime'],a)
    def test_naive_missing_and_duplicate_metadata(self):
        d=pd.DataFrame({'flight_id':['a'],TIME:pd.to_datetime(['2025-01-02'],utc=True),'LATE':[0]})
        d[OBSERVED]=d[TIME]
        for invalid in [d.assign(scheduled_departure_utc=d[TIME].dt.tz_localize(None)),d.assign(actual_departure_utc=pd.NaT),pd.concat([d,d])]:
            with self.assertRaises(ValueError): validate_development(invalid)
        a=pd.DataFrame({'column':['x','x'],'allowed':['yes','yes']})
        with self.assertRaises(ValueError): validate_audit(['x'],a)
