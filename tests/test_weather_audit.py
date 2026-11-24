import copy
import unittest
import pandas as pd
from src.weather_audit import station_checks, weather_missingness
from src.p2_registry import weather_registry, weather_source_audit, WEATHER_FIELDS
from src.audit import approve_manifest


class WeatherAuditTests(unittest.TestCase):
    def test_station_match_and_duplicate_mapping(self):
        m=pd.DataFrame(dict(origin_iata=['X','Y'], station_id=['A','B'], latitude_deg=[0,0], longitude_deg=[0,0]))
        catalog={'features':[dict(id='A',geometry={'coordinates':[0,0]},properties={'sname':'A','network':'test'})]}
        result=station_checks(m,catalog)
        self.assertEqual(result.status.tolist(),['catalog_location_consistent','pending_station_identity'])
        with self.assertRaises(ValueError): station_checks(pd.concat([m,m]),catalog)

    def test_partial_null_is_not_whole_report_missing(self):
        x=pd.DataFrame({name:[1.,None] for name,_,_,_ in WEATHER_FIELDS})
        x.loc[0,'wx_temp_c']=None
        x['weather_missing']=[0,1]
        result=weather_missingness(x).set_index('feature')
        self.assertEqual(result.loc['wx_temp_c','null_with_report_available_n'],1)
        with self.assertRaises(ValueError): weather_missingness(x.rename(columns={'weather_missing':'wx_weather_missing'}))
        x.loc[1,'wx_temp_c']=5
        with self.assertRaises(ValueError): weather_missingness(x)

    def test_weather_pending_and_old_alias_fail_closed(self):
        registry=weather_registry()
        audit=pd.DataFrame([dict(column=c,type='known-in-advance',allowed_6h='yes',allowed_1h='yes')
                            for c in ['scheduled_departure_utc','Origin']])
        for name in registry.name:
            for cp in (360,60):
                with self.subTest(name=name,cp=cp),self.assertRaises(ValueError):
                    approve_manifest({'predictors':[{'name':name}]},audit,registry,cp,'G2',
                                     assumptions={},source_audits=(weather_source_audit(),))

    def test_separate_source_audit_and_named_approval(self):
        registry=weather_registry()
        registry.loc[registry.name.eq('wx_temp_c'),['allowed_6h','allowed_1h','approval_status']]=['yes','yes','approved']
        audit=pd.DataFrame([dict(column=c,type='known-in-advance',allowed_6h='yes',allowed_1h='yes')
                            for c in ['scheduled_departure_utc','Origin']])
        external=weather_source_audit()
        assumptions={'A-WEATHER':dict(status='approved',checkpoints=[360,60],
                     features=['wx_temp_c'],sources=external.column.tolist())}
        manifest={'predictors':[{'name':'wx_temp_c'}]}
        with self.assertRaises(ValueError): approve_manifest(manifest,audit,registry,360,'G2',assumptions)
        self.assertEqual(approve_manifest(manifest,audit,registry,360,'G2',assumptions,(external,)),['wx_temp_c'])
        with self.assertRaises(ValueError):
            approve_manifest(manifest,audit,registry,360,'G2',assumptions,(external,external))
