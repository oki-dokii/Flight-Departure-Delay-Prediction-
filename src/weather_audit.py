"""Static P3 mapping review against an IEM station catalog; no observations read."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.p2_registry import WEATHER_FIELDS, WEATHER_MISSING


def weather_missingness(frame):
    """Whole-report flag and per-variable null coverage are different quantities."""
    columns = [x[0] for x in WEATHER_FIELDS]
    if WEATHER_MISSING not in frame:
        raise ValueError('Canonical weather_missing required; P3 must rename wx_weather_missing')
    flag = frame[WEATHER_MISSING]
    if not flag.isin([0,1]).all() or flag.isna().any():
        raise ValueError('weather_missing must be nonmissing binary')
    if frame.loc[flag.eq(1), columns].notna().any().any():
        raise ValueError('Missing/stale weather must leave numeric values null')
    if frame.loc[flag.eq(0), columns].isna().all(axis=1).any():
        raise ValueError('Available weather row has no measurements')
    return pd.DataFrame([dict(feature=c, rows=len(frame), missing_n=int(frame[c].isna().sum()),
        available_n=int(frame[c].notna().sum()), report_missing_n=int(flag.sum()),
        null_with_report_available_n=int((frame[c].isna() & flag.eq(0)).sum())) for c in columns])


def station_checks(mapping, catalog):
    required = ['origin_iata','station_id','latitude_deg','longitude_deg']
    if mapping[required].isna().any().any() or mapping.origin_iata.duplicated().any():
        raise ValueError('Missing or duplicate origin mapping')
    rows=[]
    for row in mapping.itertuples(index=False):
        matches=[f for f in catalog['features'] if str(f['id']) == str(row.station_id)]
        result=dict(origin=row.origin_iata, station_id=row.station_id, catalog_matches=len(matches),
                    distance_km=np.nan, status='pending_station_identity')
        if len(matches)==1:
            f=matches[0];lon,lat=f['geometry']['coordinates'][:2]
            a,b=np.radians([row.latitude_deg, lat]);dlat=b-a
            dlon=np.radians(lon-row.longitude_deg)
            h=np.sin(dlat/2)**2+np.cos(a)*np.cos(b)*np.sin(dlon/2)**2
            km=float(2*6371*np.arcsin(np.sqrt(np.clip(h,0,1))))
            result.update(distance_km=km, iem_name=f['properties'].get('sname'),
                          network=f['properties'].get('network'),
                          status='review_distance' if km>5 else 'catalog_location_consistent')
        rows.append(result)
    return pd.DataFrame(rows)


def run(mapping, catalog, output):
    document=json.loads(catalog.read_text())
    checks=station_checks(pd.read_csv(mapping),document)
    output.mkdir(parents=True,exist_ok=True)
    checks.to_csv(output/'weather_station_checks.csv',index=False)
    report=dict(mapping_rows=len(checks),mapping_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        catalog_source='https://mesonet.agron.iastate.edu/geojson/network/AZOS.geojson',
        retrieved_date='2026-11-25',distance_review_threshold_km=5,
        caveat='5 km is a review convention, not a coverage/identity guarantee; current catalog is not proof of 2025 sensor history',
        status_counts=checks.status.value_counts().to_dict(),
        development_weather_missingness='pending checkpoint matrices; no whole-year coverage used')
    (output/'weather_mapping_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    relevant=[f for f in document['features'] if f['id'] in set(checks.station_id)]
    (output/'iem_station_catalog_subset.json').write_text(json.dumps(dict(type='FeatureCollection',features=relevant),indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(checks[checks.status.ne('catalog_location_consistent')].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping',type=Path,default=Path('reports/audits/p3_065c2a1/station_map_static.csv'))
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reports/audits/p3_065c2a1'))
    run(**vars(p.parse_args()))
