"""Development-only release evidence, without reading any raw files."""
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo, reset_tzpath
import pandas as pd
import pyarrow.parquet as pq
from src.eda import write_audit, validate_development
from src.splits import TIME, OBSERVED, get_folds


def run():
    path = Path('data/processed/flights_development.parquet')
    df = pd.read_parquet(path, columns=['flight_id', TIME, OBSERVED, 'LATE',
        'FlightDate', 'CRSDepTime', 'CRSElapsedTime', 'Distance'])
    validate_development(df)
    # Conservative union of US reporting-origin timezone ambiguity sets.
    # If the union is empty, the mapped-origin count is necessarily exactly zero.
    # No inferred origin-zone mapping and no outcome-based disambiguation is used.
    reset_tzpath([])
    local = pd.to_datetime(df.FlightDate) + pd.to_timedelta(
        df.CRSDepTime // 100 * 60 + df.CRSDepTime % 100, unit='min')
    unique = pd.Series(local.unique())
    zones = ['America/New_York', 'America/Chicago', 'America/Denver',
             'America/Los_Angeles', 'America/Anchorage', 'America/Adak',
             'America/Phoenix', 'Pacific/Honolulu', 'America/Puerto_Rico',
             'Pacific/Guam', 'Pacific/Pago_Pago', 'Pacific/Saipan',
             'America/St_Thomas', 'America/Indiana/Indianapolis',
             'America/Detroit', 'America/Boise', 'America/Juneau',
             'America/Nome', 'America/Sitka', 'America/Yakutat',
             'America/Metlakatla', 'America/Indiana/Knox',
             'America/Indiana/Tell_City', 'America/Kentucky/Louisville']
    ambiguous = pd.Series(False, index=unique.index)
    for zone in zones:
        a = unique.dt.tz_localize(ZoneInfo(zone), ambiguous=True, nonexistent='NaT')
        b = unique.dt.tz_localize(ZoneInfo(zone), ambiguous=False, nonexistent='NaT')
        ambiguous |= a.notna() & b.notna() & a.ne(b)
    count = int(local.isin(unique[ambiguous]).sum())
    if count:
        raise ValueError('Nonzero conservative ambiguity count; origin-zone mapping required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    info = dict(input=str(path), sha256=digest, bytes=path.stat().st_size,
        release='phase-a-data-v1', arrival_masking=False, rows=len(df),
        paired_eligible=int((df[OBSERVED] > df[TIME]-pd.Timedelta(hours=1)).sum()),
        late_rate=float(df.LATE.mean()), ambiguous_dst_rows=count,
        ambiguity_method='Union of US origin timezone ambiguity sets; empty union proves exact zero without origin mapping',
        zones=zones, elapsed_nulls=int(df.CRSElapsedTime.isna().sum()),
        distance_nulls=int(df.Distance.isna().sum()))
    Path('reports/tables/p2_evidence.json').write_text(json.dumps(info, indent=2)+'\n')
    folds = [dict(fold=i, train=len(t), validation=len(v),
        train_late=df.LATE.iloc[t].mean(), validation_late=df.LATE.iloc[v].mean())
        for i,(t,v) in enumerate(get_folds(df), 1)]
    pd.DataFrame(folds).to_csv('reports/tables/drift_population_folds.csv', index=False)
    write_audit(pq.read_schema(path), Path('.'))
    print(json.dumps(info, indent=2))


if __name__ == '__main__': run()
