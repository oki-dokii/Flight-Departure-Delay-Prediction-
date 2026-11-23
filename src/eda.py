"""Development-only descriptive analysis. Run: python -m src.eda."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.splits import START, DEVELOPMENT_END, TIME, OBSERVED, get_folds

ROOT = Path(__file__).resolve().parents[1]

def validate_development(df):
    for name in (TIME, OBSERVED):
        if not isinstance(df[name].dtype, pd.DatetimeTZDtype) or df[name].isna().any():
            raise ValueError(f'{name}: nonmissing aware timestamps required')
    if not df[TIME].between(START, DEVELOPMENT_END, inclusive='left').all():
        raise ValueError('Input contains rows outside development period')
    if df.flight_id.isna().any() or df.flight_id.duplicated().any():
        raise ValueError('flight_id must be unique and nonmissing')
    delay = (df[OBSERVED] - df[TIME]).dt.total_seconds() / 60
    if not df.LATE.isin([0, 1]).all() or not df.LATE.eq(delay.ge(15)).all():
        raise ValueError('Invalid LATE label')

def audit_columns(schema):
    schedule = {'Year','Quarter','Month','DayofMonth','DayOfWeek','FlightDate','Reporting_Airline','DOT_ID_Reporting_Airline','IATA_CODE_Reporting_Airline','Flight_Number_Reporting_Airline','CRSDepTime','CRSArrTime','CRSElapsedTime','DepTimeBlk','ArrTimeBlk','scheduled_departure_utc','scheduled_arrival_utc'}
    static = {'Origin','Dest','Distance','DistanceGroup'} | {p+s for p in ('Origin','Dest') for s in ('AirportID','AirportSeqID','CityMarketID','CityName','State','StateFips','StateName','Wac')}
    outcomes = {'DepTime','DepDelay','DepDelayMinutes','DepDel15','DepartureDelayGroups','TaxiOut','WheelsOff','WheelsOn','TaxiIn','ArrTime','ArrDelay','ArrDelayMinutes','ArrDel15','ArrivalDelayGroups','Cancelled','CancellationCode','Diverted','ActualElapsedTime','AirTime','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay','FirstDepTime','TotalAddGTime','LongestAddGTime',OBSERVED,'actual_arrival_utc','LATE'}
    rows=[]
    for field in schema:
        c=field.name
        kind, allowed, reason='outcome-like','no','Unrecognized field: unavailable pending manual review'
        if c in schedule: kind,allowed,reason='known-in-advance','yes','Published schedule/calendar; assumes schedule known at checkpoint'
        if c in static: kind,allowed,reason='static','yes','Route or carrier-independent airport geography known from schedule'
        if c in outcomes or c.startswith('Div'): kind,allowed,reason='outcome-like','no','Target-flight operational outcome; EDA/label only'
        if c in ('Tail_Number','tail_number_missing'): kind,allowed,reason='static','conditional','Final aircraft assignment is not proven available at checkpoint'
        if c in ('flight_id','Flights'): kind,allowed,reason='static','no','Technical identifier or count; exclude from predictors'
        if c in ('scheduled_departure_utc','scheduled_arrival_utc'): allowed,reason='conditional','Requires review of outcome-assisted DST reconstruction and arrival clock validation'
        rows.append(dict(column=c,dtype=str(field.type),type=kind,allowed=allowed,allowed_6h=allowed,allowed_1h=allowed,justification=reason))
    return pd.DataFrame(rows)

def validate_audit(columns,audit):
    if audit.column.duplicated().any() or set(columns)!=set(audit.column):
        raise ValueError('Audit must cover each schema column exactly once')

def approved_features(frame,audit):
    allowed=set(audit.loc[audit.allowed.eq('yes'),'column'])
    if set(frame.columns)-allowed:
        raise ValueError('Unapproved feature columns')
    return frame

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=ROOT/'data/processed/flights_development.parquet'
    pf=pq.ParquetFile(path)
    out=ROOT/'reports/tables'; out.mkdir(parents=True,exist_ok=True)
    figs=ROOT/'reports/figures'; figs.mkdir(exist_ok=True)
    audit=audit_columns(pf.schema_arrow); validate_audit(pf.schema_arrow.names,audit)
    audit.to_csv(ROOT/'COLUMN_AUDIT.csv',index=False)
    cols=['flight_id',TIME,OBSERVED,'LATE','DepDelay','DepDel15','CRSDepTime','FlightDate','DayOfWeek','Reporting_Airline','Origin','Dest','tail_number_missing','Diverted','Flight_Number_Reporting_Airline']
    df=pd.read_parquet(path,columns=cols); validate_development(df)
    df['month']=pd.to_datetime(df.FlightDate).dt.month
    df['hour']=(df.CRSDepTime//100).astype(int)%24
    df['route']=df.Origin+' → '+df.Dest
    missing=pd.Series(0,index=pf.schema_arrow.names,dtype='int64')
    monthly_nulls={}
    monthly_rows={}
    for batch in pf.iter_batches(batch_size=100000):
        missing += pd.Series({name:batch.column(i).null_count for i,name in enumerate(batch.schema.names)})
        part=batch.to_pandas()
        for month, group in part.groupby(part.FlightDate.str[:7]):
            monthly_nulls[month]=monthly_nulls.get(month,0)+group.isna().sum()
            monthly_rows[month]=monthly_rows.get(month,0)+len(group)
    pd.DataFrame({m:monthly_nulls[m]/monthly_rows[m] for m in sorted(monthly_nulls)}).to_csv(out/'missingness_by_month.csv')
    (missing/len(df)).rename('missing_fraction').to_csv(out/'missingness.csv')
    notes=[]
    def save(name,title):
        plt.title(title+'\nDevelopment only: Jan–Oct 2025'); plt.tight_layout(); plt.savefig(figs/f'{name}.png',dpi=140); plt.close()
    for key in ['LATE','hour','DayOfWeek','month','Reporting_Airline','Origin','route','tail_number_missing','Diverted']:
        table=df.groupby(key).LATE.agg(n='size',late_rate='mean')
        if key in ('Origin','route'): table=table.nlargest(20,'n')
        table.to_csv(out/f'{key}.csv')
        if key == 'Diverted':
            continue
        table['n' if key=='LATE' else 'late_rate'].plot.bar(figsize=(10,4))
        save(key,f'{key}: '+('flight count' if key=='LATE' else 'late fraction; group counts in accompanying table'))
        notes.append(f'{key}: {len(table)} displayed groups; {int(table.n.sum()):,} flights represented. Highest displayed late fraction {table.late_rate.max():.2%}; lowest {table.late_rate.min():.2%}. These comparisons are descriptive and unadjusted for route mix.')
    plt.figure(figsize=(9,4)); df.DepDelay.hist(bins=np.arange(-60,241,5)); plt.xlabel('Signed departure delay (minutes)')
    save('delay','Central delay distribution (outside range counted in quality table)')
    (missing/len(df)).sort_values().plot.barh(figsize=(10,22)); save('missingness','All-column null fractions')
    # Pandas' US federal holiday calendar: actual/observed federal dates,
    # symmetric +/- 2-day windows, defined before viewing rates.
    from pandas.tseries.holiday import USFederalHolidayCalendar
    dates=pd.to_datetime(df.FlightDate).dt.normalize()
    holidays=USFederalHolidayCalendar().holidays('2025-01-01','2025-10-30')
    near=pd.Series(False,index=df.index)
    for holiday in holidays: near |= (dates-holiday).abs().le(pd.Timedelta(days=2))
    df['holiday_window']=near
    holiday_table=df.groupby('holiday_window').LATE.agg(n='size',late_rate='mean')
    holiday_table.to_csv(out/'holiday.csv')
    holiday_table.late_rate.plot.bar(figsize=(8,4)); save('holiday','Federal holiday +/-2 days vs other days (unadjusted)')
    notes.append('Holiday comparison uses pandas USFederalHolidayCalendar observed federal dates, +/-2 days; the comparison is unadjusted and not a causal holiday effect.')
    pivot=df.pivot_table(index='DayOfWeek',columns='hour',values='LATE',aggfunc='mean')
    support=df.pivot_table(index='DayOfWeek',columns='hour',values='LATE',aggfunc='size')
    pivot=pivot.where(support>=100)
    plt.figure(figsize=(10,4)); plt.imshow(pivot,aspect='auto',vmin=0,vmax=1); plt.colorbar(label='Late fraction'); plt.xlabel('Local scheduled hour'); plt.ylabel('BTS weekday (row 0 = Monday)'); save('hour_weekday','Hour × weekday')
    quality={'rows':len(df),'late_fraction':float(df.LATE.mean()),'delay_label_disagreements':int(df.LATE.ne(df.DepDel15).sum()),'outside_histogram':int((~df.DepDelay.between(-60,240)).sum()),'delay_quantiles':df.DepDelay.quantile([0,.01,.5,.9,.99,1]).to_dict()}
    for hours in (6,1): quality[f'already_departed_{hours}h']=int((df[OBSERVED]<=df[TIME]-pd.Timedelta(hours=hours)).sum())
    keys=['FlightDate','Reporting_Airline','Flight_Number_Reporting_Airline','Origin','Dest','CRSDepTime']
    quality['repeated_schedule_key_rows']=int(df.duplicated(keys,keep=False).sum())
    quality['delay_utc_disagreements']=int(df.DepDelay.ne((df[OBSERVED]-df[TIME]).dt.total_seconds()/60).sum())
    pd.DataFrame([
        dict(issue='tail_assignment',count=int(df.tail_number_missing.sum()),owner='P3',status='pending',action='Confirm assignment availability; keep conditional features blocked'),
        dict(issue='arrival_clock_validation',count=None,owner='P1',status='pending',action='Validate reconstructed arrival against destination-local clock'),
        dict(issue='repeated_schedule_keys',count=quality['repeated_schedule_key_rows'],owner='P1',status='review',action='Investigate distinct records; do not automatically delete'),
        dict(issue='already_departed_1h',count=quality['already_departed_1h'],owner='P1/P3',status='review',action='Define population consistently at checkpoint')
    ]).to_csv(ROOT/'reports/data_quality_issues.csv',index=False)
    (out/'quality.json').write_text(json.dumps(quality,indent=2))
    folds=[dict(fold=i,train=len(t),validation=len(v),train_late=float(df.iloc[t].LATE.mean()),validation_late=float(df.iloc[v].LATE.mean())) for i,(t,v) in enumerate(get_folds(df),1)]
    pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
    counts=json.loads((ROOT/'data/processed/cleaning_counts.json').read_text())
    for month,steps in counts.items():
        actual=int(df.FlightDate.str.startswith(month).sum())
        if actual!=steps['outside_date_window']['after']: raise ValueError('Cleaning count mismatch')
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(sha256=digest,rows=len(df),min_utc=str(df[TIME].min()),max_utc=str(df[TIME].max()),cleaning_counts_reconciled=True),indent=2))
    (ROOT/'reports/eda_observations.md').write_text('# Development observations\n\n'+f'Late prevalence: {df.LATE.mean():.2%}. Median delay: {df.DepDelay.median():g} minutes; 99th percentile: {df.DepDelay.quantile(.99):g} minutes.\n\n'+'\n\n'.join(notes)+'\n\nThese are associations, not causal effects. Airport/route displays select by volume. No test data was loaded.\n')
    return quality

if __name__=='__main__': print(json.dumps(run(),indent=2))
