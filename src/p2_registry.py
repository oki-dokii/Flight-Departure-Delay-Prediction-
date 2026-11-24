"""Generate the reviewed derived registry without altering source audit coverage."""
import json
from pathlib import Path
import pandas as pd
from src.features import EXPERIMENTAL_TAIL_PREDICTORS
from src.splits import TIME, OBSERVED

WEATHER_MISSING = 'weather_missing'
WEATHER_REVIEW = 'reports/audits/p3_065c2a1'
WEATHER_FIELDS = (
    ('wx_temp_c', 'tmpf', 'degC', '(tmpf - 32) * 5/9'),
    ('wx_wind_speed_kt', 'sknt', 'knots', 'identity'),
    ('wx_wind_gust_kt', 'gust', 'knots', 'missing gust becomes zero when wind speed exists; blocked pending review'),
    ('wx_visibility_mi', 'vsby', 'statute miles', 'identity'),
    ('wx_precip_1h_mm', 'p01i', 'mm', 'p01i.fillna(0) * 25.4; blocked: missing is not proven dry'),
)


def weather_registry():
    """Reviewed candidates, not availability approval or a source schema extension."""
    common = [TIME, 'Origin', 'station_map:origin_iata', 'station_map:station_id',
              'weather:station', 'weather:obs_time_utc', 'weather:available_at_utc',
              'weather:tmpf', 'weather:sknt', 'weather:vsby']
    rows = []
    for name, source, units, transformation in WEATHER_FIELDS + (
            (WEATHER_MISSING, '', 'binary 0/1',
             'unmapped, absent or observation age >6h; does not flag partial measurement nulls'),):
        sources = list(dict.fromkeys(common + ([f'weather:{source}'] if source else
                            [f'weather:{x[1]}' for x in WEATHER_FIELDS])))
        rows.append(dict(name=name, group='G2', source_columns=';'.join(sources),
            transformation=transformation, allowed_6h='conditional', allowed_1h='conditional',
            available_at='available_at_utc <= cutoff required; delivered join checks obs_time only',
            assumption_id='A-SCHED;A-WEATHER', approval_status='pending',
            leak_risk='report receipt/revisions unknown; missingness and station identity require review',
            observation_time='IEM valid -> UTC obs_time_utc; not a receipt timestamp',
            station_mapping=f'{WEATHER_REVIEW}/station_map_static.csv; IEM catalog comparison in weather_station_checks.csv',
            lag='not measured; six-hour age tolerance is not publication latency', units=units,
            licence='IEM public-domain terms verified 2026-11-25; https://mesonet.agron.iastate.edu/disclaimer.php',
            download_date='P3 receipt 2026-11-24T18:31:33.259597+00:00; underlying artifact not verified'))
    # Inventory the delivered name but never silently authorize both aliases.
    alias = dict(rows[-1], name='wx_weather_missing', approval_status='blocked_rename',
                 transformation='P3 delivered alias; rename to weather_missing and regenerate manifests/hashes',
                 allowed_6h='no', allowed_1h='no')
    return pd.DataFrame(rows + [alias])


def weather_source_audit():
    names = ['weather:station', 'weather:obs_time_utc', 'weather:available_at_utc',
             'station_map:origin_iata', 'station_map:station_id']
    names += [f'weather:{source}' for _, source, _, _ in WEATHER_FIELDS]
    return pd.DataFrame([dict(column=name, type='observed-by-T', allowed='conditional',
        allowed_6h='conditional', allowed_1h='conditional',
        justification='A-WEATHER pending: verify station, receipt time, lag and missingness')
        for name in names])


def build_registry():
    rows = []
    def add(name, group, sources, transform, assumption='', status='approved'):
        rows.append(dict(name=name, group=group, source_columns=';'.join(sources),
                         transformation=transform, allowed_6h='yes', allowed_1h='yes',
                         available_at='at or before scheduled departure minus checkpoint',
                         assumption_id=assumption, approval_status=status,
                         leak_risk='retrospective feed; enforce as-of and exclude target' if group != 'G0'
                         else 'schedule reconstruction requires A-SCHED',
                         observation_time='not applicable', station_mapping='not applicable',
                         lag='operational-feed assumption; not measured', units='see transformation',
                         licence='BTS source; see DATA_SOURCES.md', download_date='see data/download_manifest.json'))
    for name in ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month'):
        add(name, 'G0', [TIME], 'UTC calendar cyclic encoding; month integer', 'A-SCHED')
    for name, source, transform in (
            ('crs_elapsed_min', 'CRSElapsedTime', 'minutes; numeric coercion then fillna(0)'),
            ('distance', 'Distance', 'miles; numeric coercion then fillna(0)'),
            ('carrier', 'Reporting_Airline', 'category'), ('origin', 'Origin', 'category'),
            ('dest', 'Dest', 'category')):
        add(name, 'G0', [source], transform, 'A-SCHED')
    for name in ('origin_departures_3h', 'origin_late_rate_3h'):
        add(name, 'G1', [TIME, 'Origin', OBSERVED] + (['LATE'] if 'rate' in name else []),
            'other retained flights departed in inclusive [T-3h,T]; count or mean LATE',
            'A-SCHED;A-HISTORY')
    for feature in EXPERIMENTAL_TAIL_PREDICTORS:
        add(feature['name'], 'G3', [TIME, 'Tail_Number', 'tail_number_missing',
            'scheduled_arrival_utc', 'actual_arrival_utc', 'LATE'],
            'prior tail leg; arrived <= T; target excluded; needs continuity review',
            'A-SCHED;D1', 'pending')
    return pd.concat([pd.DataFrame(rows), weather_registry()], ignore_index=True)


if __name__ == '__main__':
    build_registry().to_csv('FEATURE_REGISTRY.csv', index=False)
    weather_source_audit().to_csv('WEATHER_SOURCE_AUDIT.csv', index=False)
