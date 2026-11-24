"""Generate the reviewed derived registry without altering source audit coverage."""
import json
from pathlib import Path
import pandas as pd
from src.features import EXPERIMENTAL_TAIL_PREDICTORS
from src.splits import TIME, OBSERVED


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
    return pd.DataFrame(rows)


if __name__ == '__main__':
    build_registry().to_csv('FEATURE_REGISTRY.csv', index=False)
