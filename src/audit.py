"""Checkpoint-aware derived-feature approval; raw-frame policy stays in src.eda."""
import json
from pathlib import Path

import pandas as pd

GROUPS = ('G0', 'G1', 'G2', 'G3')


def approve_manifest(manifest, audit, registry, checkpoint, group,
                     assumptions=None):
    if checkpoint not in (360, 60) or group not in GROUPS:
        raise ValueError('Unknown checkpoint or group')
    if assumptions is None:
        assumptions = json.loads(Path('reports/audit_assumptions.json').read_text())
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    for table, key in ((audit, 'column'), (registry, 'name')):
        if table[key].isna().any() or table[key].duplicated().any():
            raise ValueError('Duplicate or missing audit/registry rows')
    names = [p['name'] for p in manifest['predictors']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate predictor')
    flag = 'allowed_6h' if checkpoint == 360 else 'allowed_1h'
    a = audit.set_index('column')
    r = registry.set_index('name')
    for name in names:
        if name not in r.index:
            raise ValueError(f'Unknown predictor: {name}')
        row = r.loc[name]
        if row['group'] not in GROUPS or GROUPS.index(row['group']) > GROUPS.index(group):
            raise ValueError(f'{name}: group mismatch')
        if row['group'] == 'G3' and group != 'G3':
            raise ValueError(f'{name}: tail ablation only')
        if row[flag] != 'yes' or row.approval_status != 'approved':
            raise ValueError(f'{name}: unapproved at checkpoint')
        sources = row.source_columns.split(';')
        if not sources or any(c not in a.index for c in sources):
            raise ValueError(f'{name}: missing/unknown source audit row')
        ids = str(row.assumption_id).split(';')
        approved = []
        for ident in ids:
            item = assumptions.get(ident, {})
            if (item.get('status') == 'approved' and checkpoint in item.get('checkpoints', [])
                    and name in item.get('features', [])):
                approved.append(item)
        for source in sources:
            # An explicit no at a checkpoint is never overridden. Conditional
            # and prior-flight outcomes need a scoped written assumption.
            permission = a.loc[source, flag]
            if permission == 'yes':
                continue
            if permission == 'no' and a.loc[source, 'type'] != 'outcome-like':
                raise ValueError(f'{name}: source unavailable at checkpoint: {source}')
            if not any(source in item.get('sources', []) for item in approved):
                raise ValueError(f'{name}: blocked source: {source}')
    return names
