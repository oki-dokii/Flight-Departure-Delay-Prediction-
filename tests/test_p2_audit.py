import copy
import json
import unittest
from pathlib import Path
import pandas as pd
import pyarrow as pa
from src.audit import approve_manifest
from src.eda import audit_columns, validate_audit
from src.p2_registry import build_registry


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry()
        sources = sorted({c for value in self.registry.source_columns for c in value.split(';')})
        self.audit = audit_columns(pa.schema([(c, pa.string()) for c in sources]))
        self.assumptions = json.loads(Path('reports/audit_assumptions.json').read_text())
        self.manifest = {'predictors': [{'name': 'hour_sin'}]}

    def gate(self, **kw):
        return approve_manifest(kw.get('manifest', self.manifest), kw.get('audit', self.audit),
            kw.get('registry', self.registry), kw.get('checkpoint', 360), kw.get('group', 'G1'),
            kw.get('assumptions', self.assumptions))

    def test_approved_schedule_and_history(self):
        self.gate()
        self.gate(manifest={'predictors': [{'name': 'origin_late_rate_3h'}]})

    def test_outcome_unknown_and_conditional_blocked(self):
        for name in ('DepDelay', 'unknown', 'inbound_prior_late'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.gate(manifest={'predictors': [{'name': name}]})
        assumptions = copy.deepcopy(self.assumptions)
        assumptions['A-SCHED']['status'] = 'pending'
        with self.assertRaises(ValueError): self.gate(assumptions=assumptions)

    def test_tail_rejected_primary_even_if_approved(self):
        registry = self.registry.copy()
        registry.loc[registry.group.eq('G3'), 'approval_status'] = 'approved'
        for group in ('G0', 'G1', 'G2'):
            with self.assertRaises(ValueError):
                self.gate(registry=registry, group=group,
                          manifest={'predictors': [{'name': 'inbound_prior_late'}]})

    def test_checkpoint_flag(self):
        registry = self.registry.copy()
        registry.loc[registry.name.eq('hour_sin'), 'allowed_6h'] = 'no'
        with self.assertRaises(ValueError): self.gate(registry=registry)
        self.gate(registry=registry, checkpoint=60)
        audit = self.audit.copy()
        audit.loc[audit.column.eq('scheduled_departure_utc'), 'allowed_6h'] = 'no'
        with self.assertRaises(ValueError): self.gate(audit=audit)

    def test_duplicate_missing_and_unknown_source(self):
        for table in ('registry', 'audit'):
            frame = getattr(self, table)
            with self.assertRaises(ValueError): self.gate(**{table: pd.concat([frame, frame.iloc[:1]])})
            field, value = ('name', 'hour_sin') if table == 'registry' else ('column', 'scheduled_departure_utc')
            with self.assertRaises(ValueError): self.gate(**{table: frame[frame[field].ne(value)]})
        registry = self.registry.copy()
        registry.loc[registry.name.eq('hour_sin'), 'source_columns'] = 'DepDelay'
        with self.assertRaises(ValueError): self.gate(registry=registry)
        with self.assertRaises(ValueError): validate_audit(['extra'], self.audit)
