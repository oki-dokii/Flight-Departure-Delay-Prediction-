"""Exact fold-level PSI and numeric KS from batched development value counts."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.audit import approve_manifest
from src.drift import CALENDAR
from src.run_classical import read_manifest
from src.splits import DEVELOPMENT_END, GAP, OBSERVED, START, TIME, VALIDATION_WINDOWS


def summarize(train, valid, categorical):
    """Use the same train-only quantiles, smoothing and rare levels as src.drift."""
    nt, nv = sum(train.values()), sum(valid.values())
    table = pd.DataFrame({"a": train, "b": valid}).fillna(0)
    # Missing counts are stored separately from the numeric/categorical domain.
    missing = table.loc["__MISSING__"] if "__MISSING__" in table.index else pd.Series({"a": 0, "b": 0})
    table = table.drop(index="__MISSING__", errors="ignore").sort_index()
    ks = np.nan
    if categorical:
        names = np.where(table.a / nt >= .01, table.index.astype(str), "__RARE_UNSEEN__")
        grouped = table.groupby(names).sum()
        if "__RARE_UNSEEN__" not in grouped.index:
            grouped.loc["__RARE_UNSEEN__"] = [0, 0]
    else:
        values = table.index.to_numpy(float)
        cumulative = table.a.cumsum().to_numpy()
        if cumulative.size and cumulative[-1]:
            pos = np.linspace(0, 1, 11) * (cumulative[-1] - 1)
            lo = values[np.searchsorted(cumulative, np.floor(pos), side="right")]
            hi = values[np.searchsorted(cumulative, np.ceil(pos), side="right")]
            quantiles = lo + (pos - np.floor(pos)) * (hi - lo)
            cuts = np.unique(quantiles)[1:-1]
        else:
            cuts = []
        grouped = table.groupby(np.searchsorted(cuts, values, side="left")).sum().reindex(
            range(len(cuts) + 1), fill_value=0)
        if table.a.sum() and table.b.sum():
            ks = (table.a.cumsum()/table.a.sum() - table.b.cumsum()/table.b.sum()).abs().max()
    grouped.loc["__MISSING__"] = missing
    p, q = (grouped.a / nt).clip(lower=1e-6), (grouped.b / nv).clip(lower=1e-6)
    p, q = p/p.sum(), q/q.sum()
    return ((q-p)*np.log(q/p)).sum(), ks


def run(manifest, group, output):
    inputs = read_manifest(manifest)
    if inputs.sampled_ids:
        raise ValueError("Expected full development matrices")
    audit = pd.read_csv("COLUMN_AUDIT.csv")
    registry = pd.read_csv("FEATURE_REGISTRY.csv", keep_default_na=False)
    rows = []
    for cp in inputs.checkpoints:
        approve_manifest(manifest, audit, registry, cp.minutes, group)
        counts = defaultdict(Counter)
        for x, y in zip(pq.ParquetFile(cp.path).iter_batches(65536),
                        pq.ParquetFile(inputs.development).iter_batches(65536, columns=["flight_id", TIME, OBSERVED]),
                        strict=True):
            f, m = x.to_pandas(), y.to_pandas()
            if not f.flight_id.astype("string").equals(m.flight_id.astype("string")):
                raise ValueError("Feature/development ID mismatch")
            sched, obs = m[TIME], m[OBSERVED]
            if not sched.between(START, DEVELOPMENT_END, inclusive="left").all() or obs.isna().any():
                raise ValueError("Development dates invalid")
            eligible = obs > sched - pd.Timedelta(hours=1)
            for fold, (start, end) in enumerate(VALIDATION_WINDOWS, 1):
                cutoff = start-GAP
                for kind, mask in [("train", eligible & (sched < cutoff) & (obs < cutoff)),
                                   ("valid", eligible & sched.between(start, end, inclusive="left"))]:
                    if not mask.any():
                        continue
                    subset = f.loc[mask]
                    for feature in inputs.schema.columns:
                        values = subset[feature]
                        tally = counts[fold, feature, kind]
                        tally.update(values.value_counts().to_dict())
                        tally["__MISSING__"] += int(values.isna().sum())
        for fold in range(1, len(VALIDATION_WINDOWS)+1):
            for feature, role in inputs.schema.predictors:
                a, b = counts[fold, feature, "train"], counts[fold, feature, "valid"]
                psi, ks = summarize(a, b, role=="categorical")
                rows.append(dict(group=group, checkpoint_minutes=cp.minutes, fold=fold, feature=feature,
                                 calendar_by_construction=feature in CALENDAR, train_rows=sum(a.values()),
                                 validation_rows=sum(b.values()), psi=psi, ks=ks, manifest_sha256=inputs.fingerprint))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--group", required=True)
    p.add_argument("--output", required=True, type=Path)
    run(**vars(p.parse_args()))
