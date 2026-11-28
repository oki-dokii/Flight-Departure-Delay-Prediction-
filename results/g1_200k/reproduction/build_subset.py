"""Reproduce the development-only 200k cohort from the original released files."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COUNT = 200_000
SEED = 42
DEST = Path("data/backup_200k")
TIME = "scheduled_departure_utc"
OBS = "actual_departure_utc"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main(source):
    checks = {}
    for name in ("SHA256SUMS", "feature_SHA256SUMS"):
        for line in (source / "results" / name).read_text().splitlines():
            sha, path = line.split(maxsplit=1)
            checks[Path(path).name] = sha
    development = source / "data/processed/flights_development_phase_b.parquet"
    if digest(development) != checks[development.name]:
        raise ValueError("Unexpected development release hash")

    # Positions are stable because the complete source file is hash-pinned.
    month_batches = []
    for batch in pq.ParquetFile(development).iter_batches(columns=[TIME, OBS], use_threads=False):
        frame = batch.to_pandas()
        scheduled = frame[TIME]
        if not scheduled.between("2025-01-01", "2025-10-31", inclusive="left").all():
            raise ValueError("Unexpected date outside development")
        months = scheduled.dt.month.to_numpy(dtype="uint8")
        months[~(frame[OBS] > scheduled - np.timedelta64(1, "h")).to_numpy()] = 0
        month_batches.append(months)
    months = np.concatenate(month_batches)
    counts = np.bincount(months, minlength=11)[1:]
    numerators = counts * COUNT
    quota = numerators // counts.sum()
    remainder_order = np.argsort(-(numerators % counts.sum()), kind="stable")
    quota[remainder_order[:COUNT - quota.sum()]] += 1
    rng = np.random.default_rng(SEED)
    selected = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(months == m), size=n, replace=False)
        for m, n in enumerate(quota, 1)
    ]))
    chunks = []
    offset = 0
    for batch in pq.ParquetFile(development).iter_batches(columns=["flight_id", TIME, OBS, "LATE"], use_threads=False):
        end = offset + len(batch)
        lo, hi = np.searchsorted(selected, [offset, end])
        if hi > lo:
            chunks.append(batch.take(selected[lo:hi] - offset))
        offset = end
    meta = pa.Table.from_batches(chunks)
    ids = meta["flight_id"].combine_chunks()
    assert len(ids) == pc.count_distinct(ids).as_py() == COUNT
    pq.write_table(meta, DEST / "development.parquet")
    pq.write_table(meta.select(["flight_id"]), DEST / "sampled_ids.parquet")

    sources = {development.name: checks[development.name]}
    for group in ("G0", "G1"):
        manifest = json.loads((source / "data/manifests" / f"manifest_{group}.json").read_text())
        manifest["development"]["path"] = str(DEST / "development.parquet")
        manifest["sampled_ids"] = str(DEST / "sampled_ids.parquet")
        for cp in manifest["checkpoints"]:
            path = source / cp["path"]
            if digest(path) != checks[path.name]:
                raise ValueError(f"Unexpected feature hash for {path.name}")
            sources[path.name] = checks[path.name]
            chunks = []
            for batch in pq.ParquetFile(path).iter_batches(use_threads=False):
                part = batch.filter(pc.is_in(batch["flight_id"], value_set=ids))
                if len(part):
                    chunks.append(part)
            table = pa.Table.from_batches(chunks)
            if len(table) != COUNT or pc.count_distinct(table["flight_id"]).as_py() != COUNT:
                raise ValueError(f"Incomplete paired cohort in {path.name}")
            output = DEST / path.name
            pq.write_table(table, output)
            cp["path"] = str(output)
        write_json(DEST / f"manifest_{group}.json", manifest)
    write_json(DEST / "sampling.json", dict(seed=SEED, rows=COUNT, eligible=int(counts.sum()),
               months=list(range(1,11)), eligible_by_month=counts.tolist(), sampled_by_month=quota.tolist(),
               source_sha256=sources))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Repository containing the verified development and feature releases")
    main(parser.parse_args().source)
