"""Publish a development copy with audited arrival exceptions set to null."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def build():
    root = Path("data/processed")
    source = pq.ParquetFile(root / "flights_development.parquet")
    exceptions = pd.read_csv("reports/tables/arrival_exceptions.csv")
    target = root / "flights_development_phase_b.parquet"
    partial = target.with_suffix(".part")
    with pq.ParquetWriter(partial, source.schema_arrow, compression="zstd", compression_level=6) as writer:
        for i in range(source.num_row_groups):
            table = source.read_row_group(i)
            for name, rows in exceptions.groupby("unusable_timestamp"):
                mask = pc.is_in(table["flight_id"], value_set=pa.array(rows.flight_id))
                values = pc.if_else(mask, pa.scalar(None, table[name].type), table[name])
                table = table.set_column(table.schema.get_field_index(name), name, values)
            writer.write_table(table)
    partial.replace(target)


if __name__ == "__main__":
    build()
