"""Compare development arrival timestamps against separately reported destination clocks."""
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo, reset_tzpath

import pandas as pd
import pyarrow.parquet as pq
from timezonefinder import TimezoneFinder

from src.prepare import hhmm_minutes, load_airports


def audit(path):
    reset_tzpath([])
    airports = load_airports(Path("data/raw/airports.csv"))
    finder = TimezoneFinder()
    zones, totals, exceptions = {}, Counter(), []
    columns = ["flight_id", "Dest", "CRSArrTime", "ArrTime", "scheduled_arrival_utc", "actual_arrival_utc"]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=250_000):
        frame = batch.to_pandas()
        for dest, group in frame.groupby("Dest"):
            if dest not in zones:
                airport = airports.loc[dest]
                zones[dest] = ZoneInfo(finder.timezone_at(lng=airport.longitude_deg, lat=airport.latitude_deg))
            for stamp, clock in [(columns[4], columns[2]), (columns[5], columns[3])]:
                local = group[stamp].dt.tz_convert(zones[dest])
                expected = hhmm_minutes(group[clock]) % 1440
                valid = local.notna() & expected.notna()
                mismatch = valid & (local.dt.hour * 60 + local.dt.minute != expected)
                totals[stamp, "checked"] += int(valid.sum())
                totals[stamp, "missing"] += int((~valid).sum())
                totals[stamp, "mismatch"] += int(mismatch.sum())
                exceptions.extend((fid, stamp) for fid in group.loc[mismatch, "flight_id"])
    pd.DataFrame([(k, status, n) for (k, status), n in totals.items()],
                 columns=["timestamp", "status", "rows"]).to_csv("reports/tables/arrival_checks.csv", index=False)
    pd.DataFrame(exceptions, columns=["flight_id", "unusable_timestamp"]).to_csv(
        "reports/tables/arrival_exceptions.csv", index=False)


if __name__ == "__main__":
    audit("data/processed/flights_development.parquet")
