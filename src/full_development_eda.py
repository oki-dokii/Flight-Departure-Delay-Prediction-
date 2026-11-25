"""Describe released development features in bounded memory."""
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.splits import DEVELOPMENT_END, START

ROOT = Path("data/processed")
WEATHER = ["wx_temp_c", "wx_wind_speed_kt", "wx_wind_gust_kt", "wx_visibility_mi", "wx_precip_1h_mm"]
FLAGS = ["inbound_missing_no_tail", "inbound_first_leg_of_day", "inbound_not_landed_by_T"]
COLUMNS = ["flight_id", "origin", "weather_missing", *WEATHER, *FLAGS,
           "inbound_arrival_delay_min", "origin_late_rate_3h"]


def bins(x, edges):
    return pd.cut(x, edges, right=False).astype("string").fillna("missing")


def run():
    counts = defaultdict(lambda: [0, 0])
    for minutes in (360, 60):
        features = pq.ParquetFile(ROOT / "phase_b" / f"features_G3_{minutes}m.parquet")
        metadata = pq.ParquetFile(ROOT / "flights_development_phase_b.parquet")
        for x, y in zip(features.iter_batches(65536, columns=COLUMNS),
                        metadata.iter_batches(65536, columns=["flight_id", "LATE", "scheduled_departure_utc"]), strict=True):
            f, m = x.to_pandas(), y.to_pandas()
            if not f.flight_id.equals(m.flight_id.astype("string")) or not m.LATE.isin([0, 1]).all():
                raise ValueError("Development alignment or label mismatch")
            time = pd.to_datetime(m.scheduled_departure_utc, utc=True)
            if time.isna().any() or not time.between(START, DEVELOPMENT_END, inclusive="left").all():
                raise ValueError("Development time boundary failed")
            inbound = bins(f.inbound_arrival_delay_min, [-float("inf"), 0, 15, 60, float("inf")])
            groups = {
                "weather_missing": f.weather_missing,
                "visibility_miles": bins(f.wx_visibility_mi, [-float("inf"), 1, 3, 5, float("inf")]),
                "origin_late_rate_3h": bins(f.origin_late_rate_3h, [0, .1, .25, .5, 1.01]),
                "inbound_delay_minutes": inbound.mask(f[FLAGS].any(axis=1), "unavailable"),
            }
            for var in WEATHER:
                groups[var + "_present"] = f[var].notna()
            month_key = time.dt.year * 100 + time.dt.month
            for variable, values in groups.items():
                for (month, level), part in m.LATE.groupby([month_key, values]).agg(["size", "sum"]).iterrows():
                    cell = counts[minutes, month, variable, str(level)]
                    cell[0] += int(part["size"])
                    cell[1] += int(part["sum"])
    rows = [(*key, n, late, late/n) for key, (n, late) in sorted(counts.items())]
    Path("reports/full_development_eda").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["minutes", "utc_month", "variable", "level", "n", "late", "late_rate"]).to_csv(
        "reports/full_development_eda/feature_bins.csv", index=False)


if __name__ == "__main__":
    run()
