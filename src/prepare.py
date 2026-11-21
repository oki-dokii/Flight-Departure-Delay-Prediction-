"""Build the Phase A development parquet from monthly BTS files."""

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo, reset_tzpath

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from src.download import RAW
from src.splits import DEVELOPMENT_END, OBSERVED, START, TIME

OUTPUT = Path("data/processed")
REQUIRED = {"FlightDate", "Origin", "Tail_Number", "Cancelled", "Diverted",
            "CRSDepTime", "DepTime", "DepDelay", "CRSElapsedTime", "ActualElapsedTime"}
TEXT_COLUMNS = {
    "FlightDate", "Reporting_Airline", "IATA_CODE_Reporting_Airline", "Tail_Number",
    "Origin", "OriginCityName", "OriginState", "OriginStateName",
    "Dest", "DestCityName", "DestState", "DestStateName",
    "DepTimeBlk", "ArrTimeBlk", "CancellationCode",
    *(f"Div{i}{column}" for i in range(1, 6) for column in ("Airport", "TailNum")),
}


def load_airports(path):
    airports = pd.read_csv(path, usecols=["iata_code", "latitude_deg", "longitude_deg", "iso_country"])
    airports = airports.dropna().query("iso_country in ['US', 'PR', 'VI', 'GU', 'AS', 'MP']")
    # A duplicate IATA code is ambiguous even if both locations look plausible.
    return airports.drop_duplicates("iata_code", keep=False).set_index("iata_code")


def hhmm_minutes(values):
    """Convert valid BTS HHMM values to minutes; 2400 denotes next midnight."""
    number = pd.to_numeric(values, errors="coerce")
    hour, minute = number // 100, number % 100
    valid = (number % 1 == 0) & hour.between(0, 23) & minute.between(0, 59)
    valid |= number.eq(2400)
    return (hour * 60 + minute).where(valid)


def departure_times(frame, zones, scheduled_minutes, actual_minutes, delay):
    local_schedule = pd.to_datetime(frame["FlightDate"], errors="coerce", format="%Y-%m-%d")
    local_schedule += pd.to_timedelta(scheduled_minutes, unit="min")
    scheduled = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    actual = scheduled.copy()

    for zone, members in zones.groupby(zones).groups.items():
        tz = ZoneInfo(zone)
        wall_clock = local_schedule.loc[members]
        candidates = [
            wall_clock.dt.tz_localize(tz, ambiguous=dst, nonexistent="NaT").dt.tz_convert("UTC")
            for dst in (True, False)
        ]

        # DepDelay gives elapsed minutes and resolves the actual local calendar date.
        # Validate the resulting wall clock against the independently published DepTime.
        elapsed = pd.to_timedelta(delay.loc[members], unit="min")
        expected_clock = actual_minutes.loc[members] % (24 * 60)
        actual_candidates = [candidate + elapsed for candidate in candidates]
        matches = []
        for candidate in actual_candidates:
            clock = candidate.dt.tz_convert(tz).dt
            matches.append((clock.hour * 60 + clock.minute).eq(expected_clock))

        first, second = candidates
        first_matches, second_matches = matches
        use_first = first_matches & (~second_matches | first.eq(second))
        use_second = second_matches & ~first_matches
        scheduled.loc[members] = first.where(use_first).fillna(second.where(use_second))
        actual.loc[members] = actual_candidates[0].where(use_first).fillna(
            actual_candidates[1].where(use_second)
        )
    return scheduled, actual


def clean_month(raw_frame, zones, *, start=START, end=DEVELOPMENT_END):
    counts = {"input": len(raw_frame)}
    columns = sorted(REQUIRED)
    frame = raw_frame[columns].copy()

    def keep(mask, step):
        nonlocal frame
        before = len(frame)
        if not mask.all():
            frame = frame.loc[mask]
        counts[step] = {"before": before, "after": len(frame)}

    keep(~raw_frame.duplicated(), "exact_duplicates")
    keep(frame["Cancelled"].eq(0), "cancelled_or_unknown")
    counts["diverted_retained_before_time_checks"] = int(frame["Diverted"].eq(1).sum())
    keep(frame["Origin"].map(zones).notna(), "unknown_origin_zone")

    scheduled_minutes = hhmm_minutes(frame["CRSDepTime"])
    actual_minutes = hhmm_minutes(frame["DepTime"])
    delay = pd.to_numeric(frame["DepDelay"], errors="coerce")
    keep(scheduled_minutes.notna() & actual_minutes.notna() & delay.mod(1).eq(0)
         & delay.between(-1440, 10080), "invalid_or_missing_departure_fields")
    scheduled, actual = departure_times(
        frame, frame["Origin"].map(zones), scheduled_minutes.loc[frame.index],
        actual_minutes.loc[frame.index], delay.loc[frame.index]
    )
    frame = frame.assign(**{TIME: scheduled, OBSERVED: actual})
    keep(scheduled.notna() & actual.notna(), "unresolved_or_inconsistent_departure_time")
    keep(frame[TIME].between(start, end, inclusive="left"), "outside_date_window")

    computed = (frame[OBSERVED] - frame[TIME]).dt.total_seconds() / 60
    if not computed.eq(frame["DepDelay"]).all():
        raise ValueError("Computed delay differs from BTS DepDelay")
    frame = frame.assign(LATE=computed.ge(15).astype("uint8"))
    frame["Tail_Number"] = frame["Tail_Number"].astype("string").str.strip().replace("", pd.NA)
    frame["tail_number_missing"] = frame["Tail_Number"].isna()
    scheduled_duration = pd.to_numeric(frame["CRSElapsedTime"], errors="coerce")
    scheduled_duration = scheduled_duration.where(scheduled_duration.between(1, 1440))
    frame["scheduled_arrival_utc"] = frame[TIME] + pd.to_timedelta(scheduled_duration, unit="min")
    # Diverted flights have no reliable arrival at the scheduled destination.
    duration = pd.to_numeric(frame["ActualElapsedTime"], errors="coerce")
    duration = duration.where(frame["Diverted"].eq(0) & duration.between(1, 1440))
    frame["actual_arrival_utc"] = frame[OBSERVED] + pd.to_timedelta(duration, unit="min")
    counts["missing_tail_retained"] = int(frame["tail_number_missing"].sum())
    counts["diverted_retained"] = int(frame["Diverted"].eq(1).sum())
    cleaned = raw_frame.loc[frame.index].copy()
    for name in [TIME, OBSERVED, "LATE", "Tail_Number", "tail_number_missing",
                 "scheduled_arrival_utc", "actual_arrival_utc"]:
        cleaned[name] = frame[name]
    return cleaned, counts


def prepare(raw, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Use the pinned tzdata version even on machines with an OS timezone database.
    reset_tzpath([])
    airports = load_airports(raw / "airports.csv")
    finder = TimezoneFinder()
    zones = {}
    output.mkdir(parents=True, exist_ok=True)
    target = output / "flights_development.parquet"
    partial = target.with_suffix(".part")
    writer = None
    report = {}
    try:
        for month in range(1, 11):
            archive = raw / f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{month}.zip"
            with ZipFile(archive) as source:
                tables = [name for name in source.namelist() if name.lower().endswith(".csv")]
                if len(tables) != 1:
                    raise ValueError(f"Expected one CSV in {archive.name}")
                with source.open(tables[0]) as csv:
                    frame = pd.read_csv(csv, low_memory=False,
                                        dtype=dict.fromkeys(TEXT_COLUMNS, "string"),
                                        usecols=lambda name: not name.startswith("Unnamed:"))
            if not REQUIRED.issubset(frame):
                raise ValueError(f"Missing columns in {archive.name}: {REQUIRED - set(frame)}")
            prefix = f"2025-{month:02d}-"
            if not frame["FlightDate"].str.startswith(prefix, na=False).all():
                raise ValueError(f"Unexpected flight dates in {archive.name}")

            for origin in set(frame["Origin"].dropna()) - zones.keys():
                zones[origin] = None
                if origin in airports.index:
                    airport = airports.loc[origin]
                    zones[origin] = finder.timezone_at(lng=airport.longitude_deg, lat=airport.latitude_deg)

            cleaned, counts = clean_month(frame, zones)
            del frame
            cleaned.insert(0, "flight_id", prefix + cleaned.index.astype(str))
            cleaned.sort_values(TIME, kind="stable", inplace=True)
            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(partial, schema, compression="zstd", compression_level=6)
            writer.write_table(table.cast(schema, safe=True))
            del table, cleaned
            report[prefix.rstrip("-")] = counts
            print(json.dumps({prefix.rstrip("-"): counts}), flush=True)

        writer.close()
        writer = None
        os.replace(partial, target)
        (output / "cleaning_counts.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if writer is not None:
            writer.close()
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    options = parser.parse_args()
    prepare(options.raw, options.output)
