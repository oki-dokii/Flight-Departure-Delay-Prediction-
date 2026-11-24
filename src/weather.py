"""Weather data pipeline for Flight Departure Delay Prediction (Phase B, Decision D2).

Downloads hourly surface weather observations (METAR/ASOS) from the Iowa
Environmental Mesonet (IEM) archive, maps BTS origin IATA airports to ASOS
station IDs via OurAirports metadata, and exports normalized UTC weather
observations for strict backward-looking asof joins at prediction checkpoints.

Data Source & Licence:
- Source: Iowa Environmental Mesonet (IEM) ASOS/METAR Archive
- URL: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
- Licence: Public Domain / Open Data (Iowa State University Mesonet)
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
AIRPORTS_CSV = ROOT / "data/raw/airports.csv"
DEV_PARQUET = ROOT / "data/processed/flights_development_phase_b.parquet"
STATION_MAP_PATH = ROOT / "data/weather/station_map.csv"
WEATHER_PARQUET = ROOT / "data/processed/weather_asos_2025.parquet"
WEATHER_MANIFEST_PATH = ROOT / "data/weather/weather_manifest.json"

IEM_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_LICENCE = "Public Domain / Open Data (Iowa Environmental Mesonet, Iowa State University)"


def _iem_candidate_station(ident: str, iata: str) -> str:
    """Convert OurAirports ident/IATA into IEM ASOS station identifier.

    CONUS stations in IEM use 3-letter FAA/IATA IDs (e.g. 'JFK' instead of 'KJFK'),
    whereas non-CONUS stations (Alaska, Hawaii, territories) use 4-letter ICAO IDs
    such as 'PANC', 'PHNL', 'TJSJ'.
    """
    ident = str(ident).strip().upper()
    iata = str(iata).strip().upper()
    if len(ident) == 4 and ident.startswith("K"):
        return ident[1:]
    if len(ident) == 4 and ident[0] in {"P", "T"}:
        return ident
    return iata or ident


def build_station_map(
    origins: Iterable[str],
    airports_csv: Path = AIRPORTS_CSV,
    out_path: Path = STATION_MAP_PATH,
) -> pd.DataFrame:
    """Map BTS Origin IATA codes to IEM ASOS station IDs using airports.csv."""
    if not airports_csv.exists():
        raise FileNotFoundError(f"Missing airports metadata at {airports_csv}")

    airports = pd.read_csv(
        airports_csv,
        usecols=["ident", "iata_code", "name", "iso_country", "latitude_deg", "longitude_deg", "type"],
    )
    airports = airports[airports["iata_code"].notna()].copy()
    # Prefer large/medium airports if duplicate IATA codes exist
    type_rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}
    airports["rank"] = airports["type"].map(type_rank).fillna(9)
    airports = airports.sort_values(["iata_code", "rank"]).drop_duplicates(subset=["iata_code"], keep="first")
    lookup = airports.set_index("iata_code")

    rows = []
    for origin in sorted(set(origins)):
        if origin in lookup.index:
            rec = lookup.loc[origin]
            station = _iem_candidate_station(rec["ident"], origin)
            rows.append({
                "origin_iata": origin,
                "station_id": station,
                "icao_ident": rec["ident"],
                "station_name": rec["name"],
                "latitude_deg": float(rec["latitude_deg"]),
                "longitude_deg": float(rec["longitude_deg"]),
                "mapped": True,
            })
        else:
            rows.append({
                "origin_iata": origin,
                "station_id": "",
                "icao_ident": "",
                "station_name": "",
                "latitude_deg": np.nan,
                "longitude_deg": np.nan,
                "mapped": False,
            })

    df_map = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_map.to_csv(out_path, index=False)
    return df_map


def _fetch_iem_batch(
    stations: List[str],
    year1: int = 2025,
    month1: int = 1,
    day1: int = 1,
    year2: int = 2026,
    month2: int = 1,
    day2: int = 3,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch ASOS observations for a batch of station IDs from IEM."""
    station_params = "&".join(f"station={s}" for s in stations)
    data_params = "&".join(f"data={d}" for d in ("tmpf", "sknt", "gust", "vsby", "p01i"))
    url = (
        f"{IEM_BASE_URL}?{station_params}&{data_params}"
        f"&year1={year1}&month1={month1}&day1={day1}"
        f"&year2={year2}&month2={month2}&day2={day2}"
        "&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no"
        "&missing=null&trace=0.0001&direct=no&report_type=3&report_type=4"
    )

    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip() or "station,valid" not in raw:
                return pd.DataFrame()
            return pd.read_csv(
                io.StringIO(raw),
                na_values=["null", "M", ""],
                low_memory=False,
            )
        except Exception as exc:
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch IEM batch after {retries} attempts: {last_err}")


def download_asos_weather(
    station_map: pd.DataFrame,
    out_parquet: Path = WEATHER_PARQUET,
    manifest_path: Path = WEATHER_MANIFEST_PATH,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Download and normalize 2025 ASOS observations for all mapped stations."""
    stations = sorted(s for s in station_map.loc[station_map["mapped"], "station_id"].unique() if s)
    print(f"Downloading IEM ASOS observations for {len(stations)} stations in batches of {batch_size}...")

    frames = []
    for i in range(0, len(stations), batch_size):
        batch = stations[i : i + batch_size]
        df_batch = _fetch_iem_batch(batch)
        if not df_batch.empty:
            frames.append(df_batch)
        print(f"  Batch {i // batch_size + 1}/{(len(stations) + batch_size - 1) // batch_size}: "
              f"{len(df_batch):,} rows")

    if not frames:
        raise RuntimeError("No weather observations downloaded from IEM")

    raw_df = pd.concat(frames, ignore_index=True)

    # Normalize timestamps and physical units
    raw_df["obs_time_utc"] = pd.to_datetime(raw_df["valid"], utc=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["station", "obs_time_utc"]).copy()

    tmpf = pd.to_numeric(raw_df["tmpf"], errors="coerce")
    sknt = pd.to_numeric(raw_df["sknt"], errors="coerce")
    gust = pd.to_numeric(raw_df["gust"], errors="coerce")
    vsby = pd.to_numeric(raw_df["vsby"], errors="coerce")
    p01i = pd.to_numeric(raw_df["p01i"], errors="coerce")

    norm = pd.DataFrame({
        "station": raw_df["station"].astype(str).str.strip(),
        "obs_time_utc": raw_df["obs_time_utc"],
        "wx_temp_c": ((tmpf - 32.0) * (5.0 / 9.0)).astype("float32"),
        "wx_wind_speed_kt": sknt.astype("float32"),
        # When station reports wind speed but no gust, gust is 0.0 kt
        "wx_wind_gust_kt": np.where(sknt.notna() & gust.isna(), 0.0, gust).astype("float32"),
        "wx_visibility_mi": vsby.astype("float32"),
        # When station reports observation but no precip column, precip is 0.0 mm
        "wx_precip_1h_mm": (p01i.fillna(0.0) * 25.4).astype("float32"),
    })

    # Keep only rows that have at least one primary meteorological measurement
    has_obs = norm[["wx_temp_c", "wx_wind_speed_kt", "wx_visibility_mi"]].notna().any(axis=1)
    norm = norm.loc[has_obs].sort_values(["obs_time_utc", "station"]).reset_index(drop=True)
    # Deduplicate identical (station, obs_time_utc), keeping last report
    norm = norm.drop_duplicates(subset=["station", "obs_time_utc"], keep="last").reset_index(drop=True)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    norm.to_parquet(out_parquet, index=False, compression="zstd")

    # Update station map with actual coverage status
    covered_stations = set(norm["station"].unique())
    station_map = station_map.copy()
    station_map["has_2025_data"] = station_map["station_id"].isin(covered_stations)
    station_map.to_csv(STATION_MAP_PATH, index=False)

    sha256 = hashlib.sha256(out_parquet.read_bytes()).hexdigest()
    manifest = {
        "source": "Iowa Environmental Mesonet (IEM) ASOS/METAR Archive",
        "url_template": IEM_BASE_URL,
        "licence": IEM_LICENCE,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "date_range_utc": ["2025-01-01T00:00:00Z", "2026-01-03T00:00:00Z"],
        "stations_requested": len(stations),
        "stations_with_data": int(len(covered_stations)),
        "origins_total": int(len(station_map)),
        "origins_covered": int(station_map["has_2025_data"].sum()),
        "observations_rows": int(len(norm)),
        "parquet_path": str(out_parquet.relative_to(ROOT)),
        "sha256": sha256,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(norm):,} weather observations to {out_parquet} "
          f"({manifest['origins_covered']}/{manifest['origins_total']} origins covered).")
    return norm


def load_weather_context(
    weather_parquet: Path = WEATHER_PARQUET,
    station_map_path: Path = STATION_MAP_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-sorted weather observations and origin->station mapping."""
    if not weather_parquet.exists() or not station_map_path.exists():
        raise FileNotFoundError(
            f"Weather artifacts missing ({weather_parquet}, {station_map_path}). "
            "Run `python -m src.weather` first."
        )
    weather_df = pd.read_parquet(weather_parquet)
    weather_df["obs_time_utc"] = pd.to_datetime(weather_df["obs_time_utc"], utc=True)
    weather_df = weather_df.sort_values("obs_time_utc").reset_index(drop=True)
    station_map = pd.read_csv(station_map_path)
    return weather_df, station_map


def main():
    if not DEV_PARQUET.exists():
        raise FileNotFoundError(f"Development parquet not found: {DEV_PARQUET}")
    origins = pq.read_table(DEV_PARQUET, columns=["Origin"])["Origin"].to_pandas().dropna().unique()
    station_map = build_station_map(origins)
    download_asos_weather(station_map)


if __name__ == "__main__":
    main()
