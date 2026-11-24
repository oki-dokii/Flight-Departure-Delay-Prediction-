# Feature Specification (Phase B)

**Author:** Person 3 (Feature Engineering Lead)  
**Revision:** `phase-b-v1` | **Data Release:** `phase-b-data-v1` (`flights_development_phase_b.parquet`)  
**Audience:** P1 (Protocol & Final Run), P2 (Column Audit & Report), P4 (Classical Models), P5 (Boosting Models)

---

## 1. Checkpoint Design and Justification

Predictions are made at discrete time cutoffs ($T$) defined relative to each flight's scheduled gate departure time in UTC ($\text{CRSDepTime}_\text{UTC}$):

$$\text{Cutoff}_T = \text{scheduled\_departure\_utc} - \Delta T$$

- **Strategic Dispatch Horizon ($T = 360\text{ min}$ / $6\text{h}$):** Models dispatch planning, long-range rotation expectations, and early-day weather regimes.
- **Tactical Gate Horizon ($T = 60\text{ min}$ / $1\text{h}$):** Models gate operations, recent surface weather reports, 3-hour origin departure congestion, and inbound aircraft arrival status.

All full-scale feature Parquets contain **all 5,749,041 `flight_id`s** from `flights_development_phase_b.parquet` (required by `src/final_run.py`). `src/splits.py` and `src/final_run.py` filter the 1 early-departed flight (`observed <= scheduled - 1h`) at train/evaluation time while keeping its record in `history_df`.

---

## 2. Feature Groups (`G0`–`G3`)

Each feature group is exported as paired 360m and 60m Parquets with its own manifest (`data/manifests/manifest_{G0,G1,G2,G3}.json`) conforming to P4's `FeatureSchema` and `read_manifest()` contracts:

| Group | Included Features | Count | Audit & Protocol Status |
|---|---|---|---|
| **G0** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month`, `crs_elapsed_min`, `distance`, `carrier`, `origin`, `dest` | 10 | Approved (Static schedule & route) |
| **G1** | **G0** + `origin_departures_3h`, `origin_late_rate_3h` | 12 | Approved (Phase A audited set) |
| **G2** | **G1** + `wx_temp_c`, `wx_wind_speed_kt`, `wx_wind_gust_kt`, `wx_visibility_mi`, `wx_precip_1h_mm`, `wx_weather_missing` | 18 | **Primary Audited Phase B Set** |
| **G3** | **G2** + `inbound_missing_no_tail`, `inbound_first_leg_of_day`, `inbound_not_landed_by_T`, `inbound_arrival_delay_min`, `turnaround_buffer_min`, `inbound_prior_late` | 24 | **Decision D1 Ablation Only** |

---

## 3. Core Anti-Leakage & Protocol Rules

1. **The Checkpoint Rule:** No event timestamped $> \text{Cutoff}_T(i)$ may enter feature computation for flight $i$.
2. **Target Self-Exclusion:** Origin airport congestion counts and delay rates over $[\text{Cutoff}_T - 3\text{h}, \text{Cutoff}_T]$ exclude the target flight itself when present in `history_df`, even if it departed unusually early.
3. **Cancellation Exclusion Rationale:** BTS On-Time records do not timestamp *when* a cancellation decision was announced. Aggregating cancellations by scheduled departure time would leak post-cutoff cancellations, so cancellation rates are excluded.
4. **Weather Backward `asof` Join (Decision D2):**
   - **Source:** Iowa Environmental Mesonet (IEM) ASOS/METAR hourly archive (`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`), Public Domain / Open Data.
   - **Join Rule:** `pd.merge_asof(..., direction="backward", allow_exact_matches=True)` matching `obs_time_utc <= Cutoff_T` per mapped origin station (`data/weather/station_map.csv`).
   - **Staleness Policy:** Observations older than **6 hours** relative to `Cutoff_T`, or unmapped stations, set `wx_weather_missing = 1` (`uint8`) and leave numeric weather values as `NaN` (imputed fold-locally by downstream pipelines).
5. **Decision D1 Resolution (Tail-Chain Features):** BTS provides retrospective final aircraft tail assignments rather than live dispatch feeds. Tail-derived inbound features are omitted from the primary audited set (**G2**) and isolated in **G3** as a clearly labelled ablation under the operational-feed assumption, computed on P1's arrival-audited `flights_development_phase_b.parquet`.
6. **Positional Index Invariance & `FeatureSchema` Compliance:** `X.index.equals(df.index)` is strictly enforced; `indicator` columns are strictly binary `{0, 1}` with zero `NaN`s; `flight_id` is stored as the first physical column.
