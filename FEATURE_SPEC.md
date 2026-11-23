# Feature Specification (Phase A)

**Author:** Person 3 (Feature Engineering Lead)  
**Status:** Frozen for Phase A Prototype  
**Audience:** P1 (Timestamps), P2 (Column Audit & Report), P4 (Classical Models), P5 (Boosting Models)

---

## 1. Checkpoint Design and Justification

Predictions are made at discrete time cutoffs ($T$) defined relative to each flight's scheduled gate departure time in UTC ($\text{CRSDepTime}_\text{UTC}$):

$$\text{Cutoff}_T = \text{scheduled\_departure\_utc} - \Delta T$$

Two distinct checkpoints are established to serve different operational horizons:

### Checkpoint 1: $T = 6\text{ hours}$ ($\Delta T = 6\text{h}$) — Strategic / Dispatch Horizon
- **Operational Reality:** At 6 hours prior to departure, airlines are planning gate assignments, dispatch routing, crew rest rotations, and identifying high-risk connection banks.
- **Available Information:** Full static schedule, seasonal/calendar patterns, published aircraft tail assignment, the scheduled rotation chain, and prior legs already completed earlier in the morning.
- **Unavailable / Unknown Information:** Exact surface queue status at origin, immediate ground turnaround progress, and near-term METAR weather changes.
- **Decision Value:** Allows hub controllers and passengers advance warning to proactively reroute or rebook before arriving at the airport.

### Checkpoint 2: $T = 1\text{ hour}$ ($\Delta T = 1\text{h}$) — Tactical / Gate Operations Horizon
- **Operational Reality:** At 1 hour prior to departure, passenger boarding is imminent (or underway for widebodies), aircraft turnaround is in progress, and ground crews are actively loading bags.
- **Available Information:** Exact landing time and taxi-in delay of the inbound aircraft (if landed by $T$), observed departure queue delays at the origin airport over the last 3 hours, and latest ground conditions.
- **Unavailable / Unknown Information:** Pushback time, taxi-out duration, or actual departure time of the target flight. Any event timestamped $> T$ is strictly forbidden.
- **Decision Value:** High-fidelity tactical prediction for gate managers, ramp coordinators, and connection dispatchers.

---

## 2. Core Protocol Rules & Anti-Leakage Safeguards

1. **The Checkpoint Rule:** A feature for flight $i$ may only compute functions of events whose timestamp $\le \text{Cutoff}_{T}(i)$. Any column or event occurring after $\text{Cutoff}_T(i)$ is strictly quarantined.
2. **Positional Index Invariance:** To maintain strict alignment with P1's `get_folds()` in `src/splits.py` and `evaluate()` in `src/evaluate.py`, `build_features(df, checkpoint_delta)` preserves the input DataFrame's exact row order and index (`assert X.index.equals(df.index)`).
3. **No Silent Imputation:** Missing inbound chains (due to missing tail numbers, first leg of the day, or diverted flights) are surfaced through explicit indicator flags rather than silent default fills.
4. **Target & Outcome Quarantining:** Target flight actual outcomes (`DepTime`, `DepDelay`, `DepDel15`, `TaxiOut`, `WheelsOff`, `ArrTime`, `ArrDelay`, `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`, `actual_departure_utc`) are strictly excluded from the feature matrix.

---

## 3. Feature Dictionary

### Group A: Calendar, Schedule & Route (Static at $T$)

These features are known months in advance upon schedule publication and carry zero leak risk.

| Feature Name | Type | Source Columns | Availability | Definition / Encoding | Leak Risk |
|---|---|---|---|---|---|
| `hour_sin` | Float | `scheduled_departure_utc` | Booking | $\sin(2\pi \cdot \text{hour} / 24)$ | None |
| `hour_cos` | Float | `scheduled_departure_utc` | Booking | $\cos(2\pi \cdot \text{hour} / 24)$ | None |
| `dow_sin` | Float | `scheduled_departure_utc` | Booking | $\sin(2\pi \cdot \text{dayofweek} / 7)$ | None |
| `dow_cos` | Float | `scheduled_departure_utc` | Booking | $\cos(2\pi \cdot \text{dayofweek} / 7)$ | None |
| `month` | UInt8 | `scheduled_departure_utc` | Booking | Calendar month (1–12) | None |
| `crs_elapsed_min` | Float | `CRSElapsedTime` | Booking | Scheduled gate-to-gate block duration in minutes | None |
| `distance` | Float | `Distance` | Booking | Flight distance in miles | None |
| `carrier` | Category | `Reporting_Airline` | Booking | Two-letter IATA airline code | None |
| `origin` | Category | `Origin` | Booking | Three-letter IATA origin airport code | None |
| `dest` | Category | `Dest` | Booking | Three-letter IATA destination airport code | None |

---

### Group B: Airport Congestion & Upstream Delays by $T$

Reflects congestion and ground delay cascades at the origin airport leading up to the checkpoint.

> **CRITICAL PROTOCOL SAFEGUARD (Self-Exclusion):** Only departures that **actually pushed back at or before $\text{Cutoff}_T$** (`actual_departure_utc <= Cutoff_T`) are used to compute the historical delay rate. Future departures (`> Cutoff_T`) are strictly excluded.
>
> Furthermore, **the target flight itself is strictly excluded from its own congestion window**, even if it departed unusually early prior to cutoff.

| Feature Name | Type | Source Columns | Availability | Definition / Formula | Leak Risk |
|---|---|---|---|---|---|
| `origin_departures_3h` | Int32 | `Origin`, `OBSERVED`, $\text{Cutoff}_T$ | $T$ | Number of other flights that actually departed from this origin in $[\text{Cutoff}_T - 3\text{h}, \text{Cutoff}_T]$ (target flight excluded) | None (backward window with self-exclusion) |
| `origin_late_rate_3h` | Float | `Origin`, `OBSERVED`, `LATE`, $\text{Cutoff}_T$ | $T$ | Fraction of other flights departing from this origin in $[\text{Cutoff}_T - 3\text{h}, \text{Cutoff}_T]$ that left $\ge 15\text{m}$ late (target flight excluded) | None (backward window with self-exclusion) |

---

### Group C: Experimental / Opt-In Inbound Aircraft Chain (`Tail_Number`)

> **LEAKAGE & AUDIT RESOLUTION:** BTS data provides retrospective final aircraft tail assignments rather than live dispatch feeds at checkpoint $T$. Following Person 2's column audit classification (`conditional` — *"Final aircraft assignment is not proven available at checkpoint"*), all tail-derived features are **strictly excluded from the primary audited predictor set** (`AUDITED_PREDICTORS`). They are quarantined into an experimental module (`EXPERIMENTAL_TAIL_PREDICTORS`) accessible solely via `include_tail=True`.

When live dispatch feed evidence or ACARS/ADS-B transponder logging is integrated in Phase B, the previous leg's details are linked via `Tail_Number` and evaluated against $\text{Cutoff}_T$.

#### 4-State Inbound Indicator Architecture:
To prevent conflation between missing tails, first flights, and airborne aircraft, four mutually-exclusive states are defined:

1. **`inbound_missing_no_tail` (UInt8):** 1 if `tail_number_missing == 1` or `Tail_Number` is null. No physical chain can be constructed.
2. **`inbound_first_leg_of_day` (UInt8):** 1 if `Tail_Number` is known, but no preceding flight was operated by this aircraft earlier in the same calendar day.
3. **`inbound_not_landed_by_T` (UInt8):** 1 if a preceding flight exists, but either:
   - Its `actual_arrival_utc > \text{Cutoff}_T` (aircraft is still airborne or taxiing at checkpoint), OR
   - Its `actual_arrival_utc` is `null` (e.g. diverted flight with no reliable gate arrival).
4. **Normal Landed State (All three indicators = 0):** The inbound flight has safely arrived at the gate at or before $\text{Cutoff}_T$. Its actual arrival delay and turnaround time are fully observable.

| Feature Name | Type | Role | Source Columns | Status |
|---|---|---|---|---|
| `inbound_missing_no_tail` | UInt8 (0/1) | indicator | `tail_number_missing` | Experimental (Opt-in) |
| `inbound_first_leg_of_day` | UInt8 (0/1) | indicator | `Tail_Number`, `TIME` | Experimental (Opt-in) |
| `inbound_not_landed_by_T` | UInt8 (0/1) | indicator | `actual_arrival_utc`, $\text{Cutoff}_T$ | Experimental (Opt-in) |
| `inbound_arrival_delay_min` | Float | numeric | `actual_arrival_utc`, `scheduled_arrival_utc` | Experimental (Opt-in) |
| `turnaround_buffer_min` | Float | numeric | `scheduled_departure_utc`, `actual_arrival_utc` | Experimental (Opt-in) |
| `inbound_prior_late` | Float | indicator | Prior leg `LATE` | Experimental (Opt-in) |

---

### Group D: Weather Integration (Phase B Roadmap)

In Phase A, weather indicators are prepared as stubs:
- Hourly ASOS/METAR surface observations joined on `(Origin, timestamp)` using the latest report timestamp $\le \text{Cutoff}_T$.
- Variables: Wind speed, gust, visibility (SM), precipitation indicator, ceiling height.
- All missing values encoded with an explicit `weather_missing` flag.

---

## 4. Verification & Unit Test Suite

The feature pipeline in `src/features.py` is guarded by automated test suites in `tests/test_features.py`:
1. **Negative Leak Test:** Perturbs target flight actual departures and other post-cutoff events; verifies feature matrix is identical.
2. **Positive Sensitivity Test:** Perturbs an inbound arrival delay occurring before cutoff; verifies downstream feature accurately reflects the update.
3. **Exact Boundary Test:** Verifies an event occurring at exactly $\text{Cutoff}_T$ is included ($\le \text{Cutoff}_T$).
4. **Diverted Inbound Test:** Verifies null `actual_arrival_utc` automatically triggers `inbound_not_landed_by_T = 1` and does not leak.
5. **Indicator Mutual Exclusivity:** Verifies that no row ever asserts multiple inbound indicator flags simultaneously.
6. **Index & Order Preservation:** Verifies that `X.index.equals(df.index)` is preserved to guarantee bug-free `iloc` execution in `src/evaluate.py`.

---

## 5. Alignment with Person 2 Column Audit (`COLUMN_AUDIT.md`)

All source features strictly conform to the column classifications established by Person 2 in `COLUMN_AUDIT.md` and `COLUMN_AUDIT.csv`:

### Source Column Alignment Table

| Column Name | Audit Classification | Allowed in $X$? | P3 Feature Pipeline Role |
|---|---|---|---|
| `scheduled_departure_utc` | Known-in-advance | Yes | Base timestamp for cutoff $T$, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month` |
| `CRSElapsedTime` | Known-in-advance | Yes | Scheduled duration (`crs_elapsed_min`) |
| `Distance` | Known-in-advance | Yes | Flight distance in miles (`distance`) |
| `Reporting_Airline` | Known-in-advance | Yes | Categorical airline code (`carrier`) |
| `Origin` | Known-in-advance | Yes | Categorical origin airport (`origin`), spatial grouping for congestion |
| `Dest` | Known-in-advance | Yes | Categorical destination airport (`dest`) |
| `Tail_Number` | Conditional | Conditional (Strictly Guarded) | Used solely to link prior legs. Evaluated through 4-state indicator flags. |
| `tail_number_missing` | Static metadata | Yes | Drives `inbound_missing_no_tail` indicator |
| `scheduled_arrival_utc` | Known-in-advance | Yes | Used for prior leg arrival delay calculation when prior leg landed $\le T$ |
| `actual_arrival_utc` | Outcome-like (Prior leg only) | Yes (Prior leg $\le T$ only) | Quarantined if $> T$ or target flight. Used for inbound delay only when landed $\le T$. |
| `actual_departure_utc` (`OBSERVED`) | Outcome-like | No for target flight | Quarantined for target flight. Prior departure events aggregated only when $\le T$. |
| Target Delay Columns (`DepDelay`, `DepTime`, `TaxiOut`, etc.) | Outcome-like | Strictly Forbidden | Quarantined completely from feature matrix $X$ |

### Resolution of Conditional `Tail_Number` Audit
In `COLUMN_AUDIT.csv`, Person 2 flagged `Tail_Number` as `conditional` (*"Final aircraft assignment is not proven available at checkpoint"*).
- **Audit Decision & Pre-Merge Quarantine:** Because BTS tables provide the retrospective final aircraft assignment rather than timestamped live dispatch release snapshots, tail-derived features cannot guarantee live operational availability at $T=360\text{m}$ or $T=60\text{m}$.
- **Resolution:** All tail-derived indicators and inbound delay variables are **strictly excluded from the primary audited predictor set**. The primary delivered feature matrices (`features_360m.parquet` and `features_60m.parquet`) contain only the 12 fully approved predictors (calendar, route, distance, scheduled duration, airline, origin, dest, and backward-looking airport congestion). Tail-derived features are maintained in code as experimental features under `include_tail=True` for Phase B exploration once live dispatch transponder feeds are connected.

---

## 6. Model Handoff Manifest Schema (`feature_manifest.json`)

To ensure seamless integration with P4 (Classical models) and P5 (Boosting models), a standardized JSON manifest (`data/sample/feature_manifest.json`) defines the predictor dictionary, typing roles (`numeric`, `categorical`, `indicator`), and exact checkpoint file locations:

- **Revision:** `phase-a-v1`
- **Release:** `phase-a-data-v1`
- **Checkpoints:** 360 min ($T=6\text{h}$) and 60 min ($T=1\text{h}$)
- **Total Audited Predictors:** 12 validated features matching P4's `FeatureSchema` contract (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month`, `crs_elapsed_min`, `distance`, `carrier`, `origin`, `dest`, `origin_departures_3h`, `origin_late_rate_3h`).
- **Sampled IDs Reference:** `data/sample/sampled_ids.parquet` (20,000 flights matching P4's exact sampler across Jan–Oct 2025).

