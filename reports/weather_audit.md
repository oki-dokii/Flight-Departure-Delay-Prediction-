# P2 weather audit and handoff — 2026-11-25

**Decision: all six candidates remain pending at both checkpoints.** Reviewed P3 revision `065c2a1` and P4 revision `0c41969` without merging or changing their feature/model/harness code. Static manifests and metadata snapshots are under `reports/audits/p3_065c2a1/`. `reports/tables/weather_predictor_audit.csv` is the six-row audit; `FEATURE_REGISTRY.csv` includes those candidates and the blocked delivered alias. External lineage is in `WEATHER_SOURCE_AUDIT.csv`; the source Parquet audit remains unchanged.

## Name selected by the user

The canonical column is **`weather_missing`**, role `indicator`, nonmissing binary 0/1. User selected this on 2026-11-25. P4 already requires this name (`src/phase_b_classical.py:73–74` at the reviewed revision). P3 must rename `wx_weather_missing` in `src/features.py:59,276,331`, manifests, tests and spec, regenerate affected matrices and hashes, and confirm the handoff. P2 does not silently rename matrix columns or authorize both names. P3/P4 acknowledgement remains pending; no contact/thread for those teammates is connected here.

## Predictor review

| Canonical predictor | Source and units | Current handling / audit result |
| --- | --- | --- |
| wx_temp_c | IEM tmpf: °F → °C, `(tmpf−32)×5/9` | Conversion consistent with source definition; availability and partial-null coverage pending |
| wx_wind_speed_kt | sknt: knots | Identity conversion; partial-null coverage pending |
| wx_wind_gust_kt | gust: knots | Missing gust becomes 0 when speed exists; missing does not prove no gust; preserve null or supply a documented source-backed rule |
| wx_visibility_mi | vsby: miles | Identity conversion; partial-null coverage pending |
| wx_precip_1h_mm | p01i: inches → mm, ×25.4 | Missing precipitation becomes 0; cannot establish dry conditions; retain missingness. Accumulation is since previous hourly reset, not necessarily exactly 60 minutes |
| weather_missing | availability flag: 0/1 | Requires P3 rename. Flags absent/unmapped or >6-hour-old report; does not represent per-variable completeness |

Unit definitions and missing-value cautions checked against [IEM download documentation](https://mesonet.agron.iastate.edu/request/download.phtml?network=IA_ASOS). The site describes missing observations as potentially unreported, sensor-missing or quality-control-missing; zero precipitation is therefore not established by a missing value. [IEM terms](https://mesonet.agron.iastate.edu/disclaimer.php) state public-domain use; attribution is appreciated. P3 records download time `2026-11-24T18:31:33.259597+00:00`; P2 verified the receipt metadata, not the observation artifact. Source lineage/code evidence: `src/weather.py:170–196` at the reviewed P3 revision.

## Station mapping

The supplied map has 349 unique origins and 349 distinct station IDs, no missing identifiers. P2 compared against the official [IEM station catalog](https://mesonet.agron.iastate.edu/geojson/network/AZOS.geojson), retrieved 2026-11-25. There are 347 single catalog matches within 5 km and 2 unresolved IDs: **PBI and PPG**. The 5 km cutoff is a declared review convention, not proof of spatial representativeness. PPG maps from ICAO NSTU to PPG in the heuristic; P3 must verify its supported IEM identifier. Do not automatically substitute a station from proximity alone. The current catalog does not prove 2025 sensor identity or operating history. Per-origin evidence, distances and hashes are saved in weather_station_checks.csv and weather_mapping_summary.json. Reproduce the static comparison with `python -m src.weather_audit --catalog PATH_TO_CATALOG`.

The map's whole-year `has_2025_data` flag is deliberately excluded from the persisted static map and never used as development coverage. Flight-weighted missingness by checkpoint/fold/month and individual-variable coverage remain pending the actual matrices.

## Observation time versus availability

`src/features.py:299–327` performs a backward station-specific asof join with exact matches and accepts an observation age of six hours. This enforces **observation_time <= cutoff**, including equality; it does not prove **receipt/available_at <= cutoff**. `src/weather.py:195–196` keeps the last duplicate observation timestamp without a receipt/revision timestamp. P3/P1 must supply publication/receipt evidence or approve a named explicit operational-feed/latency assumption, including historical corrections and same-timestamp duplicates. A six-hour staleness threshold is not a latency allowance. `A-WEATHER` remains pending with no source permissions.

The preprocessing retains a report if any of temperature, speed or visibility is present (`src/weather.py:192–196`). Thus `weather_missing=0` can coexist with null individual variables. P2's coverage checker reports partial nulls separately and rejects stale/missing-report rows carrying numeric values. No coverage percentages are inferred from metadata.

## Closed-data protection and downstream handoff

P3's weather artifact metadata spans dates beyond development. Its loader (`src/weather.py:239`) reads that entire artifact, and the feature builder calls it without a development filter (`src/build_phase_b_features.py:119–122`). **P2 did not run either loader or read/download that observation artifact.** Request development-only weather context/receipt evidence rather than opening the mixed-period file. No raw ZIP or November/December flight/observation records were read.

The local checkout and published releases contain neither full G0–G3 matrices nor P4/P5 selected validation probability files. P3's committed manifests and checksum list alone are not the data handoff. Do not rebuild/guess weather values from raw arrivals, substitute pilot results as full-scale results, or refit default models and label them selected-model calibration.

P3 must provide development-only paired matrices and hashes, corrected missing name/semantics, per-flight matched observation/station/availability sidecar (for timing audit), and the Phase B development release identity. G3 inbound views remain D1 ablation-only and require the feed/assignment/continuity review; no raw arrival substitution is permitted.

P4/P5 must provide selected probability exports for all folds/seeds/checkpoints, selected runs, experiment identity, exact manifest/development hashes, feature group, fit mode and training cohort. The reader `python -m src.selected_calibration --handoff HANDOFF.json` validates IDs/labels/days/keys and reproduces exported AP/ROC-AUC (and Brier where supplied), then reports Brier/skill, calibration-in-the-large, ECE, bin n and day-block bootstrap intervals **without fitting any model**. Template: `reports/templates/selected_calibration_handoff.json`. P5 must deliver the same identity columns or an explicitly documented export adapter; old pilot scores alone are insufficient.

After approved full matrices arrive, use `python -m src.drift --manifest MANIFEST --group G2 --performance reports/selected_calibration/tables/calibration_summary.csv --output reports/full_g2_drift` for full feature PSI/KS and matched performance. Weather/inbound late-rate-bin EDA and per-variable coverage are pending that handoff; existing pilot findings retain their original labels. Full analyses, selected reliability/Brier findings and any resulting report conclusions are pending, not failed runs.
