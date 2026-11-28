# Data sources and cleaning

## Sources

| Input | Publisher and scope | Use and licence |
| --- | --- | --- |
| 2025 Reporting Carrier On-Time Performance monthly archives | U.S. DOT Bureau of Transportation Statistics (BTS). Reporting carriers' domestic nonstop flights under the 2025 reporting directive. | Public U.S. government data. The related DOT Performance catalog record links to U.S. government works terms. |
| `airports.csv` | OurAirports, downloaded from a pinned snapshot dated 2026-01-01 in the publisher’s GitHub mirror. IATA codes, country codes and coordinates. | Public domain per OurAirports. |
| IANA time zones | Coordinates are mapped with the bundled boundary data in `timezonefinder==6.5.9`; historical UTC offsets use `tzdata==2025.2`. | TimezoneFinder code is MIT and the upstream timezone boundary data is ODbL. IANA tz database data is public domain. |

- [BTS monthly archives](https://www.transtats.bts.gov/PREZIP/)
- [BTS release history](https://www.transtats.bts.gov/ReleaseInfo.asp)
- [BTS field dictionary](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ)
- [2025 BTS technical directive](https://www.bts.gov/explore-topics-and-geography/modes/aviation/number-39-technical-directive-reporting-time)
- [DOT Performance catalog licence](https://catalog.data.gov/dataset/performance-321c9)
- [OurAirports data terms](https://ourairports.com/data/)
- [TimezoneFinder licence](https://timezonefinder.readthedocs.io/en/latest/3_about.html)
- [IANA timezone database distribution](https://data.iana.org/time-zones/tz-link.html)

BTS listed September 2026 as the latest available reporting-carrier month on 2026-11-21. The working cohort uses January through December 2025, so development contains several seasons. Use all reporting origins that can be matched to an unambiguous airport and timezone to retain earlier legs from smaller airports. The team can confirm this scope before freezing feature work.

The downloader records the URL, UTC download timestamp and SHA-256 of each input in adjacent JSON receipts. The reference run's receipts are recorded in [data/download_manifest.json](data/download_manifest.json). November and December archives are reserved unopened for final evaluation.

## Cleaning and data dictionary handoff

1. Remove exact duplicate source rows. Keep distinct records even when their schedule fields match.
2. Exclude cancelled flights and records without an explicit uncancelled status. Retain diverted flights with valid departure data because the prediction target is set at origin.
3. Match origins by IATA code to airport coordinates and IANA zones. Drop rows with unmatched or ambiguous origins.
4. Validate scheduled and actual HHMM fields and require an integer BTS `DepDelay` between −1 day and +7 days. This broad guard rejects unusable extreme values before timestamp reconstruction.
5. Interpret `FlightDate` in the origin zone. Interpret scheduled `2400` as the next day's midnight. Exclude ambiguous or nonexistent local scheduled times using the origin time zone. For a unique scheduled instant, add BTS `DepDelay` and check that the resulting actual local clock matches the published value, including midnight rollovers.
6. Apply the development UTC bounds. Compute `LATE` from actual minus scheduled gate departure using an inclusive 15-minute threshold. Check the resulting minutes against BTS `DepDelay`. This equality is a reconstruction check since `DepDelay` supplies the elapsed time. Matching `DepTime` is the separate wall-clock check.
7. Keep missing tail numbers as null with a `tail_number_missing` flag. Add BTS scheduled elapsed minutes to scheduled departure for `scheduled_arrival_utc`. Use actual elapsed flight time to derive `actual_arrival_utc` only for non-diverted flights with a plausible elapsed time. Otherwise leave it null.

The parquet preserves the original BTS fields and adds `flight_id`, `scheduled_departure_utc`, `actual_departure_utc`, `scheduled_arrival_utc`, `actual_arrival_utc`, `tail_number_missing` and `LATE`. Sort by the relevant UTC event time when building historical features. Use `flight_id` to align metadata and feature matrices from the same input release. Departure and arrival timestamps and delay fields are outcomes for the target flight. A prior flight's outcome may be used only when available by the prediction checkpoint. Treat tail-number availability at the checkpoint as a separate question for the column audit.

Row counts before and after each filtering step are emitted per month and saved to `data/processed/cleaning_counts.json`. The BTS archive may be revised by the publisher. Compare its receipt with the reference manifest when reproducing the dataset.
