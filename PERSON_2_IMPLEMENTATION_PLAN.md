# Person 2 implementation plan: Phase A

Prepared from the current repository, the supplied Phase A role allocation and the project brief on 2026-11-23. This is an implementation plan; it does not claim that EDA, a complete audit or data validation has already been performed.

## Review corrections before implementation

Reviewed on 2026-11-23. The newly pasted copy ends at figure 10; use this complete repository file as the canonical plan.

The review identified implementation gaps rather than a need to change the frozen protocol: audit enforcement must reach the feature matrix, cleaned data cannot represent all scheduled traffic, arrival timestamps need independent checks, and notebook execution and statistical displays need explicit contracts. Corrections are incorporated below. P1/P3 decisions remain tracked dependencies; they do not block starting P2's development-only EDA.

## 1. Your responsibility, in plain language

You own three things: explain what the data says, decide which columns can safely become model inputs, and assemble the report structure. Your most important ML engineering contribution is preventing apparently strong results that depend on information unavailable when a prediction is made.

P1 owns downloading, cleaning, UTC conversion, labels and splits. P3 owns feature construction and weather joins. P4/P5 own models and tuning. You review the evidence and document decisions across those interfaces; you do not need to implement their pipelines for Phase A.

The team lead's dataset recommendation means:

- **BTS is the main flight table:** one row describes a flight operation, including its schedule and eventual outcomes. Those outcomes provide labels and descriptive analysis, but most cannot be predictors.
- **NOAA weather is an optional second table:** actual observations joined by airport and time. BTS `WeatherDelay` is a post-flight cause attribution, not a substitute for weather measurements.
- **Tail number links aircraft legs:** a previous arrival may help explain the next departure. The link and every observation still need an availability argument at the checkpoint.
- **Suggested scope is guidance:** this repository already uses 2025 and all origins with usable timezone matches. Do not silently replace it with the suggested top 20–30 airports or a different year.

The PDF defines the assessment requirements; the pasted Phase A document allocates team work. The PDF specifies teams of 3–4 while the allocation has five people. Flag this to the lead for instructor confirmation; continue P2 work meanwhile.

## 2. What already exists and what is missing

| Repository item | Observed behavior | P2 action |
| --- | --- | --- |
| `src/download.py` | Downloads all 12 BTS 2025 archives plus pinned airport metadata, with receipts | Reuse provenance; do not run a full download merely for EDA |
| `src/prepare.py` | Opens January–October only; preserves source columns and adds UTC times, label, ID and tail-missing flag | Use its output and review cleaning consequences; do not duplicate cleaning |
| `src/splits.py` | Frozen UTC bounds and four expanding validation folds; separate final-test function | Import constants; use `get_folds()` for fold summaries |
| `src/evaluate.py` | ROC-AUC, average precision labelled PR-AUC, accuracy, majority baseline, three seeds | Match report terminology; do not implement another harness |
| `DATA_SOURCES.md` | Source links, licences, timestamp reconstruction and cleaning decisions | Build on it and verify citations; identify assumptions explicitly |
| `SPLIT_DESIGN.md` | Exact boundaries and gap rationale | Reference it rather than inventing another split |
| `data/download_manifest.json` | Recorded source URLs, hashes and download timestamps | Cite recorded download dates, separately from your access date |
| `data/processed/` and `data/raw/` | Only `.gitkeep` files present in this checkout | Obtain development parquet and cleaning counts from P1/the README-linked release |
| `notebooks/`, `reports/` | Only placeholders present | Add P2 artifacts here |
| Audit, EDA and report | No implementation found | These are your primary deliverables |
| Runtime | README requests Python 3.12; current `python3` is 3.13 and lacks `timezonefinder` | Set up the documented environment before claiming checks pass |

Verification attempted: `python3 -m unittest discover -s tests -v` stopped during import with `ModuleNotFoundError: timezonefinder`. No real-data statistics or test-period data were inspected.

### Fixed data contract

- Input: `data/processed/flights_development.parquet`, plus `cleaning_counts.json`.
- Target: `LATE = 1` for a gate departure delay of **at least 15 minutes**.
- Development UTC interval: `[2025-01-01, 2025-10-31)`.
- Validation months: July, August, September and October, with October ending at the exclusive development boundary. Each training cutoff is one day before validation starts, and both scheduled and actual departure must precede that cutoff.
- Excluded gap: `[2025-10-31, 2025-11-02)`.
- Final test UTC interval: `[2025-11-02, 2026-01-02)`, subject to the source's 2025 local-flight-date cohort. Leave it untouched; test flight count is a final-run placeholder.
- Proposed checkpoint timestamps: `scheduled_departure_utc - 6 hours` and `scheduled_departure_utc - 1 hour`. P3/team must finalize the second checkpoint.

## 3. Deliverables and proposed file layout

| File | Purpose | Completion criterion |
| --- | --- | --- |
| `src/eda.py` | Small reusable input guards, aggregations and quality summaries | Explicit development input; rejects rows outside frozen bounds; no raw archive discovery |
| `notebooks/01_eda.ipynb` | Narrative EDA with 12 figures | Runs top to bottom with relative paths, counts and written observations |
| `COLUMN_AUDIT.csv` | Canonical, machine-readable audit | Exactly one reviewed row per actual parquet column, including derived fields |
| `COLUMN_AUDIT.md` | Readable rules, ambiguity decisions and P3 handoff | Explains target-flight versus historical-flight permissions and links the CSV |
| `reports/data_collection.md` | Approximately half a page on provenance and coverage | Cited collection process, scope, licence/download provenance and limitations |
| `reports/limitations.md` | Running evidence-based limitations | Each item has consequence, mitigation and owner where unresolved |
| `reports/data_quality_issues.csv` | Actionable feedback to P1 | Issue, affected count, examples, severity, proposed action and resolution |
| `reports/report.tex` | Compileable report skeleton | Required sections, owner placeholders, references and results-table headers |
| `reports/figures/`, `reports/tables/` | Reproducible exports used in the report | Stable filenames, explicit cohort and denominators |
| `tests/test_eda.py` | Focused synthetic checks for safeguards | Rejects bad time bounds; verifies audit coverage and label boundary checks |

Keep the notebook narrative and plotting separate from reusable guards. Do not add a large analysis framework. Propose `matplotlib`, `seaborn`, `jupyter` and `nbconvert` as development dependencies with P1-agreed pinned versions; retain existing core dependency pins. LaTeX avoids adding a Word generation workflow; confirm the team's preferred report format during the normal sync.

## 4. Work package A: safe input and quality review

Implement the input guard before any plots. Keep its API independent of `get_folds()`: a legitimate one-month development sample cannot populate all four validation folds. Run fold summaries only for the full development handoff; otherwise explicitly mark them unavailable. `get_folds()` returns positional indices, so always select rows with `.iloc`, never `.loc`.



1. Require the development parquet explicitly. If it is missing, stop with the README handoff instructions. Require cleaning counts for cleaning reconciliation and final sign-off, but allow provisional EDA if only that sidecar is missing; mark the reconciliation pending. In either case, do not automatically download/open November or December.
2. Inspect parquet schema first, then project only columns needed for each analysis. Full BTS data may be too large to load all columns at once; batch missingness counts and other full-data aggregations with PyArrow.
3. Require timezone-aware, nonmissing scheduled timestamps and assert every row is inside `START <= scheduled_departure_utc < DEVELOPMENT_END` imported from `src.splits`.
4. Check nonmissing unique `flight_id`, binary nonmissing `LATE`, required fields and row totals against P1's final per-month cleaning counts. Reconcile monthly counts by local `FlightDate`, not UTC month, since boundary flights can cross dates.
5. Recompute UTC departure-delay minutes and verify `LATE == (delay >= 15)`. Cross-check `DepDelay` and `DepDel15` when present. Document that equality with `DepDelay` is partly by construction: P1 uses it to reconstruct actual UTC time.
6. Report exact duplicates separately from repeated business keys such as date/carrier/flight number/origin/destination/scheduled departure. A repeated key is an investigation, not automatic permission to delete flights.
7. Summarize missingness, invalid HHMM values, out-of-range delay values, nonpositive durations, departure/arrival ordering, tail missingness and diverted rows. Compare raw exclusions using P1's cleaning log; cleaned data cannot reveal records already removed.
8. Flag unusually early departures, long delays and inconsistent scheduled/actual durations. Show frequencies and examples; do not invent new cleaning thresholds or silently trim records.
9. Export a provenance summary: input identity/hash if provided, row count, UTC bounds, local-date range, schema, package versions and sampling seed.

Descriptive EDA can inspect outcomes on development flights. That does **not** authorize using those columns as model inputs. Use the initial training window for decisions such as rare-category thresholds or proposed feature selection. Later development folds may be shown as time-shift diagnostics; record any decisions informed by them and retain the fixed protocol. Never turn full-development late-rate plots into a feature mapping: P3 must recompute historical rates as of each checkpoint and fit learned transforms only within training folds.

## 5. Work package B: notebook and figure plan

Each figure needs a question, clearly labelled units, cohort/date range, denominator and a short evidence-based observation. Use all development rows for aggregated counts where feasible. Limit sampling to expensive visual displays, with a fixed seed and disclosure. Do not balance classes before computing prevalence.

| # | Figure | Engineering interpretation to write |
| --- | --- | --- |
| 1 | `LATE` counts and percentages | Actual prevalence and why accuracy alone is inadequate |
| 2 | Signed departure-delay histogram, central view plus tail inset | Early departures, mass near zero and long right tail; count values outside the display range |
| 3 | Late rate by scheduled **origin-local** hour | Operational time-of-day pattern; normalize `2400` and avoid treating HHMM as a decimal hour |
| 4 | Late rate by scheduled local day of week | Map BTS weekday codes correctly; report flight counts |
| 5 | Monthly late rate with monthly volume | January–October development variation; one year cannot establish recurring seasonality |
| 6 | Late rate by reporting carrier | Use stable reporting-carrier identity; carrier differences may reflect route mix |
| 7 | Late rate by origin airport | Display top origins selected by volume, while retaining the full cohort |
| 8 | Late rate for common directional routes | Origin–destination is directional; show volume and avoid ranking tiny groups |
| 9 | Column missingness percentages by local month | Include every column; identify structural nulls in diversion/delay-cause fields |
| 10 | Late rate around predeclared US holiday windows | Define windows before inspecting outcomes; compare with nearby same-weekday dates, with counts |
| 11 | Scheduled local hour × weekday late-rate heatmap | Reveal interactions; mask low-support cells under a declared descriptive display rule |
| 12 | Missing-tail and diverted status: counts and late rates | Quantify cleaning/coverage implications; these diagnostic outcomes are not approved predictors |

Supporting tables: cleaning funnel from P1's counts; delay quantiles/min/max; duplicate findings; target consistency; fold row counts and label rates using `get_folds()`; coverage by month; data-quality issue register. Use counts as the default uncertainty context. If displaying binomial intervals, label them descriptive independence-based intervals, not reliable significance tests: flights share airport/day/weather shocks. For stronger uncertainty claims, use a documented day-block bootstrap and discuss remaining temporal dependence. Avoid causal claims or significance fishing across many routes.

Holiday labels should use the flight's local scheduled calendar, with an explicitly listed calendar source and actual versus observed dates. Do not attempt Thanksgiving/Christmas analysis using the holdout. Optional later: historical inbound coverage and weather coverage plots after P3 supplies checkpoint-safe features, not fabricated Phase A results.

## 6. Work package C: complete column audit, delivered to P3 by Day 2

Generate the inventory from the actual parquet schema. Do not equate the small `REQUIRED` set in `src/prepare.py` with the complete schema: the preparation code preserves source columns. Read all pages of the BTS dictionary, including diversion fields on later pages. Until the parquet arrives, draft family rules but label schema coverage incomplete.

Minimum required CSV fields: `column`, `type`, `allowed`, `justification`. Add `dtype`, `role`, `allowed_6h`, `allowed_1h`, `available_at`, `source_reference` and `assumption` to make it useful for implementation. Use explicit decisions such as `yes`, `no`, `conditional`; ambiguity defaults to `no`. Separate physical data type from the four availability categories.

| Column family / examples | Availability type | Target-flight model permission |
| --- | --- | --- |
| `FlightDate`, calendar fields, `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, schedule time blocks | Known-in-advance | Yes under documented schedule-as-known assumption; historical BTS is not a versioned schedule feed |
| Carrier, flight number, origin/destination and their ordinary geographic identifiers, `Distance`, `DistanceGroup` | Static or known-in-advance, per field | Generally eligible; redundant IDs and numeric category codes need explicit modelling treatment |
| `Tail_Number` | Static identity with uncertain assignment availability | Conditional; absent independent evidence, exclude from primary checkpoint features or state a justified idealized assignment assumption |
| `tail_number_missing` | Derived from uncertain tail assignment | Same restriction as its source; missingness can itself leak final operational assignment |
| `DepTime`, `DepDelay`, `DepDelayMinutes`, `DepDel15`, `DepartureDelayGroups`, actual UTC departure | Outcome-like | No; label/quality/EDA roles only |
| `TaxiOut`, `WheelsOff`, `WheelsOn`, `TaxiIn`, `ArrTime`, all actual arrival delays/groups, `ActualElapsedTime`, `AirTime`, actual UTC arrival | Outcome-like | No for the target flight |
| All five delay-cause fields, including `WeatherDelay` and `LateAircraftDelay` | Outcome-like | No; weather and aircraft history must be built from separate eligible evidence |
| `Cancelled`, `CancellationCode`, `Diverted` | Outcome-like | No final-status predictors; cancellation used to define cohort, diversion for documented handling/diagnostics |
| Gate-return fields and **all** `Div*` fields, including diversion airport IDs and tail numbers | Outcome-like | No; geographic-looking names do not make diversion outcomes static |
| `LATE` | Outcome-like | Target only |
| `flight_id` | Static technical identifier | Alignment only; no model input because it encodes source month/row identity |
| `scheduled_departure_utc`, `scheduled_arrival_utc` | Known-in-advance in intended meaning | Eligible subject to reconstruction caveat below |
| `Flights` and other constant/helper fields | Verify dictionary meaning per field | Audit explicitly; normally omit constants/redundancy rather than calling them useful predictors |
| Previous-flight outcomes and actual weather observations | Observed-by-T | Conditional derived features only, with observation/availability timestamp at or before each target checkpoint |

The audit needs an enforcement handoff: the current evaluation harness rejects `LATE` in X, but does not reject other forbidden outcomes. P3/P4/P5 must select features through an explicit reviewed allowlist before calling it. Do not implement this by dropping only familiar forbidden column names. Treat `conditional` as blocked until a named assumption is approved. For derived features, maintain a separate feature registry containing source columns, transformation, checkpoint availability and approval; the source-schema audit alone cannot validate a historical aggregate. Include a synthetic check that an injected `DepDelay` column and an unapproved conditional feature are rejected by that integration.

These are starting rules, not a completed per-column audit. Check both directions: every input column has one audit row, and every audit row maps to the current schema. Fail on unknown columns; do not default to allowing every numeric field.

### Availability questions to resolve with P1/P3

1. **Final tail assignment:** the BTS directive describes the tail corresponding to actual gate departure; it does not establish the assignment at either prediction checkpoint. Record the team's assumption, and plan a schedule-only comparison if inbound features assume assignment knowledge.
2. **Outcome-assisted schedule reconstruction:** `departure_times()` uses `DepDelay` and actual local departure to resolve some DST ambiguity. Request a flag for outcome-assisted cases and a sensitivity check excluding ambiguous local schedules or using independently resolved schedule timestamps. A scheduled feature's name does not establish leak safety.
3. **Publication delay:** historical monthly BTS records are not a live operational feed. Event time at or before T is necessary but may not prove availability to an operator by T. Document an idealized operational-feed assumption or require an availability timestamp/latency allowance.
4. **Inbound continuity:** P3 must check the previous aircraft leg actually terminates at the target origin, was eligible at T, and is not a later-assigned or diverted leg. A missing prior leg is not proof that the target is the aircraft's first flight of the day.
5. **Cleaning removes history:** cancellation, invalid departure and missing-time filtering may remove relevant aircraft/airport history. Request coverage indicators and missing-reason handling, not silent zero fills.
6. **Weather:** audit station mapping, observation/report time, availability lag, units, missingness and the chosen provider's licence. Reject nearest-time joins that can select future observations. Reanalysis, forecasts and observations need different availability arguments.

7. **Congestion denominator:** the cleaned parquet excludes cancellations and invalid departure records. It therefore cannot supply an unbiased count of all scheduled departures or a complete airport history. P3 must either obtain a separate schedule/history source preserving those records or label features as counts/rates within the retained cohort. P2 must not describe those counts as total airport demand.
8. **Arrival reconstruction validation:** `actual_arrival_utc` is calculated from departure plus `ActualElapsedTime`; current preparation does not independently compare its destination-local clock with `ArrTime`. Ask P1 to check destination timezone/HHMM and arrival-delay consistency before P3 relies on it for inbound availability. Check `scheduled_arrival_utc` against the scheduled destination-local clock too. Until then record these arrival features as pending validation.
9. **Target already departed:** at the proposed checkpoint, some extremely early departures may already have occurred. Report the fraction where `actual_departure_utc <= checkpoint` for each checkpoint. Ask P1/P3 to define the eligible prediction population consistently; P2 must not silently filter it using outcomes or change fold membership.

Do not implement these P1/P3 changes inside the P2 EDA notebook. Record issue IDs and review their resolution.

## 7. Work package D: collection summary, limitations and report

Write the collection summary from primary references and existing receipts. Cover the electronically filed monthly carrier reports, domestic nonstop passenger scope, reporting/voluntary coverage, local clock fields, gate departure versus takeoff, and the transformation into this project's filtered cohort. Separate reported facts from your assumptions. The 2025 directive is the applicable year-specific reference; avoid substituting a current carrier list for the 2025 one.

Start these limitations with consequences:

- US reporting-carrier coverage limits generalization to international flights and missing operators; codeshares and reporting definitions require care.
- Excluding cancellations makes the target conditional on a usable departure, rather than total passenger disruption.
- One year and development data through October cannot fully characterize recurrent seasons or the final winter/holiday regime.
- Missing legs, tails and diverted arrivals can break aircraft chains.
- Final schedules/tail assignments and assumed feed latency may overstate real-time predictability.
- Timestamp reconstruction and timezone matching have assumptions; extreme-value filters and removed rows affect coverage.
- Airport/carrier differences are associations with confounding by schedules, routes and operating conditions.
- Weather station matching, incomplete observations and as-of joins affect coverage if weather is added.

Use this section ownership in a compileable LaTeX skeleton; placeholders must say “pending,” not contain invented results:

| Report section | Content owner |
| --- | --- |
| Problem, target and checkpoints | P2 coordinates; P1/P3 supply definitions |
| Dataset, collection, source/licence/download date, EDA | P2, with P1 provenance |
| Cleaning, missing-value decisions and UTC handling | P1; P2 integrates evidence |
| Split design and evaluation protocol | P1 |
| Complete column audit | P2 |
| Feature engineering and availability | P3; P2 reviews against audit |
| Classical model setup, tuning and PCA analysis | P4 |
| Boosting setup and tuning | P5 |
| Results table | P4/P5 supply harness exports; P2 assembles |
| Extra analysis | Shared owners named per analysis; P2 coordinates figures |
| Limitations, conclusions and reproducibility | P2 integrates team contributions |

Results-table placeholders: model, checkpoint, ROC-AUC mean/spread, PR-AUC (average precision) mean/spread, optional accuracy, folds/seeds and feature configuration. Include majority baseline, Decision Tree, Random Forest, paired PCA/no-PCA classifier and all five boosting variants. Keep baseline and PCA comparison explicit rather than relying on the Phase A checklist's model count. Report final test start now; leave final test counts/results pending.

For Phase B, reserve figure slots for checkpoint predictability, feature-group ablation, calibration/drift and precision at a fixed alert rate. Their existence does not authorize tuning or opening test data during Phase A.

## 8. Three-day execution order and handoffs

| Day | Ordered work | Concrete output / dependency |
| --- | --- | --- |
| 1 | Read brief/dictionary/directive; inspect code; draft audit rules and collection summary; scaffold report; obtain P1 development handoff | Draft audit, report sections and unresolved availability questions; no need to wait idle for data |
| 2 | Inspect real schema; complete per-column audit; implement development guard and quality summaries; create EDA figures | Audit to P3 **by end of Day 2**; quality issue register to P1; notebook partially populated |
| 3 | Finish 12 plots and observations; reconcile cleaning counts; review feature spec against audit; finish report skeleton and limitations; run checks | Reproducible P2 deliverables and reviewable handoff |

Critical handoff from P1: development parquet, cleaning counts, schema/provenance identity and timezone/reconstruction notes. If unavailable, work with an explicitly identified development-month sample supplied by P1; label every result preliminary and do not claim full-cohort completeness. Do not substitute an unrelated Kaggle dataset.

Handoff to P3: canonical audit, checkpoint decisions/assumptions, missingness risks and specific feature-spec review comments. Handoff to P1: counts and examples for issues; propose changes for P1 to implement. Handoff to P4/P5: report table contract and limitations relevant to encoding/preprocessing. Use the team's requested `p2-eda` branch and reviewed PR when implementation starts; this planning change does not switch branches or publish anything.

## 9. Acceptance checks

- [ ] Development parquet is the only flight input opened by P2 analysis; every plot passes the UTC cohort guard.
- [ ] Notebook executes with an explicit repository-root working-directory contract and deterministic sampling. Resolve paths from a root containing `src/splits.py` and `DATA_SOURCES.md`; do not assume Jupyter starts in the repository root. Run with `nbconvert` using a configured execution directory or establish that root in the first cell.
- [ ] Audit covers every actual source and derived column once; no pending permission is treated as approved.
- [ ] Synthetic checks cover out-of-window/naive/missing timestamps, audit omissions/duplicates and the 14/15-minute label boundary. They never read actual holdout data.
- [ ] Aggregations preserve true prevalence and display group sample sizes; no silent trimming/imputation of source records.
- [ ] Eight to twelve figures (target twelve; consolidate overlapping plots if needed) have written observations and report-ready exports; no claims appear before real-data computation.
- [ ] Cleaning totals and quality findings are reconciled with P1; unresolved items have owners.
- [ ] P3's proposed features comply with the audit or carry an explicit documented unresolved assumption.
- [ ] Collection summary has primary-source citations and source receipt dates; limitations are specific to this pipeline.
- [ ] Report skeleton compiles and has all required sections, owners and honest result placeholders.
- [ ] Existing protocol tests pass in the documented environment; missing dependencies are resolved before reporting success.

## References and verification notes

- Supplied project brief: `Topic2_Flight_Departure_Delay_Prediction.pdf`, especially Sections 3–6 and 9–10. Supplied Phase A allocation defines P2's Day 3 deliverables and Day 2 audit handoff.
- Repository sources inspected: `README.md`, `DATA_SOURCES.md`, `SPLIT_DESIGN.md`, `requirements.txt`, all current `src/*.py`, `tests/test_protocol.py`, and `data/download_manifest.json`.
- [BTS field dictionary](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ): field definitions distinguish schedule, signed delays, clipped delays and operational outcomes. Accessed 2026-11-23; consult all dictionary pages when completing the inventory.
- [BTS 2025 Technical Reporting Directive 39](https://www.bts.gov/explore-topics-and-geography/modes/aviation/number-39-technical-directive-reporting-time): collection requirements, coverage and gate/tail definitions. Accessed 2026-11-23. The event-availability concerns above are engineering inferences from these definitions, not a claim that BTS supplies a real-time feed.
- [BTS on-time tables](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD): primary dataset entry point. Keep the repository's frozen 2025 scope.
- Licences and exact source download receipts are already referenced in `DATA_SOURCES.md`; P2 should verify their underlying terms before final report submission. A NOAA/IEM provider and its terms remain a P3/team decision, not a verified input in this checkout.
