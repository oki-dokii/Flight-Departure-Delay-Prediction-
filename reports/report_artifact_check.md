# P2 report-to-artifact check

The checker reads the compiled report and small committed CSVs only. It imports no model code, fits nothing, and opens no flight Parquet, raw archive, weather observation archive or holdout. Run it separately from G1 training. LaTeX and PDF rendering are single-process tasks; no feature or model regeneration is needed.

```sh
python -m pip install -r requirements-report.txt
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf reports/report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf reports/report.tex
python -m src.report_artifact_check
python -m unittest tests.test_report_artifact_check -v
```

The PDF is `output/pdf/report.pdf`. `reports/report_claims.json` registers each numeric claim with a contextual regular expression, evidence label, source CSV, exact row selector, aggregation, scale and display precision. Comparisons use the rendered precision rather than an arbitrary tolerance. Means are means of already-exported run metrics, not pooled probability recomputations. Optional section bounds disambiguate numeric table rows; absent or ambiguous claims fail. A changed layout can require updating a pattern, but never change an expected number to make a check pass: fix the report or its actual upstream artifact.

Outputs under `reports/checks/`:

- `report_text.txt`: extracted PDF text with page-number footers removed.
- `report_numeric_diff.csv`: claim identity, evidence label, value index, expected display value, extracted value and status.
- `report_check.json`: pass/fail, explicit coverage, blank-cell checks, PDF/manifest/CSV hashes.

Exit status is nonzero for a mismatch, missing/ambiguous claim, nonblank held-out test cell, nonblank selected calibration placeholder or missing evidence label. Source selectors fail closed on missing or nonunique rows. CSV paths are restricted to the committed report table and P5 pilot table directories. Blank cells are checked in both LaTeX source and extracted PDF. The focused synthetic tests cover wrong values, wrong model context, duplicate claims, selector errors, forbidden paths, evidence labels and populated placeholder/test cells. They train no models.

## Evidence labels and coverage

**Full-development validation:** source population/EDA aggregates, arrival audit counts and eligible-fold prevalence/decomposition. These descriptive statistics do not constitute full-scale model validation. **Pilot:** rehearsal feature drift, AP/normalized AP and seed spread, default-model calibration and selected P5 pilot scores/alert metrics. **Held-out test:** closed; all score cells stay empty. Selected full-development model calibration cells also stay empty until the selected prediction handoff is accepted.

The manifest checks the report's EDA table, population count/rate, tail completeness, arrival counts, population drift rates/decomposition, largest feature PSI, rehearsal LightGBM fold metrics/seed spread, calibration narrative and P5 pilot table/AP/alert narrative. This is a reviewable claim inventory, not exhaustive discovery of every number in prose or figures. Protocol dates, horizons, sample design, threshold choices, unit conversions, release hashes/JSON identity, static station metadata, P5 encoding/SHAP descriptions, figure pixels and repeated limitations prose require separate review. When adding statistical findings, register their contextual claim and evidence label in the manifest. Preserve development/pilot identity; never populate selected/full-scale slots from rehearsal defaults.

Weather and tail decisions are consolidated in `reports/tables/p2_exploratory_column_audit.csv`. It supplements `COLUMN_AUDIT.csv`, `FEATURE_REGISTRY.csv` and `WEATHER_SOURCE_AUDIT.csv`; it changes no availability approval or source schema. Full G2/G3 EDA/drift and selected-model calibration remain dependent on frozen development-only matrices and selected probabilities. No external documentation was refreshed during this task: weather findings retain the reviewed revision/date from the existing audit.
