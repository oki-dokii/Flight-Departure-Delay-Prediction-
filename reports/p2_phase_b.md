# P2 Phase B review and implementation

## Gate 0 and work order

Fetched origin; branch `p2-phase-b` starts at `origin/main` commit `25f06faabbd00057e7b6a4b3b92c3774218b8c8d`. Local main and P2-features were left untouched. No push or commit was made. New Python 3.12 venv at a temporary external location; installed requirements.txt, requirements-eda.txt and requirements-boosting.txt. Baseline command `python -m unittest discover -s tests -v`: 31 tests, no failures, one skipped before sample generation. Regenerated pilot with `python -m src.build_sample_features`: 20,000 paired rows and 12 predictors, validated through read_manifest and paired_metadata. No full G0–G3 manifests were present.

Development file: `data/processed/flights_development.parquet`, 265,113,328 bytes, SHA-256 `8166d5f05de4ace74dd75d69ee2867686f45237105599a927f9a519382a29e57`; exactly the Phase A release provenance. No masking claim is made. src.eda was not run before Task 0. First edits separated generated issues/observations from curated files and added repeatability checks; then registry/gate, analyses and report work followed.

## Findings and requests (teammate code unchanged)

- **P3, UTC versus local calendar:** `src/features.py:91–101` obtains hour, weekday and month from `df[TIME].dt`, a UTC timestamp. EDA `src/eda.py` obtains origin-local hour from CRSDepTime. Timezone and DST shift the UTC calendar relative to operations. Justify the choice or provide a paired origin-local calendar ablation; do not describe UTC hour as local hour.
- **P3, silent imputation:** `src/features.py:103–106` coerces then fills CRSElapsedTime and Distance with zero; FEATURE_SPEC.md section 2 requires explicit missingness. `src.p2_evidence` found zero source nulls in each column for this development release, resolving null impact for this release only. Validate coercion/invalid values too, and add explicit indicators or fail if future data violate that evidence.
- **P3/P4/P5, numeric month:** `src/features.py:101` encodes month as an ordered integer. Under `src/splits.py:12–18,36–40`, fold 1 trains before July; November/December would be unseen calendar values at final scoring. Request a no-month ablation on exactly paired folds/seeds and checkpoints. No feature/model was changed by P2.
- **P1, DST final-run risk:** `src/prepare.py:53–75` constructs both fall-back schedule candidates, adds DepDelay and validates against actual local departure. The computed development ambiguity count is zero; that does not clear the repeated 01:00–01:59 hour on 2025-11-02 in DST-observing origins. Resolve the test-window schedule from independent schedule evidence or explicitly document/filter ambiguous cases before freeze; DepDelay must not silently determine a predictor's schedule instant. This is a note for P1, not authorization to open the test.
- **P1/all, spread interpretation:** `src/evaluate.py:20–31` seeds change the month-stratified training draw; `src/evaluate.py:93–95` pools population standard deviation across fold and seed. Report within-fold between-seed spread alongside pooled spread. `drift_seed_spread.csv` decomposes the pooled AP variance into variance of fold means and mean within-fold seed variance. Seeds measure training-draw variation, not independent generalization trials.
- **P3, tail availability and continuity:** FEATURE_SPEC.md section 1 claims published assignment at six hours, whereas source audit and its section 3 quarantine assignment. `src/features.py:229–240` shifts the preceding tail record without a same-day or destination-to-origin continuity condition. D1 limits these to G3; keep pending until the written feed and continuity assumptions are approved. The registry does not authorize G3 merely because its name is requested.

## Reproduction

From root in the new Python 3.12 environment (fixed seeds 42, 73, 101):

```sh
python -m src.build_sample_features
python -m src.p2_evidence
python -m src.p2_registry
python -m src.calibration --manifest data/sample/feature_manifest.json
python -m src.drift --manifest data/sample/feature_manifest.json
python -m unittest discover -s tests -v
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build reports/report.tex
```

The analysis commands accept `--manifest` for full P3 matrices and `--group` for the reviewed group. Calibration accepts selected settings in rehearsal_config.json format via `--config`; the current run uses that file's default pilot settings, not P5's selected tuned configurations. Preprocessing and models come from teammates' factories unchanged. All validation probabilities are in ignored `data/processed/p2_calibration/`, not a final-run directory.

## Blockers and pending handoffs

- P3 full G0/G1/G2/G3 manifests are absent. Feature drift/calibration use the exact paired eligible pilot; full-population feature PSI and tuned/full-scale calibration are pending. Full-population prevalence and composition decomposition are computed separately and clearly labelled.
- Task 5 weather/inbound EDA and coverage remain pending G2/G3 matrices and approval; no raw arrival columns were substituted. Weather metadata fields and station mapping/lag/licence/download date need P3 verification. No G3 results are fabricated.
- P1/P3 arrival-masked release is not the local analysis file. P2 avoids arrival predictors entirely. Existing arrival-check counts are checked from the committed CSV; masking is not claimed.
- Optional recalibration is deferred; no calibrator was fitted to validation/test. P4/P5 must provide selected configs for later calibration reruns, and paired no-month ablations. Final test count and scores remain pending P1's single run.

## Acceptance checklist and evidence

| Check | Result | Evidence |
| --- | --- | --- |
| Gate 0: branch base, new environment, baseline, digest and paired pilot | PASS | Gate section above; p2_evidence.json; manifest loads with 20,000 paired IDs and 12 predictors |
| Generators reproduce audit and preserve curated files | PASS | test_p2_generation runs the complete generator twice on synthetic development; the full real-data generator was also rerun with byte/hash assertions; p2_generation_check.json |
| Source audit equals schema; every delivered manifest predictor registered | PASS | validate_audit on all 116 source fields; 18 registry rows (12 primary and 6 pending tail); both checkpoint gates accept all 12 delivered predictors |
| Unknown/outcome/conditional/group/checkpoint/duplicate/missing checks | PASS | test_p2_audit.py; no changes to evaluate.py, splits.py, features.py, models or teammates' runners |
| A-SCHED computed evidence; D1 documented consistently | PASS | 0 ambiguous development rows in p2_evidence.json; audit prose/CSV, registry, assumptions, limitations, issue register and report |
| Drift text, tables, figures, normalized AP and training-only PSI | PASS (pilot feature scope) | drift_analysis.md; drift_*.csv/PNG/JSON; full paired-population prevalence/composition, exact manifest pilot feature distributions |
| Calibration probabilities, Brier/ECE, day-block intervals and tests | PASS (pilot/default config scope) | 240 evaluations, 480,000 validation probability rows; calibration_summary.csv, calibration_bins.csv, provenance and figures; test_calibration.py |
| P1/P3/P4/P5 review findings with file/line evidence | PASS | Findings above; no teammate feature/model/protocol code edited |
| Closed test/raw-file protection and positional folds | PASS | Read-call/holdout/raw scan of new scripts found only explicit development inputs, manifests and report tables; no raw/ZIP/test input or final scorer invocation; get_folds indices used with .iloc. Existing synthetic boundary tests contain dates but read no real holdout |
| Python 3.12 tests and relative project paths | PASS | Final unittest discovery: 43 tests, all pass, no skips. requirements-p2-lock.txt pins the resolved environment; P2 CLI inputs/outputs are repository-relative |
| LaTeX build and visual QA | PASS | latexmk succeeded; output/pdf/report.pdf is 8 pages; rendered pages inspected; no overfull/underfull boxes or LaTeX warnings |
| Task 5 and full G0–G3/tuned analysis | PENDING | No full P3 matrices or selected classical/all-model configs; tail and weather approval/evidence pending. Optional recalibration deferred until must work complete |

The prior committed folds.csv had pre-eligibility counts: the new generated table follows the unchanged get_folds one-hour target rule. July validation is 616,011 rather than 616,012; later training counts each decrease by one. Rates differ only in the corresponding final digits. No split or filtering rule was changed by P2.

## Files changed

- Audit/output safety: src/eda.py, src/audit.py, src/p2_registry.py, FEATURE_REGISTRY.csv, COLUMN_AUDIT.csv, COLUMN_AUDIT.md, reports/audit_assumptions.json, reports/data_quality_issues.csv, reports/data_quality_generated.csv, reports/eda_generated.md and regenerated reports/tables/folds.csv.
- Analysis implementations: src/p2_evidence.py, src/drift.py, src/calibration.py and requirements-p2-lock.txt. Added focused tests in tests/test_p2_generation.py, test_p2_audit.py, test_drift.py and test_calibration.py.
- Evidence and narrative: reports/p2_phase_b.md, reports/drift_analysis.md, reports/calibration_analysis.md, reports/limitations.md, reports/report.tex, output/pdf/report.pdf, reports/tables/{p2_*,drift_*,calibration_*} and reports/figures/{drift_*,calibration_*}. Ignored runtime pilot matrices and per-flight calibration probabilities stay under data/sample and data/processed.

Needed next: **P1** settle fall-back test schedule reconstruction, publish/reconcile masked full release and retain the single-run freeze; **P3** deliver frozen group manifests, weather provenance and continuity/assignment evidence, justify UTC calendar or run local-time ablation; **P4/P5** provide selected configs and paired no-month comparisons, then rerun full-scale calibration/drift with the same commands. Keep G3 outside primary headline results. No GitHub push was performed.
