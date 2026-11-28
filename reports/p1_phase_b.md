# Phase B protocol and data handoff

The Phase B v1 development Parquet contains the same 5,749,041 flight IDs and departure labels as Phase A. Use it to build full-scale features. The release also carries the reproducible 20,000-flight Phase A checkpoint pilot. Its manifest targets the original Phase A development release. Extract pilot matrices and manifest under `data/sample/` and the revised development file under `data/processed/`. Run `shasum -a 256 -c SHA256SUMS` in the download directory before moving files.

## Population and seed decision

Use flights still awaiting departure at the one-hour checkpoint for both horizons. One development flight departed exactly 60 minutes early. Its history record remains available for other flights. The paired target population is 5,749,040. Across the development data, zero rows share the composite key of local flight date, reporting airline, flight number, origin, destination and scheduled local departure. Retain distinct IDs.

Reconstructed arrivals were converted to each destination's pinned timezone and compared with BTS HHMM, including midnight rollover. Of 5,749,041 scheduled arrivals, 153 clocks disagreed. Of 5,732,258 comparable actual arrivals, 79 disagreed. Another 16,783 actual arrivals were unavailable for comparison. The release masks only the mismatched timestamp fields. See `reports/tables/arrival_checks.csv` and `arrival_exceptions.csv`. Rebuild using `python -m src.audit_arrivals && python -m src.phase_b_dataset`.

D5 uses seeds 42, 73 and 101 for common training draws within each fold, stratified proportionally by scheduled UTC month. Each seed draws about 90% of the available training flights, capped at about 500,000 per fold, with at least two per month when available. All candidates share the draw for a given fold and seed. Validation includes every eligible row in the validation cohort. Fit all preprocessing on the draw. Select by mean validation average precision. Final refits use every eligible development flight whose label was observed before the boundary.

## Results and verification

The committed master and summary are protocol pilots on the same paired 20,000 IDs used for the Phase A handoff. They cover ten configurations including majority, two checkpoints, four folds and three seeds. They establish a reproducible comparison while full-scale tuning is in progress. Each run records parameters, code revision, row counts, runtime and metrics. After selection, run the aggregator on the selected configuration outputs from the same cohort, feature group and revision. Use a separate output directory for each experiment.

```bash
python -m src.aggregate_results --classical data/processed/phase_b/classical \
  --boosting data/processed/phase_b/boosting --experiment phase_b_protocol_pilot_20k
python -m src.final_run --manifest data/sample/feature_manifest.json \
  --config results/rehearsal_config.json
```

The second command rehearses on fold four at seed 42, saves per-flight probabilities and labels, and writes scores in `data/processed/final_rehearsal/`. In the reference run all 20 model/checkpoint pairs matched the independent evaluator's ROC-AUC and PR-AUC within 1e-12.

## Single final run after team freeze

The November and December source ZIPs reside in `data/raw/`. Both compressed-file digests were checked against the source manifest. The EDA notebook calls the development-only reader in `src/eda.py`. Final test starts at 2025-11-02 00:00 UTC. Its eligible flight count is produced at scoring time.

1. Freeze the full development manifest, input hashes, audited predictors, chosen parameters per model and checkpoint, seed and code revision. Start from the rehearsal configuration format. Supply sklearn pipeline parameter names to override defaults. Set `frozen_revision` in the run configuration to the exact checked-out commit. Keep the signed-off configuration alongside the run record.
2. At final evaluation, prepare the held-back November and December archives using `from src.prepare import prepare, RAW, OUTPUT; prepare(RAW, OUTPUT / "holdout", holdout=True)`. The shared cleaner applies the UTC test bounds and destination-clock checks. Build matching holdout matrices using P3's frozen feature builder and eligible event history through each checkpoint. Supply all cleaned test IDs to both files. The scorer enforces common checkpoint eligibility. Create a second manifest of the same format, with its metadata path pointing to `flights_holdout.parquet` and its checkpoint paths pointing to the held-out matrices.
3. Run `python -m src.final_run --manifest FULL_DEV_MANIFEST --config FROZEN_CONFIG --holdout-manifest HOLDOUT_MANIFEST` from the repository root. The scorer reserves `data/processed/final_once/` exclusively before reading holdout records. Existing output stops repeat scoring. Preserve a partial directory for investigation if interrupted.
4. Use the aggregator with the selected validation runs and `--test-scores data/processed/final_once/scores.csv`. The prediction files contain flight ID, observed label and probability per model/checkpoint. The run record contains exact test start, eligible test flight count and training size.

The full weather and ablation matrices are P3's deliverable. Publish their digests and frozen manifests when available, before the team authorizes final scoring.

## Primary feature group and development cutoff verification

The primary benchmark uses G1 (schedule and recent origin congestion). G0 is the schedule baseline. The G2 weather analysis is exploratory because the archive gives observation times and does not establish when operators received the reports. G3 is the retrospective tail-assignment ablation.

The development check aligned the released G0 and G1 matrices with the audited development data by flight ID and rebuilt the schedule columns without departure outcomes. For each origin, an independent calculation counted actual departures and late departures in the inclusive three-hour interval ending at the checkpoint. It excluded the target flight whenever its own departure fell in that interval. All counts and late fractions matched the G1 matrices across 5,749,041 flights at each horizon. One target needed self-exclusion at one hour. See the [checkpoint counts](full_development_eda/cutoff_verification.json).
