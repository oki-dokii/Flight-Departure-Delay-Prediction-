# Freeze and one-time test run

P1 owns the final G1 evaluation. G0 is the schedule baseline. G2 and G3 remain exploratory development analyses. An operating job, a cache of candidate scores, or PR mergeability cannot authorize test access.

## Revision and boundary decisions

A separate clone returned exit code 0 and **no output** for `git diff --stat 4780bd7 6c2dfb2 -- src tests`. The unrestricted diff contains only `reports/p1_phase_b.md` and `reports/full_development_eda/cutoff_verification.json`. Use **4780bd7c656e486e4710066c89c7601de3d15b7b** as the candidate execution revision because it also identifies the G1 fits. The scorer checks the literal HEAD. Run from a clean detached checkout of that SHA, with the frozen configuration supplied as an external input. Commit the configuration, its hash and the separate audit-tool revision to the reporting branch. Repeat the source/tests comparison if the run revision moves.

**P1 accepted the existing October boundary on November 26.** The release and `src/splits.py` implement [2025-01-01, 2025-10-31) UTC. The latest scheduled timestamp in the development release is October 30 at 23:59 UTC. Final training additionally requires an observed departure before October 31 at 00:00 UTC and after the one-hour checkpoint. This preserves the documented 48-hour gap before November 2. Including the entire UTC day of October 31 requires a different development release, a revised gap, rebuilt features and repeated scoring/rehearsal. Extending a date in the final scorer alone cannot recover those rows. The boundary decision is recorded in the test run log. Team approval of the final model freeze is still pending.

## Subset refit decision

If the sampling design is accepted, refit on eligible members of the fixed 200,000-flight development cohort and score the entire eligible November–December holdout once at both checkpoints. Keep the original cutoff on observed training labels and use full available history for congestion. If a continuous cohort is required, revise and validate the protocol before test access. The full-scale gates and command below apply to the full-development queue. Keep its `pilot=false` requirement intact. For the subset path, bind the rehearsal config to the 200k experiment’s manifest, development and checkpoint hashes, use the subset’s verified 240 rows and selected recipes, rehearse October and capacity against the subset manifest, and keep the same single-run and prediction-rebuild checks. The test manifest must cover the full eligible test population.

## Full-development queue gates

- [ ] Verify both G1 feature hashes and the audited development hash against their releases. Record the feature manifest, configuration, environment, code and run identities. Confirm the full-data cutoff replay at both checkpoints.
- [ ] Require successful classical and boosting runs, selected recipes, per-flight prediction exports and the G1 master table. Verify that each fold's training labels were observable by the training cutoff and that each pair of checkpoints uses the shared eligible IDs. `src.p4_handoff` reconstructs the canonical fits and checks training and validation ID digests, counts, labels and score rows. `src.selected_calibration` checks the selected probability exports. `src.aggregate_results` matches P5's per-fit provenance to P4's and checks the 240 expected validation keys.
- [ ] Explicitly require both experiments to have `pilot=false`, `feature_group=G1`, `fit_mode=full_fold` and the actual full manifest/development/feature hashes. The original aggregator compares the two pilot flags for equality and also has a legacy recipe-free path. Use the stricter publication command below.
- [ ] Rehearse all 20 model/checkpoint combinations on the October validation fold at seed 42. Compare IDs, labels, probabilities and both scores with saved exports. Complete a capacity rehearsal of all eligible development refits on the scoring machine. Give these jobs exclusive use of benchmark resources before optional G0 follow-up.
- [ ] Finish report plots and calibration figures before submission. A rendering failure may be repaired after rehearsal starts. Missing score exports or a failed numerical check blocks the freeze.
- [ ] Before test access, record team agreement, the final 20 recipes, run revision, thresholds, class definitions, exact G1 inputs, machine and package versions, and recovery procedure. Set `frozen_revision` to the execution SHA. If missing-only recovery has not been independently implemented and rehearsed, record that a partial run will require adjudication.

From the reporting checkout containing the release assets and audit helper, use new output directories. Here `B` denotes the verified benchmark output and `M` the released G1 manifest.

```sh
B=data/processed/p4/local_full
M=data/processed/phase_b/manifest_G1.json
python -m tools.freeze_audit validation --manifest "$M" --classical "$B/G1" \
  --boosting "$B/G1_boosting/aggregator_inputs" --output results/full_g1_validation
```

### Live queue evidence

The local queue under `data/processed/p4/local_full/` adopted the existing G1 worker on November 26. Required numerical stages produce `G1_COMPLETE.json` on success. October rehearsal and the capacity run precede `G1_CAPACITY_COMPLETE.json` and G0. Inspect `logs/ordered_queue.log` and `QUEUE_FAILED.json` for failures; neither a missing plot nor an active PID proves that numerical gates passed. Keep the marker contents and rehearsal evidence with the freeze record.

## One final evaluation

1. Verify the written sign-off. Record SHA-256 digests of the configuration, code revision, both manifests, development/holdout files and feature matrices. Verify the scoring checkout's HEAD and clean source tree. The scoring directory must be new.
2. At the authorized run, prepare the reserved archives with the frozen cleaner and feature builder. Use the same G1 schema, historical event availability, checkpoint offsets and units as in development. The locked test window is [2025-11-02, 2026-01-02) UTC. Apply the common eligibility filter once for both horizons. Record the resulting flight count at scoring.
3. Invoke `python -m src.final_run --manifest "$M" --config "$CONFIG" --holdout-manifest "$HOLDOUT"` in the pinned checkout. Preserve stdout/stderr and exit status. The program reserves `data/processed/final_once` exclusively and writes one prediction Parquet per combination. It writes `scores.csv` and `run.json` at the end. A missing summary cannot establish that test predictions were never produced.
4. After all files exist, use the checked audit tool from the reporting checkout with the same verified files. The audits expect manifest paths to resolve to the same data from that repository root. Supply a fresh destination for rebuilt scores. Recompute test metrics before publishing the final table.

```sh
python -m tools.freeze_audit rebuild --folder "$FINAL" --config "$CONFIG" \
  --holdout-manifest "$HOLDOUT" --output "$NEW_SCORES"
python -m tools.freeze_audit validation --manifest "$M" --classical "$B/G1" \
  --boosting "$B/G1_boosting/aggregator_inputs" --test-scores "$NEW_SCORES" \
  --output results/full_g1_final
```

The rebuild checks the final run identity, all 20 output names, unique flight IDs, exact membership and N against the independently filtered holdout, aligned ground-truth labels and valid probabilities. It recomputes average precision and ROC-AUC and compares existing scores at absolute tolerance 1e-12. Report validation mean and population standard deviation separately from test scores, test start and N.

## Crash recovery

1. Stop scoring. Record time, exit status, stage and whether any metric or prediction was viewed. Make a separate read-only snapshot of the full partial directory, logs, manifests and configuration. Preserve the original path and exclusive-run guard. Record a checksum list for the snapshot.
2. Inventory **without computing or viewing metrics**. Record each expected model/checkpoint, file presence, byte count, hash, schema and footer row count. Compare IDs and N with frozen cohort membership without displaying probabilities or labels. Treat truncated, unreadable and mismatched files as disputed artifacts. Preserve them for review.
3. If all predictions are complete, reconstruct the summary from the saved files without refitting. If the crash also prevented `run.json` from being written, first independently reconcile the frozen execution receipt, file hashes and cohort with the team. The rebuild tool requires a final run record; record any authorized reconstruction and its source evidence separately.
4. If predictions are missing, continue **only missing model/checkpoint pairs**, in a fresh recovery directory, using a reviewed missing-only driver, the same code SHA, data hashes, recipes, seeds and library versions. Preserve every previously valid prediction file byte-for-byte. Log narrowly authorized replacements of invalid files and keep their originals. The stock `final_run` has no missing-only switch; starting it in another empty directory would rerun completed models. Without a validated continuation path, stop and disclose the partial evaluation.
5. Assemble a new verified inventory referencing original and recovered files, then rebuild all metrics once. Log the failure, snapshot hash, missing keys, access history, decision, operator, recovery evidence and any effect on interpretation in [the test run log](reports/test_run_log.md). Never change the model selection or retry to improve a score.

## PR #8 merge checklist

- [ ] Boundary decision and team freeze approval recorded.
- [ ] Full G1 numerical handoffs and both rehearsals passed at the recorded revision.
- [ ] Primary master contains exactly 240 validation rows; reconstructed test summary contains 20 unique model/checkpoint pairs with identical eligible IDs and N.
- [ ] Test log and recovery disclosure complete. Reported table generated from verified artifacts. Report PDF opens and pilot/validation/test figures have clear labels.
- [ ] Mandatory UI integrated, smoke-tested from a clean checkout, and linked in README. Verify graders can access the cited results and releases.
- [ ] Run the suite against the exact merge candidate, resolve PR conflicts if present, and review the diff between the trained source and merge candidate. Explain any code differences. Merge only after the checks pass.
