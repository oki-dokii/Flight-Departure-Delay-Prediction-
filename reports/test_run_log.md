# Final G1 test run log

**Status on November 26, 2026: awaiting freeze.** This entry documents preparation only.

| Record before test access | Value |
|---|---|
| Frozen execution revision and audit-tool revision | Pending |
| Confirmed development boundary, test interval and eligibility rule | Pending sign-off |
| Manifest, matrix, development, holdout and configuration hashes | Pending |
| Rehearsal evidence, capacity result and team approval | Pending |
| Environment, operator and authorized start time | Pending |

### Development preparation on November 26

The running classical experiment records G1, full-fold mode, Python 3.12.13 and training revision `4780bd7c656e486e4710066c89c7601de3d15b7b`. Its development digest and both checkpoint digests match the published SHA-256 lists. The on-disk G1 manifest hashes to `4327e89b7969930603b714eae3079620ba770506191624fc8230e272614a87b8`. The experiment identifier is `80af11b9b78fe22649873e4bdcf782f6cbe4c6956e85d73242df162e5560da0a`. The saved cutoff replay covers 5,749,041 development flights at each checkpoint.

The existing split excludes scheduled and observed departures at or after October 31, 2025 at 00:00 UTC from final training. The test starts November 2 at 00:00 UTC after a 48-hour gap. This is the implemented boundary awaiting team sign-off. Final recipes, capacity, authorization and test access remain pending.

After execution, record exit status, finish time, eligible N, 20 prediction hashes, rebuilt-score comparison and final table location.

For every interruption, append the timestamp, failing stage, exit status, snapshot path and checksum-manifest digest, artifact inventory, whether any test outputs were viewed, missing or disputed model/checkpoint keys, adjudication and authorizer, unchanged execution identity, recovered files and hashes, and the publication disclosure. Preserve earlier entries.
