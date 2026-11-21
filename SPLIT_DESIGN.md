# Frozen split design

Use scheduled gate departure in UTC to assign flights to chronological periods. Dates below start at 00:00 UTC and intervals include the start and exclude the end. The cohort is limited to BTS local flight dates in 2025. All folds expand from 2025-01-01.

| Fold | Train end (exclusive) | Validation start | Validation end (exclusive) |
| --- | --- | --- | --- |
| 1 | 2025-06-30 | 2025-07-01 | 2025-08-01 |
| 2 | 2025-07-31 | 2025-08-01 | 2025-09-01 |
| 3 | 2025-08-31 | 2025-09-01 | 2025-10-01 |
| 4 | 2025-09-30 | 2025-10-01 | 2025-10-31 |

Each training fold requires both scheduled and actual departure to precede its training end, so labels used to fit a model are already observable. The 24-hour boundary buffer also precedes the proposed 6-hour and 1-hour checkpoints. Model preprocessing and encoders are fit on each fold's training rows. Feature builders must independently enforce event availability at each flight's checkpoint.

The development period ends 2025-10-31. The next 48 hours are excluded. The test period begins **2025-11-02** and ends 2026-01-02, which includes the UTC spillover of December 31 departures. Roughly two months gives a sustained forward test that covers late-year seasonal and holiday variation. The Phase A program only opens January through October archives. The 48-hour gap covers November 1 local schedules that fall on October 31 UTC at Pacific origins, so the development parquet can be built without opening November. October local schedules can also spill into the gap. The final test row count is deferred until the one-time final run.

`get_folds(frame)` returns positional index pairs for development rows in input order. For the final run only, prepare the holdout with the same cleaning rules, combine its metadata with development metadata, and call `get_test_split(frame)`. Final training labels must have been observable before the development end. Keep the held-out period out of EDA, feature selection, tuning and fitting.
