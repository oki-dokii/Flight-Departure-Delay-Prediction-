# Selected-model lift, cost and grouped SHAP

## Full-development validation — alert budget

Record the feature group, checkpoint, selected recipe, revision and validation prediction manifest. Duplicate a row for each model, checkpoint and budget. Compare on matching flight IDs. Rank by descending score within each fold and seed, breaking ties by ascending flight ID. For budget fraction b and N flights, alert the first k = ceil(bN). If D departures were late and H were alerted late departures, precision = H/k, lift = (H/k)/(D/N), and cost per flight = [k + 5(D − H)]/N. Leave lift undefined when D is zero. Compute each metric before averaging across fold–seed runs. Give the mean and population SD and retain the monthly breakdown so changes in prevalence are visible.

| Model / group | Checkpoint | Budget | Precision | Lift | Cost per flight |
|---|---|---:|---:|---:|---:|
| | | 5% | | | |
| | | 10% | | | |
| | | 20% | | | |

Report where the budget catches more late flights, how the illustrative cost changes as the budget expands, and whether the pattern holds across months. The cost assumptions describe an alerting scenario. They do not estimate savings from preventing delays.

## Full-development validation — grouped SHAP

Record the selected model and encoding at each checkpoint, fold, seed, validation case count, training-background source and permutation count. For comparability with the pilot, explain up to 128 October validation flights using seed 42, outer-training backgrounds and 32 permutation orders. Replace each original input before encoding so its one-hot columns move together. Average signed contributions across permutations for each flight, then average their absolute values across flights. Keep units on the predicted-probability scale.

| Model / group | Checkpoint | Original feature | Mean absolute contribution | Rank |
|---|---|---|---:|---:|
| | 6 h | | | |
| | 1 h | | | |

Describe whether schedule or recent congestion dominates at each horizon. Mean absolute contributions measure magnitude. A claim about direction requires signed case-level evidence. Correlated inputs can share attribution and marginal replacements can produce unusual feature combinations.

## Held-out test

Populate only from authorized frozen-run artifacts under the freeze runbook. Record the run ID, prediction manifest and cohort size before copying scores. Preserve blank cells until then.

| Model / group | Checkpoint | Budget | N | Precision | Lift | Cost per flight |
|---|---|---|---:|---:|---:|---:|
| | 6 h | | | | | |
| | 1 h | | | | | |
