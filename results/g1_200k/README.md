# G1 benchmark on 200,000 development flights

[Model results](results_summary.md) · [240 fold and seed scores](master_results.csv) · [Lift](lift.png) · [Illustrative cost](cost.png) · [Permutation SHAP](shap.png)

The cohort uses seed 42 and proportional sampling without replacement across January–October. Each validation target has both checkpoints. Congestion uses the full available flight history. The recorded results use the training revision in classical_experiment.json. Its pilot flag identifies an explicit subset. The 20,000-flight pilot, full-development refits and held-out test are separate experiments.

To reproduce, use Python 3.12 and the full development and feature releases described in the project README. Create a separate checkout at c54898849d3e5fc089be843a16095ba7c9f9b8e4. From this results folder copy the three reproduction scripts and SHA256SUMS into data/backup_200k in that checkout. From its root run:

```sh
python -m pip install -r requirements-boosting.txt matplotlib==3.10.3
python data/backup_200k/build_subset.py /path/to/checkout-with-full-development-releases
python -m data.backup_200k.verify
bash data/backup_200k/run.sh classical
bash data/backup_200k/run.sh boosting
```

The builder verifies release checksums before sampling. The verifier checks packaged file hashes and cohort alignment. The model runners record the selected recipes, environment, training identities and validation IDs. All published mean scores and population standard deviations use four monthly folds and three seeds. No held-out results have been generated.

The [congestion audit](congestion_audit.json) compares all sampled IDs with the full source matrices. Both congestion columns matched at both checkpoints. Training and feature code at the two benchmark revisions is identical; Git shows only the added report checker under src between 4780bd7 and c548988. The experiments use distinct cohorts and select settings independently. Fold/seed spread conditions on one cohort draw.
