#!/usr/bin/env bash
set -euo pipefail
[[ "$(git rev-parse HEAD)" == c54898849d3e5fc089be843a16095ba7c9f9b8e4 ]] || { echo 'Use the revision in START_HERE.md'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
B=data/backup_200k
R=data/processed/backup_200k_g1
mkdir -p "$R/logs"
python -m data.backup_200k.verify
case "${1:-}" in
 classical)
    python -m src.phase_b_classical --manifest "$B/manifest_G1.json" --parent-manifest "$B/manifest_G0.json" --group G1 --pilot --class-weights --forest-jobs "${FOREST_JOBS:-2}" --output "$R/classical" 2>&1 | tee "$R/logs/classical.log"
    python -m src.p4_handoff --manifest "$B/manifest_G1.json" --run "$R/classical"
    ;;
 boosting)
    python -m src.phase_b_boosting --manifest "$B/manifest_G1.json" --classical-run "$R/classical" --output "$R/boosting" 2>&1 | tee "$R/logs/boosting.log"
    python -m src.plot_boosting "$R/boosting"
    python -m src.aggregate_results --classical "$R/classical" --boosting "$R/boosting/aggregator_inputs" --experiment g1_backup_200k --output "$R/results"
    ;;
 *) echo 'Choose classical or boosting'; exit 1 ;;
esac
