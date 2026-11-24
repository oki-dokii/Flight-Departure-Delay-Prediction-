"""Feature-group & tail-chain ablation analysis for Phase B (P3 Extra Analysis).

Evaluates G0 (schedule), G1 (+congestion), G2 (+weather), and G3 (+tail chain, D1)
at both T=360m and T=60m checkpoints using src.evaluate.evaluate() and a fixed
LightGBM configuration across all 4 rolling folds and SEEDS=(42, 73, 101).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluate import evaluate
from src.models_boosting import make_booster
from src.run_classical import checkpoint_features, paired_metadata, read_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data/manifests"
TABLES_DIR = ROOT / "reports/tables"
FIGURES_DIR = ROOT / "reports/figures"

GROUP_LABELS = {
    "G0": "G0: Schedule & Route",
    "G1": "G1: + Origin Congestion",
    "G2": "G2: + Surface Weather",
    "G3": "G3: + Tail Chain (D1)",
}


def run_ablation(groups: List[str]) -> None:
    all_runs = []

    for group in groups:
        manifest_path = MANIFEST_DIR / f"manifest_{group}.json"
        inputs = read_manifest(manifest_path)
        development = paired_metadata(inputs)

        for cp in inputs.checkpoints:
            print(f"\n=== Evaluating {group} @ {cp.minutes}m ===")
            X = checkpoint_features(cp, inputs.schema, development.index)
            runs, summary = evaluate(
                lambda seed, s=inputs.schema: make_booster("lightgbm", s, seed),
                X,
                development,
            )
            model_runs = runs.loc[runs["estimator"] == "model"].copy()
            model_runs["group"] = group
            model_runs["group_label"] = GROUP_LABELS.get(group, group)
            model_runs["checkpoint_minutes"] = cp.minutes
            all_runs.append(model_runs)

            mean_pr = model_runs["pr_auc"].mean()
            mean_roc = model_runs["roc_auc"].mean()
            print(f"  {group} @ {cp.minutes}m -> PR-AUC={mean_pr:.4f}, ROC-AUC={mean_roc:.4f}")

    df_runs = pd.concat(all_runs, ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    runs_path = TABLES_DIR / "p3_ablation_runs.csv"
    df_runs.to_csv(runs_path, index=False)

    summary_df = (
        df_runs.groupby(["group", "group_label", "checkpoint_minutes"], sort=False)[["pr_auc", "roc_auc", "accuracy"]]
        .agg(["mean", "std"])
    )
    summary_df.columns = [f"{m}_{s}" for m, s in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_path = TABLES_DIR / "p3_ablation_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved ablation summary to {summary_path.relative_to(ROOT)}")

    # Compute incremental deltas per checkpoint
    delta_rows = []
    for cp_min, grp_df in summary_df.groupby("checkpoint_minutes"):
        by_group = grp_df.set_index("group")
        for g_prev, g_curr, desc in [
            ("G0", "G1", "Origin congestion (3h)"),
            ("G1", "G2", "Origin surface weather (D2)"),
            ("G2", "G3", "Inbound tail chain (D1 ablation)"),
        ]:
            if g_prev in by_group.index and g_curr in by_group.index:
                delta_rows.append({
                    "checkpoint_minutes": cp_min,
                    "transition": f"{g_prev} -> {g_curr}",
                    "feature_addition": desc,
                    "delta_pr_auc": by_group.loc[g_curr, "pr_auc_mean"] - by_group.loc[g_prev, "pr_auc_mean"],
                    "delta_roc_auc": by_group.loc[g_curr, "roc_auc_mean"] - by_group.loc[g_prev, "roc_auc_mean"],
                })
    deltas_df = pd.DataFrame(delta_rows)
    deltas_path = TABLES_DIR / "p3_ablation_deltas.csv"
    deltas_df.to_csv(deltas_path, index=False)
    print(f"Saved incremental deltas to {deltas_path.relative_to(ROOT)}")

    # Plot grouped bar chart of PR-AUC across groups and checkpoints
    pivot_mean = summary_df.pivot(index="group", columns="checkpoint_minutes", values="pr_auc_mean").loc[groups]
    pivot_std = summary_df.pivot(index="group", columns="checkpoint_minutes", values="pr_auc_std").loc[groups]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(len(groups))
    width = 0.35
    bars_360 = ax.bar(
        [i - width / 2 for i in x],
        pivot_mean[360],
        width,
        yerr=pivot_std[360],
        capsize=4,
        label="T = 6h (360 min)",
        color="#4C72B0",
    )
    bars_60 = ax.bar(
        [i + width / 2 for i in x],
        pivot_mean[60],
        width,
        yerr=pivot_std[60],
        capsize=4,
        label="T = 1h (60 min)",
        color="#DD8452",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=12, ha="right")
    ax.set_ylabel("Validation PR-AUC (mean ± std across 4 folds × 3 seeds)")
    ax.set_title("Phase B Feature-Group Ablation (LightGBM on Full 5.75M Cohort)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in list(bars_360) + list(bars_60):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "p3_ablation_chart.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved ablation figure to {fig_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=["G0", "G1", "G2", "G3"])
    args = parser.parse_args()
    run_ablation(args.groups)


if __name__ == "__main__":
    main()
