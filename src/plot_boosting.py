"""Summarize and plot P5 validation outputs for a single feature manifest."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models_boosting import MODELS


def summarize(folder):
    runs = pd.read_csv(folder / "operations.csv")
    keys = ["model", "checkpoint", "fold", "seed"]
    if runs.duplicated(keys).any() or len(runs) != 240:
        raise ValueError("Expected 240 distinct model/checkpoint/fold/seed scores")
    selected = runs.loc[runs.model.isin(MODELS)].rename(columns={"checkpoint": "checkpoint_minutes"})
    selected.to_csv(folder / "selected_boosting_runs.csv", index=False)
    # Match P1's aggregator interface so the final sweep can feed the shared master.
    for (minutes, name), group in selected.groupby(["checkpoint_minutes", "model"]):
        dest = folder / "aggregator_inputs" / str(minutes) / name
        dest.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["model", "checkpoint_minutes"]).assign(estimator="model").to_csv(
            dest / "runs.csv", index=False)
    metric_cols = ["pr_auc", "roc_auc", *[c for c in runs if c.startswith(("precision_at", "lift_at", "cost_at"))]]
    grouped = runs.groupby(["model", "checkpoint"])[metric_cols]
    summary = grouped.mean().join(grouped.std(ddof=0), rsuffix="_std").reset_index()
    summary.to_csv(folder / "summary.csv", index=False)
    candidates = pd.read_csv(folder / "candidates.csv")
    enc = candidates.groupby(["model", "checkpoint", "native", "candidate"]).pr_auc.mean()
    enc.groupby(level=[0,1,2]).max().rename("best_mean_pr_auc").to_csv(folder / "encoding.csv")
    for metric, title in [("lift", "Lift over fold prevalence"),
                          ("cost", "Illustrative cost per flight")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for ax, cp in zip(axes, [360, 60]):
            data = summary.query("checkpoint == @cp").set_index("model")
            for name, row in data.iterrows():
                ax.plot([5, 10, 20], [row[f"{metric}_at_{p}"] for p in (5,10,20)],
                        marker="o", linestyle="-" if name in MODELS else "--", label=name)
            ax.set(title=f"{cp // 60} hours before departure", xlabel="Alert budget (% of flights)", xticks=[5,10,20])
            ax.grid(alpha=.2)
        axes[0].set_ylabel(title)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8)
        fig.tight_layout(rect=[0,.13,1,1])
        fig.savefig(folder / f"{metric}.png", dpi=150)
        plt.close(fig)
    impacts = pd.read_csv(folder / "shap.csv")
    pivot = impacts.pivot(index="feature", columns="checkpoint", values="mean_abs_shap")
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    ax = pivot.rename(columns={360:"6 hours",60:"1 hour"}).plot.barh(figsize=(8,5))
    ax.set(xlabel="Mean absolute contribution to delay probability", ylabel="")
    ax.figure.tight_layout()
    ax.figure.savefig(folder / "shap.png", dpi=150)
    plt.close(ax.figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
