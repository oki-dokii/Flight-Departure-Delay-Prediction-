"""Render the Phase B classical figures from saved diagnostics."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.phase_b_classical import CHECKPOINTS


def score_curve(axis, rows, x, label=None):
    stats = rows.groupby(x, sort=False).pr_auc.agg(["mean", lambda v: v.std(ddof=0)])
    axis.errorbar(stats.index, stats["mean"], yerr=stats.iloc[:, 1], marker="o", capsize=2, label=label)


def save(fig, target, title):
    fig.suptitle(title)
    fig.savefig(target)
    plt.close(fig)


def render(folder, output):
    output.mkdir(parents=True, exist_ok=True)
    load = lambda name: pd.read_csv(folder / f"{name}.csv")
    identity = json.loads((folder / "experiment.json").read_text())
    scope = "pilot" if identity["pilot"] else "full development cohort"
    tag = f'{identity["feature_group"]} {scope}'
    candidates = load("candidate_scores")
    for name in ("experiment.json", "runtime.json", "selected_recipes.json", "summary.csv", "pca_selected_deltas.csv"):
        copyfile(folder / name, output / name)
    candidates.groupby(["model", "checkpoint_minutes", "parameters"], sort=False).agg(
        pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
        seconds=("runtime_seconds", "mean"), fits=("fold", "size")).reset_index().to_csv(
            output / "search_summary.csv", index=False)
    load("rules").query("fold == 4 and seed == 42").to_csv(output / "october_tree_rules.csv", index=False)
    loadings = load("loadings")
    loadings.assign(magnitude=loadings.loading.abs()).sort_values("magnitude", ascending=False).groupby(
        ["checkpoint_minutes", "component"]).head(5).drop(columns="magnitude").to_csv(
            output / "october_pca_loadings.csv", index=False)
    params = pd.json_normalize(candidates.parameters.map(json.loads)).add_prefix("recipe.")
    candidates = candidates.join(params)
    paths = load("paths")
    selected = load("selected_study")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for minutes, (depth_ax, alpha_ax, path_ax) in zip(CHECKPOINTS, axes, strict=True):
        rows = candidates.loc[candidates.checkpoint_minutes.eq(minutes)]
        tree = rows.loc[rows["recipe.family"].eq("decision_tree")].copy()
        tree["depth"] = tree["recipe.parameters.classifier__max_depth"].map(
            lambda value: "Unbounded" if pd.isna(value) else str(int(value)))
        tree["alpha"] = tree["recipe.parameters.classifier__ccp_alpha"]
        tree["min_leaf"] = tree["recipe.parameters.classifier__min_samples_leaf"]
        winner = selected.query('checkpoint_minutes == @minutes and model == "decision_tree"').iloc[0]
        choice = json.loads(winner.parameters)["parameters"]
        base = tree.loc[tree.alpha.eq(0) & tree.min_leaf.eq(choice["classifier__min_samples_leaf"])]
        score_curve(depth_ax, base, "depth")
        depth_ax.set(xlabel="Maximum depth", ylabel="Validation PR-AUC", title=f"{minutes // 60}h  Depth at α = 0")
        for name, group in tree.loc[tree.model.str.contains("_alpha_")].groupby(tree.model.str.split("_alpha_").str[0]):
            base = tree.loc[tree.model.eq(name)]
            score_curve(alpha_ax, pd.concat([base, group]).sort_values("alpha"), "alpha", name)
        alpha_ax.set_xscale("symlog", linthresh=1e-7)
        alpha_ax.set(xlabel="Pruning α", ylabel="Validation PR-AUC", title="Pruning performance")
        alpha_ax.legend(fontsize=7)
        for label, path in paths.query("checkpoint_minutes == @minutes and fold == 1 and seed == 42").groupby("candidate"):
            path_ax.step(path.alpha, path.impurity, where="post", label=label)
        path_ax.set_xscale("symlog", linthresh=1e-7)
        path_ax.set(xlabel="Effective α", ylabel="Leaf impurity", title="First-fold training path")
        path_ax.legend(fontsize=7)
    save(fig, output / "decision_tree.pdf", f"Decision Tree on {tag}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    trees = "recipe.parameters.classifier__n_estimators"
    fields = [f"recipe.parameters.classifier__{key}" for key in ("max_depth", "min_samples_leaf", "max_features", "class_weight")]
    for minutes, axis in zip(CHECKPOINTS, axes, strict=True):
        forest = candidates.loc[candidates.checkpoint_minutes.eq(minutes) & candidates.model.str.startswith("forest_")]
        finalist = forest.loc[forest[trees].eq(256)].iloc[0]
        for field in fields:
            forest = forest.loc[forest[field].isna() if pd.isna(finalist[field]) else forest[field].eq(finalist[field])]
        forest = forest.loc[forest[trees].ge(64)].sort_values(trees)
        score_curve(axis, forest, trees)
        elapsed = forest.groupby(trees).runtime_seconds.mean()
        other = axis.twinx()
        other.plot(elapsed.index, elapsed, color="tab:orange", marker="s")
        axis.set(xlabel="Number of trees", ylabel="Validation PR-AUC", xticks=[64, 128, 256], title=f"{minutes // 60}h")
        other.set_ylabel("Mean seconds per fit and score", color="tab:orange")
    save(fig, output / "random_forest.pdf", f"Random Forest finalist on {tag}")

    spectrum, deltas = load("spectrum"), load("pca_selected_deltas")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for minutes, (curve, changes) in zip(CHECKPOINTS, axes, strict=True):
        lines = spectrum.loc[spectrum.checkpoint_minutes.eq(minutes)]
        lines = lines.loc[lines.fit_mode.eq(identity['fit_mode'])].drop_duplicates(["fold", "seed", "component"])
        for _, fit in lines.groupby(["fold", "seed"]):
            curve.plot(fit.component, fit.cumulative_variance, color="tab:blue", alpha=0.3)
        for target in (0.9, 0.95):
            curve.axhline(target, color="gray", linestyle="--", linewidth=0.6)
        curve.set(title=f"{minutes // 60}h  Full training variance", xlabel="Components", ylabel="Cumulative variance")
        for name, pair in deltas.loc[deltas.checkpoint_minutes.eq(minutes)].groupby("model"):
            stats = pair.groupby("fold").delta_pr_auc.agg(["mean", lambda v: v.std(ddof=0)])
            changes.errorbar(stats.index, stats["mean"], yerr=stats.iloc[:, 1], marker="o", label=name)
        changes.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        changes.set(xlabel="July–October fold", ylabel="PCA minus control PR-AUC", xticks=[1, 2, 3, 4])
        changes.legend()
    save(fig, output / "pca.pdf", f"Matched PCA comparison on {tag}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    importance = load("importance")
    for minutes, axis in zip(CHECKPOINTS, axes, strict=True):
        rows = importance.loc[importance.checkpoint_minutes.eq(minutes)]
        stat = rows.groupby("feature").pr_auc_drop.agg(["mean", lambda v: v.std(ddof=0)]).sort_values("mean")
        axis.barh(stat.index, stat["mean"], xerr=stat.iloc[:, 1], capsize=2)
        axis.set(xlabel="Validation PR-AUC drop", title=f"{minutes // 60}h")
    save(fig, output / "permutation_importance.pdf", f"Selected tree permutation importance on {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.folder, args.output)
