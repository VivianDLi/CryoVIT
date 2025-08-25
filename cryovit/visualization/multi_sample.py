"""Make plots comparing multi sample performance."""

import functools
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from statannotations.Annotator import Annotator

from cryovit.visualization.utils import merge_experiments, significance_test, compute_stats

matplotlib.use("Agg")
colors = sns.color_palette("deep")[:2]
sns.set_theme(style="darkgrid", font="Open Sans")

hue_palette = {
    "3D U-Net": colors[0],
    "CryoViT": colors[1],
    "SAM2": colors[2],
    "MedSAM": colors[3],
}

group_names = {
    "hd": "Diseased",
    "healthy": "Healthy",
    "old": "Aged",
    "young": "Young",
    "neuron": "Neurons",
    "fibro_cancer": "Fibroblasts and Cancer Cells",
}

def plot_df(df: pd.DataFrame, pvalues: pd.Series, title: str, ax: Axes):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        pvalues (pd.Series): Series containing p-values for annotations.
        title (str): The title of the plot.
        ax (Axes): Axes object for plotting the figure.
    """
    sample_counts = df["Sample"].value_counts()
    sorted_samples = sample_counts.sort_values(ascending=True).index.tolist()

    params = dict(
        x="Sample",
        y="TEST_DiceMetric",
        hue="Model",
        data=df,
        order=sorted_samples,
    )

    sns.boxplot(
        showfliers=False,
        palette=hue_palette,
        linewidth=1,
        width=0.6,
        medianprops=dict(linewidth=2, color="firebrick"),
        ax=ax,
        **params,
    )
    sns.stripplot(
        dodge=True,
        marker=".",
        alpha=0.5,
        palette="dark:black",
        ax=ax,
        **params,
    )

    k1, k2 = df["Model"].unique()
    pairs = [[(s, k1), (s, k2)] for s in pvalues.index]

    annotator = Annotator(ax, pairs, **params)
    annotator.configure(color="blue", line_width=1, verbose=False)
    annotator.set_pvalues_and_annotate(pvalues.values)

    current_labels = ax.get_xticklabels()
    new_labels = [
        f"{label.get_text()}\n(n={sample_counts[label.get_text()] // 2})"
        for label in current_labels
    ]

    ax.set_ylim(-0.05, 1.15)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Dice Score")
    ax.set_xticklabels(new_labels, ha="center")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="lower right", shadow=True)


def process_multi_experiment(exp_type: str, exp_group: Tuple[str, str], exp_names: Dict[str, str], exp_dir: Path, result_dir: Path):
    result_dir.mkdir(parents=True, exist_ok=True)
    df = merge_experiments(exp_dir, exp_names, keys=["Model", "Type"])
    forward_df = df[df["Type"] == "forward"]
    backward_df = df[df["Type"] == "backward"]

    s1_count = forward_df["Sample"].nunique()
    s2_count = backward_df["Sample"].nunique()

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[s1_count, s2_count]) # Set width ratios based on unique sample counts
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot forward comparison (s1 vs. s2)
    p_values = {}
    for model in exp_names.values():
        if model == "CryoViT":
            continue
        test_fn = functools.partial(significance_test, model_A="CryoViT", model_B=model, key="Model", test_fn="wilcoxon")
        m_name = model.replace(" ", "").lower()
        p_values[model] = compute_stats(forward_df, group_keys=["Sample", "Model"], file_name=result_dir / f"{exp_group.join("_")}_{m_name}_{exp_type}_stats.csv", test_fn=test_fn)
    title = f"{group_names[exp_group[0]]} to {group_names[exp_group[1]]} Shift"
    plot_df(forward_df, p_values, "Model", title, ax1)

    # Plot backward comparison (s2 vs. s1)
    p_values = {}
    for model in exp_names.values():
        if model == "CryoViT":
            continue
        test_fn = functools.partial(significance_test, model_A="CryoViT", model_B=model, key="Model", test_fn="wilcoxon")
        m_name = model.replace(" ", "").lower()
        p_values[model] = compute_stats(backward_df, group_keys=["Sample", "Model"], file_name=result_dir / f"{list(reversed(exp_group)).join("_")}_{m_name}_{exp_type}_stats.csv", test_fn=test_fn)
    title = f"{group_names[exp_group[1]]} to {group_names[exp_group[0]]} Shift"
    plot_df(backward_df, p_values, "Model", title, ax2)
    
    # Adjust layout and save the figure
    fig.suptitle(f"Model Comparison Across {group_names[exp_group[0]]}/{group_names[exp_group[1]]} Domain Shifts")
    fig.supxlabel("Sample Name (Count)")
    fig.supylabel("Dice Score")
    
    plt.tight_layout()
    plt.savefig(result_dir / f"{exp_group[0]}_{exp_group[1]}_domain_shift.svg")
    plt.savefig(result_dir / f"{exp_group[0]}_{exp_group[1]}_domain_shift.png", dpi=300)