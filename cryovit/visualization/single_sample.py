"""Make plots comparing single sample performance."""

import functools
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from cryovit.config import Sample
from cryovit.visualization.utils import merge_experiments, significance_test, compute_stats

matplotlib.use("Agg")
colors = sns.color_palette("deep")[:4]
sns.set_theme(style="darkgrid", font="Open Sans")

hue_palette = {
    "3D U-Net": colors[0],
    "CryoViT": colors[1],
    "SAM2": colors[2],
    "MedSAM": colors[3],
    "CryoViT with Sparse Labels": colors[1],
    "CryoViT with Dense Labels": colors[2],
}

def plot_df(df: pd.DataFrame, pvalues: Dict[str, pd.Series], key: str, title: str, file_name: str):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        pvalues (Dict[str, pd.Series]): Dictionary containing p-values for annotations for each model.
        key (str): The column name used to group data points in the plot.
        title (str): The title of the plot.
        file_name (str): Base file name for saving the plot images.
    """
    sample_counts = df["sample"].value_counts()
    num_models = df[key].nunique()
    sorted_samples = sample_counts.sort_values(ascending=True).index.tolist()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    params = dict(
        x="sample",
        y="dice_metric",
        hue=key,
        data=df,
        order=sorted_samples,
    )

    sns.boxplot(
        showfliers=False,
        palette=hue_palette,
        linewidth=1,
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

    for k2 in pvalues:
        pairs = [[(s, "CryoViT"), (s, k2)] for s in pvalues[k2].index]
        annotator = Annotator(ax, pairs, **params)
        annotator.configure(color="blue", line_width=1, verbose=False)
        annotator.set_pvalues_and_annotate(pvalues[k2].values)

    current_labels = ax.get_xticklabels()
    new_labels = [
        f"{Sample[label.get_text()].value}\n(n={sample_counts[label.get_text()] // num_models})"
        for label in current_labels
    ]

    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, ha="center")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.suptitle(title)
    fig.supxlabel("Sample Name (Count)")
    fig.supylabel("Dice Score")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="lower center", shadow=True)

    plt.tight_layout()
    plt.savefig(f"{file_name}.svg")
    plt.savefig(f"{file_name}.png", dpi=300)

def process_single_experiment(exp_type: str, exp_group: str, exp_names: Dict[str, List[str]], exp_dir: Path, result_dir: Path):
    result_dir.mkdir(parents=True, exist_ok=True)
    df = merge_experiments(exp_dir, exp_names, keys=["model"])
    p_values = {}
    for values in exp_names.values():
        model = values[0]
        if model == "CryoViT":
            continue
        test_fn = functools.partial(significance_test, model_A="CryoViT", model_B=model, key="model", test_fn="wilcoxon")
        m_name = model.replace(" ", "").lower()
        p_values[model] = compute_stats(df, group_keys=["sample", "model"], file_name=result_dir / f"{exp_group}_{m_name}_{exp_type}_stats.csv", test_fn=test_fn)
    if exp_type != "sparse":
        plot_df(df, p_values, "model", f"Model Comparison on Individual {exp_group.upper()} Samples for {exp_type.capitalize()}", result_dir / f"{exp_group}_{exp_type}_comparison")
    else:
        plot_df(df, p_values, "model", "CryoViT: Sparse vs Dense Labels Comparison on Individual Samples", result_dir / f"sparse_vs_dense_comparison")