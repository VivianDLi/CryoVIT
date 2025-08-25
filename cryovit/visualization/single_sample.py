"""Make plots comparing single sample performance."""

import functools
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

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

def plot_df(df: pd.DataFrame, pvalues: pd.Series, key: str, title: str, file_name: str):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        pvalues (pd.Series): Series containing p-values for annotations.
        key (str): The column name used to group data points in the plot.
        title (str): The title of the plot.
        file_name (str): Base file name for saving the plot images.
    """
    sample_counts = df["Sample"].value_counts()
    sorted_samples = sample_counts.sort_values(ascending=True).index.tolist()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    params = dict(
        x="Sample",
        y="TEST_DiceMetric",
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

    k1, k2 = df[key].unique()
    pairs = [[(s, k1), (s, k2)] for s in pvalues.index]

    annotator = Annotator(ax, pairs, **params)
    annotator.configure(color="blue", line_width=1, verbose=False)
    annotator.set_pvalues_and_annotate(pvalues.values)

    current_labels = ax.get_xticklabels()
    new_labels = [
        f"{label.get_text()}\n(n={sample_counts[label.get_text()] // 2})"
        for label in current_labels
    ]

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

def process_single_experiment(exp_type: str, exp_group: str, exp_names: Dict[str, str], exp_dir: Path, result_dir: Path):
    df = merge_experiments(exp_dir, exp_names, key="Model")
    p_values = {}
    for model in exp_names.values():
        if model == "CryoViT":
            continue
        test_fn = functools.partial(significance_test, model_A="CryoViT", model_B=model, key="Model", test_fn="wilcoxon")
        m_name = model.replace(" ", "").lower()
        p_values[model] = compute_stats(df, group_keys=["Sample", "Model"], file_name=result_dir / f"{exp_group}_{m_name}_{exp_type}_stats.csv", test_fn=test_fn)
    plot_df(df, p_values, "Model", f"{exp_type.capitalize()} {exp_group.upper()} Comparison", result_dir / f"{exp_group}_comparison")