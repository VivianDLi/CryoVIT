"""Make plots comparing fractional LOO sample performance."""

import functools
from pathlib import Path

import pandas as pd

from cryovit.visualization.utils import (
    compute_stats,
    merge_experiments,
    significance_test,
)


def plot_df(
    df: pd.DataFrame,
    pvalues: pd.Series,
    key: str,
    title: str,
    file_name: str,
):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        pvalues (pd.Series): Series containing p-values for annotations.
        key (str): The column name used to group data points in the plot.
        title (str): The title of the plot.
        file_name (str): Base file name for saving the plot images.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statannotations.Annotator import Annotator

    matplotlib.use("Agg")
    colors = sns.color_palette("deep")[:3]
    sns.set_theme(style="darkgrid", font="Open Sans")

    hue_palette = {
        "3D U-Net": colors[0],
        "CryoViT": colors[1],
        "CryoViT with Sparse Labels": colors[1],
        "CryoViT with Dense Labels": colors[2],
    }

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    params = {
        "x": "split_id",
        "y": "dice_metric",
        "hue": key,
        "data": df,
    }

    sns.boxplot(
        showfliers=False,
        palette=hue_palette,
        linewidth=1,
        medianprops={"linewidth": 2, "color": "firebrick"},
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
    annotator.set_pvalues_and_annotate(pvalues[k2].values)

    current_labels = ax.get_xticklabels()
    new_labels = [f"{label.get_text()}0%" for label in current_labels]
    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, ha="center")

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.suptitle(title)
    fig.supxlabel("Fraction of Training Data")
    fig.supylabel("Dice Score")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="lower center", shadow=True)

    plt.tight_layout()
    plt.savefig(f"{file_name}.svg")
    plt.savefig(f"{file_name}.png", dpi=300)


def process_fractional_experiment(
    exp_type: str,
    label: str,
    exp_names: dict[str, list[str]],
    exp_dir: Path,
    result_dir: Path,
):
    key = "model" if exp_type != "sparse" else "label_type"
    df = merge_experiments(exp_dir, exp_names, keys=[key])
    test_fn = functools.partial(
        significance_test,
        model_A=(
            "CryoViT" if exp_type != "sparse" else "CryoViT with Sparse Labels"
        ),
        model_B=(
            "3D U-Net" if exp_type != "sparse" else "CryoViT with Dense Labels"
        ),
        key=key,
        test_fn="ttest_rel",
    )
    p_values = compute_stats(
        df,
        group_keys=["split_id", key],
        file_name=str(result_dir / f"{label}_{exp_type}_stats.csv"),
        test_fn=test_fn,
    )
    if exp_type != "sparse":
        plot_df(
            df,
            p_values,
            key,
            f"Model Comparison on All {label.upper()} Samples",
            str(result_dir / f"{label}_{exp_type}_comparison"),
        )
    else:
        plot_df(
            df,
            p_values,
            key,
            "CryoViT: Sparse vs. Dense Labels Comparison on All Samples",
            str(result_dir / "fractional_sparse_vs_dense_comparison"),
        )
