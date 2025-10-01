"""Make plots comparing single sample performance."""

import functools
from pathlib import Path

import pandas as pd

from cryovit.types import Sample
from cryovit.visualization.utils import (
    compute_stats,
    merge_experiments,
    significance_test,
)


def _plot_df(
    df: pd.DataFrame,
    pvalues: pd.Series,
    key: str,
    title: str,
    file_name: str,
    is_sample: bool,
):
    # import here to avoid unnecessary dependencies if function not used
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

    fig = plt.figure(figsize=(20, 6))
    ax = plt.gca()
    if not is_sample:
        df["split_id"] = df["split_id"].astype(int)
    label_key = "sample" if is_sample else "split_id"
    label_counts = df[label_key].value_counts()
    num_models = df[key].nunique()
    sorted_labels = label_counts.sort_values(ascending=True).index.tolist()

    params = {
        "x": label_key,
        "y": "dice_metric",
        "hue": key,
        "data": df,
        "order": sorted_labels,
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
    annotator.set_pvalues_and_annotate(pvalues.values)

    current_labels = ax.get_xticklabels()
    if is_sample:
        new_labels = [
            f"{Sample(label.get_text()).value}\n(n={label_counts[label.get_text()] // num_models})"
            for label in current_labels
        ]
    else:
        new_labels = [f"{label.get_text()}0%" for label in current_labels]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(new_labels, ha="center")

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.suptitle(title)
    fig.supxlabel(
        "Sample Name (Count)"
        if is_sample
        else f"Fraction of Training Data\n(n={label_counts[10] // num_models})"
    )
    fig.supylabel("Dice Score")

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="lower center", shadow=True)

    plt.tight_layout(rect=(0.01, 0.01, 1.0, 1.0))
    plt.savefig(f"{file_name}.svg")
    plt.savefig(f"{file_name}.png", dpi=300)


def process_sparse_experiment(
    exp_type: str,
    exp_group: str,
    exp_names: dict[str, list[str]],
    exp_dir: Path,
    result_dir: Path,
):
    """Plot sparse sample experiment results with box and strip plots including annotations for statistical tests.

    Args:
        exp_type (str): Type of experiment, e.g. "sparse"
        exp_group (str): Group of experiments, e.g. "single", "fractional"
        exp_names (dict[str, list[str]]): Dictionary mapping experiment names to models
        exp_dir (Path): Directory containing experiment results
        result_dir (Path): Directory to save results
    """

    result_dir.mkdir(parents=True, exist_ok=True)
    df = merge_experiments(exp_dir, exp_names, keys=["label_type"])
    if exp_group == "single":
        df = df[df["split_id"] == 10]  # Use data from 100% of training data
    test_type = "wilcoxon" if exp_group == "single" else "ttest_rel"
    test_fn = functools.partial(
        significance_test,
        model_A=("CryoViT with Sparse Labels"),
        model_B=("CryoViT with Dense Labels"),
        key="label_type",
        test_fn=test_type,
    )
    group_key = "sample" if exp_group == "single" else "split_id"
    p_values = compute_stats(
        df,
        group_keys=[group_key, "label_type"],
        file_name=str(result_dir / f"{exp_group}_{exp_type}_stats.csv"),
        test_fn=test_fn,
    )
    title = f"CryoViT: Sparse vs. Dense Label Comparison on {'Individual' if exp_group == 'single' else 'Fractional'} Samples"
    _plot_df(
        df,
        p_values,
        "label_type",
        title,
        str(result_dir / f"{exp_group}_{exp_type}_comparison"),
        is_sample=(exp_group == "single"),
    )
