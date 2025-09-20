"""Make plots comparing single sample performance."""

import functools
from pathlib import Path

import pandas as pd

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
):
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
    }

    label_counts = df["label"].value_counts()
    num_models = df[key].nunique()
    n_samples = df["label"].nunique()
    sorted_labels = label_counts.sort_values(ascending=True).index.tolist()
    fig = plt.figure(figsize=(12 if n_samples > 6 else 6, 6))
    ax = plt.gca()

    params = {
        "x": "label",
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
    full_labels = {
        "mito": "Mitochondria",
        "cristae": "Cristae",
        "microtubule": "Microtubules",
        "granule": "Granules",
        "bacteria": "Bacteria",
    }
    new_labels = [
        f"{full_labels[label.get_text()]}\n(n={label_counts[label.get_text()] // num_models})"
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


def process_multi_label_experiment(
    exp_type: str,
    exp_names: dict[str, list[str]],
    exp_dir: Path,
    result_dir: Path,
):
    """Plot DataFrame results with box and strip plots including annotations for statistical tests.

    Args:
        exp_type (str): Type of experiment, i.e., "multi_label"
        exp_names (dict[str, list[str]]): Dictionary mapping experiment names to model used and label
        exp_dir (Path): Directory containing the experiment results
        result_dir (Path): Directory to save the results
    """

    result_dir.mkdir(parents=True, exist_ok=True)
    df = merge_experiments(exp_dir, exp_names, keys=["model", "label"])
    test_fn = functools.partial(
        significance_test,
        model_A="CryoViT",
        model_B="3D U-Net",
        key="model",
        test_fn="wilcoxon",
    )
    p_values = compute_stats(
        df,
        group_keys=["label", "model"],
        file_name=str(result_dir / f"{exp_type}_stats.csv"),
        test_fn=test_fn,
    )

    _plot_df(
        df,
        p_values,
        "model",
        "Model Comparison on All Samples for each Label",
        str(result_dir / f"{exp_type}_comparison"),
    )
