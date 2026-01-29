"""Make plots comparing multi sample performance."""

import functools
from pathlib import Path

import pandas as pd

from cryovit.types import Sample
from cryovit.visualization.utils import (
    compute_stats,
    merge_experiments,
    significance_test,
)

group_names = {
    "hd": "Diseased",
    "healthy": "Healthy",
    "old": "Aged",
    "young": "Young",
    "neuron": "Neurons",
    "fibro_cancer": "Fibroblast/Cancer Cells",
}


def _plot_df(
    df: pd.DataFrame,
    pvalues: pd.Series,
    key: str,
    title: str,
    ax,
):
    import matplotlib
    import seaborn as sns
    from statannotations.Annotator import Annotator

    matplotlib.use("Agg")
    colors = sns.color_palette("deep")[:3]
    sns.set_theme(style="darkgrid", font="Open Sans")

    hue_palette = {
        "3D U-Net": colors[0],
        "CryoViT": colors[1],
        "SAM2": colors[2],
    }

    sample_counts = df["sample"].value_counts()
    num_models = df[key].nunique()
    sorted_samples = sample_counts.sort_values(ascending=True).index.tolist()

    params = {
        "x": "sample",
        "y": "dice_metric",
        "hue": key,
        "data": df,
        "order": sorted_samples,
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

    k1, k2, k3 = df[key].unique()
    pairs = (
        [[(s, k1), (s, k2)] for s in pvalues.index]
        + [[(s, k1), (s, k3)] for s in pvalues.index]
        + [[(s, k2), (s, k3)] for s in pvalues.index]
    )

    annotator = Annotator(ax, pairs, **params)
    annotator.configure(color="blue", line_width=1, verbose=False)
    annotator.set_pvalues_and_annotate(pvalues.values)

    current_labels = ax.get_xticklabels()
    new_labels = [
        f"{Sample[label.get_text()].value}\n(n={sample_counts[label.get_text()] // num_models})"
        for label in current_labels
    ]

    ax.set_ylim(-0.05, 1.15)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(range(len(new_labels)))
    ax.set_xticklabels(new_labels, ha="center")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="lower right", shadow=True)


def process_multi_experiment(
    exp_type: str,
    exp_group: tuple[str, str],
    exp_names: dict[str, list[str]],
    exp_dir: Path,
    result_dir: Path,
):
    """Plot domain-shift results with box and strip plots including annotations for statistical tests.

    Args:
        exp_type (str): Type of experiment, i.e., "multi"
        exp_group (tuple[str, str]): Tuple containing the names of the two experiment groups to compare
        exp_names (dict[str, list[str]]): Dictionary mapping experiment group names to a model and comparison direction, e.g., "forward" or "backward"
        exp_dir (Path): Directory containing the experiment results
        result_dir (Path): Directory to save the results
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    sns.set_theme(style="darkgrid", font="Open Sans")

    result_dir.mkdir(parents=True, exist_ok=True)
    df = merge_experiments(exp_dir, exp_names, keys=["model", "type"])
    forward_df = df[df["type"] == "forward"]
    backward_df = df[df["type"] == "backward"]

    s1_count = forward_df["sample"].nunique()
    s2_count = backward_df["sample"].nunique()

    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(
        1, 2, width_ratios=[s1_count, s2_count]
    )  # Set width ratios based on unique sample counts

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Plot forward comparison (s1 vs. s2)
    total_ps = None
    for m1, m2 in [
        ("CryoViT", "3D U-Net"),
        ("CryoViT", "SAM2"),
        ("3D U-Net", "SAM2"),
    ]:
        test_fn = functools.partial(
            significance_test,
            model_A=m1,
            model_B=m2,
            key="model",
            test_fn="wilcoxon",
        )
        p_values = compute_stats(
            forward_df,
            group_keys=["sample", "model"],
            file_name=str(
                result_dir
                / f"{'_'.join(list(exp_group))}_{exp_type}_{m1}_{m2}_stats.csv"
            ),
            test_fn=test_fn,
        )
        total_ps = (
            p_values if total_ps is None else pd.concat([total_ps, p_values])
        )
    assert total_ps is not None

    title = f"{group_names[exp_group[0]]} to {group_names[exp_group[1]]} Shift"
    _plot_df(forward_df, total_ps, "model", title, ax1)

    # Plot backward comparison (s2 vs. s1)
    total_ps = None
    for m1, m2 in [
        ("CryoViT", "3D U-Net"),
        ("CryoViT", "SAM2"),
        ("3D U-Net", "SAM2"),
    ]:
        test_fn = functools.partial(
            significance_test,
            model_A=m1,
            model_B=m2,
            key="model",
            test_fn="wilcoxon",
        )
        p_values = compute_stats(
            backward_df,
            group_keys=["sample", "model"],
            file_name=str(
                result_dir
                / f"{'_'.join(list(reversed(exp_group)))}_{exp_type}_{m1}_{m2}_stats.csv"
            ),
            test_fn=test_fn,
        )
        total_ps = (
            p_values if total_ps is None else pd.concat([total_ps, p_values])
        )
    assert total_ps is not None

    title = f"{group_names[exp_group[1]]} to {group_names[exp_group[0]]} Shift"
    _plot_df(backward_df, total_ps, "model", title, ax2)

    # Adjust layout and save the figure
    if (
        "Cells" in group_names[exp_group[0]]
        or "Cells" in group_names[exp_group[1]]
    ):
        domain = "Cell Type"
    else:
        domain = "Diseased/Healthy"
    fig.suptitle(f"Model Comparison Across {domain} Domain Shifts")
    fig.supxlabel("Sample Name (Count)")
    fig.supylabel("Dice Score")

    plt.tight_layout(rect=(0.01, 0.01, 1.0, 1.0))
    plt.savefig(result_dir / f"{exp_group[0]}_{exp_group[1]}_domain_shift.svg")
    plt.savefig(
        result_dir / f"{exp_group[0]}_{exp_group[1]}_domain_shift.png", dpi=300
    )
