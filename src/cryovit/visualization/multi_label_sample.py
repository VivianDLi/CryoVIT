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
    pvalues: dict[tuple[str, str], pd.Series],
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
        "SAM2": colors[2],
    }

    label_counts = df["label"].value_counts()
    num_models = df[key].nunique()
    sorted_labels = label_counts.sort_values(ascending=True).index.tolist()
    fig = plt.figure(figsize=(20, 6))
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

    pairs = []
    values = []
    for (model_A, model_B), ps in pvalues.items():
        pairs += [[(s, model_A), (s, model_B)] for s in ps.index]
        values += ps.values.tolist()
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
        "mito_membrane": "Mitochondrial Membrane",
        "er_membrane": "ER Membrane",
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

    plt.tight_layout(rect=(0.01, 0.01, 1.0, 1.0))
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
    df = df[df["split_id"] == 10]  # Use data from 100% of training data
    p_values = {}
    for pairing in [
        ("CryoViT", "3D U-Net"),
        ("CryoViT", "SAM2"),
        ("3D U-Net", "SAM2"),
    ]:
        model_A, model_B = pairing
        test_fn = functools.partial(
            significance_test,
            model_A=model_A,
            model_B=model_B,
            key="model",
            test_fn="ttest_rel",
        )
        for label in df["label"].unique():
            sub_df = df[df["label"] == label]
            print(
                label,
                len(sub_df[sub_df["model"] == "CryoViT"]),
                len(sub_df[sub_df["model"] == "3D U-Net"]),
                len(sub_df[sub_df["model"] == "SAM2"]),
            )
        p_values[pairing] = compute_stats(
            df,
            group_keys=["label", "model"],
            file_name=str(result_dir / f"{exp_type}_stats.csv"),
            test_fn=test_fn,
        )

    _plot_df(
        df,
        p_values,
        "model",
        "Model Comparison on All Samples",
        str(result_dir / f"{exp_type}_comparison"),
    )
