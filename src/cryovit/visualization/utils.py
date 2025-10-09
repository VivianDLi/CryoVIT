from collections.abc import Callable
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def merge_experiments(
    exp_dir: Path,
    exp_names: dict[str, list[str]],
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Merge multiple experiment results into a single DataFrame, adding additional labels for each experiment.

    Args:
        exp_dir (Path): The directory containing experiment results (.csvs).
        exp_names (dict[str, list[str]]): A dictionary mapping experiment names to labels.
        keys (list[str]): The column names to assign to the experiment labels in the merged DataFrame. If None, defaults to ["model"].

    Raises:
        ValueError: If the specified experiment directory does not exist.

    Returns:
        pd.DataFrame: A DataFrame containing merged experiment data.
    """

    if not exp_dir.exists():
        raise ValueError(f"The directory {exp_dir} does not exist")

    if keys is None:
        keys = ["model"]
    results = []

    for exp_name, value in exp_names.items():
        result_csv = exp_dir / f"{exp_name}.csv"
        df = pd.read_csv(result_csv)
        for key, val in zip(keys, value, strict=True):
            df[key] = val
        results.append(df)

    return pd.concat(results, axis=0, ignore_index=True)


def significance_test(
    df,
    model_A: str,
    model_B: str,
    key: str = "model",
    test_fn: str = "wilcoxon",
) -> float:
    """Perform a significance test with a specific function between two models on a grouped DataFrame.

    Args:
        group (pd.DataFrame): Grouped DataFrame to perform the test on.
        model_A (str): The name of the first model.
        model_B (str): The name of the second model.
        key (str): Column name to get model names.
        test_fn (str): The statistical test function to use (e.g., wilcoxon, ttest_rel).

    Raises:
        ValueError: If an unknown test function is provided.

    Returns:
        float: The p-value from the statistical test.
    """

    score_A = df[df[key] == model_A].sort_values("tomo_name").dice_metric
    score_B = df[df[key] == model_B].sort_values("tomo_name").dice_metric
    assert len(score_A) == len(
        score_B
    ), f"The two models must have the same number of samples for comparison. {model_A} has {len(score_A)}, {model_B} has {len(score_B)}"

    if test_fn == "wilcoxon":
        _, pvalue = wilcoxon(
            score_A, score_B, method="exact", alternative="two-sided"
        )
    elif test_fn == "ttest_rel":
        _, pvalue = ttest_rel(score_A, score_B, alternative="two-sided")
    else:
        raise ValueError(f"Unknown test function: {test_fn}")

    return pvalue  # type: ignore


def compute_stats(
    df: pd.DataFrame, group_keys: list[str], file_name: str, test_fn: Callable
) -> pd.Series:
    """Compute statistical summaries for the DataFrame and save them to a file.

    Args:
        df (pd.DataFrame): The DataFrame to compute statistics on.
        group_keys (list[str]): The column names used to group data for statistics. The first element is used for the p-value calculation.
        file_name (str): The file path to save the statistics.
        test_fn (Callable): A function to compute p-values between groups.

    Returns:
        pd.Series: A Series containing p-values for statistical tests.
    """

    grouped = df.groupby(group_keys, sort=False)["dice_metric"].agg(
        mean="mean",
        std="std",
        median="median",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
    )

    transforms = {
        "Median Dice Score": lambda row: f"{row['median']:.2f}",
        "Mean Dice Score ± Std": lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}",
        "Dice Score Quartiles (Q1 - Q3)": lambda row: f"{row['Q1']:.2f} - {row['Q3']:.2f}",
    }

    values = {
        col: grouped.apply(func, axis=1) for col, func in transforms.items()
    }
    stats_df = pd.DataFrame.from_dict(values).unstack(level=-1)

    pvalues = df.groupby(group_keys[0]).apply(test_fn, include_groups=False)
    pvalues_formatted = pvalues.apply(lambda x: f"{x:.2e}")
    stats_df["p-value"] = pvalues_formatted[stats_df.index]

    if (
        group_keys[0] != "split_id"
    ):  # Don't count n for splits grouping (i.e., fractional sample)
        sample_counts = df[group_keys[0]].value_counts(ascending=True)
        stats_df = stats_df.loc[sample_counts.index]
    stats_df.reset_index(names=group_keys[0]).to_csv(file_name, index=False)  # type: ignore

    return pvalues
