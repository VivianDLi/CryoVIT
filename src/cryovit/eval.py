import argparse
from pathlib import Path

from cryovit._logging_config import setup_logging
from cryovit.run.eval_model import run_evaluation
from cryovit.utils import load_files_from_path

if __name__ == "__main__":
    setup_logging("INFO")

    parser = argparse.ArgumentParser(
        description="Run model evaluation given a test folder (or text file specifying files)."
    )
    parser.add_argument(
        "test_data",
        type=str,
        help="Directory or .txt file of test tomograms",
    )
    parser.add_argument(
        "test_labels",
        type=str,
        help="Directory or .txt file of test labels",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="list of label names in ascending-value order..",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        required=False,
        help="Path to the directory (result_dir / results) to save the evaluation results: a .csv file containing evaluation metrics. Defaults to the current directory.",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="If set, will save visualizations of the predictions in the result_dir folder.",
    )

    args = parser.parse_args()
    test_data = Path(args.test_data)
    test_labels = Path(args.test_labels)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else Path.cwd()
    result_dir.mkdir(parents=True, exist_ok=True)

    ## Sanity Checking
    assert test_data.exists(), "Test data path does not exist."
    assert test_labels.exists(), "Test labels path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert (
        model_path.suffix == ".model"
    ), "Model path must point to a .model file."

    test_paths = load_files_from_path(test_data)
    test_label_paths = load_files_from_path(test_labels)

    run_evaluation(
        test_paths,
        test_label_paths,
        args.labels,
        model_path,
        result_dir,
        visualize=args.visualize,
    )
