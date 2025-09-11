import argparse
from pathlib import Path

from cryovit._logging_config import setup_logging
from cryovit.run.infer_model import run_inference
from cryovit.utils import load_files_from_path

if __name__ == "__main__":
    setup_logging("INFO")

    parser = argparse.ArgumentParser(
        description="Run model inference given a data folder (or text file specifying files)."
    )
    parser.add_argument(
        "data",
        type=str,
        help="Directory or .txt file of data tomograms",
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
        help="Path to the directory to save the inference results. Defaults to the current directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        required=False,
        help="Threshold for binary segmentation (default: 0.5)",
    )

    args = parser.parse_args()
    data = Path(args.data)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else Path.cwd()
    result_dir.mkdir(parents=True, exist_ok=True)

    ## Sanity Checking
    assert data.exists(), "Data path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert (
        model_path.suffix == ".model"
    ), "Model path must point to a .model file."

    data_paths = load_files_from_path(data)

    run_inference(data_paths, model_path, result_dir, threshold=args.threshold)
