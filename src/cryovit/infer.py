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

    args = parser.parse_args()
    data = Path(args.data)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else Path.cwd()

    ## Sanity Checking
    assert data.exists(), "Data path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert (
        model_path.suffix == ".model"
    ), "Model path must point to a .model file."
    assert (
        result_dir.exists() and result_dir.is_dir()
    ), "Result directory either does not exist or isn't a directory."

    data_paths = load_files_from_path(data)

    run_inference(data_paths, model_path, result_dir)
