import argparse
from pathlib import Path

from cryovit._logging_config import setup_logging
from cryovit.run.dino_features import DEFAULT_WINDOW_SIZE, run_dino
from cryovit.utils import load_files_from_path

if __name__ == "__main__":
    setup_logging("INFO")

    parser = argparse.ArgumentParser(
        description="Calculate DINOv2 features for a given training dataset folder (or text file specifying files)."
    )
    parser.add_argument(
        "train_data",
        type=str,
        help="Directory or .txt file of training tomograms",
    )
    parser.add_argument(
        "result_dir",
        type=str,
        help="Directory to save the DINO features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        required=False,
        help="Batch size for DINO feature extraction (default: 64)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        required=False,
        help="Window size for DINO feature extraction (default: 630)",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Whether to save PCA visualization of DINO features",
    )

    args = parser.parse_args()
    train_data = Path(args.train_data)
    result_dir = Path(args.result_dir)
    batch_size = args.batch_size
    window_size = (
        args.window_size
        if args.window_size is not None
        else DEFAULT_WINDOW_SIZE
    )

    ## Sanity Checking
    assert train_data.exists(), "Training data path does not exist."
    result_dir.mkdir(parents=True, exist_ok=True)

    train_files = load_files_from_path(train_data)
    run_dino(
        train_files,
        result_dir,
        batch_size,
        window_size=window_size,
        visualize=args.visualize,
    )
