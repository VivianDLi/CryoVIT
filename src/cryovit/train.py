import argparse
from pathlib import Path

from cryovit._logging_config import setup_logging
from cryovit.run.train_model import run_training
from cryovit.utils import (
    id_generator,
    load_files_from_path,
)

if __name__ == "__main__":
    setup_logging("INFO")

    parser = argparse.ArgumentParser(
        description="Run model training given a training and optional validation folder (or text file specifying files)."
    )
    parser.add_argument(
        "train_data",
        type=str,
        help="Directory or .txt file of training tomograms",
    )
    parser.add_argument(
        "train_labels",
        type=str,
        help="Directory or .txt file of training labels",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="list of label names in ascending-value order..",
    )
    parser.add_argument(
        "--label_key", type=str, required=True, help="Label key to train on."
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=False,
        help="Directory or .txt file of validation tomograms",
    )
    parser.add_argument(
        "--val_labels",
        type=str,
        required=False,
        help="Directory or .txt file of validation labels",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the training results: a .zip file containing model weights and metadata. Defaults to the current directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Name to identify the model. If not provided, a random name will be generated.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="cryovit",
        choices=["cryovit", "unet3d", "sam2", "medsam"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        required=False,
        help="Number of training epochs. Default is 50.",
    )
    parser.add_argument(
        "--log_training",
        "-l",
        action="store_true",
        help="If set, will log training metrics to TensorBoard.",
    )

    args = parser.parse_args()
    train_data = Path(args.train_data)
    train_labels = Path(args.train_labels)
    val_data = Path(args.val_data) if args.val_data else None
    val_labels = Path(args.val_labels) if args.val_labels else None
    model_type = args.model
    model_name = args.name or model_type + "_" + id_generator()
    result_path = Path(args.result_path) if args.result_path else Path.cwd()

    ## Sanity Checking
    assert train_data.exists(), "Training data path does not exist."
    assert train_labels.exists(), "Training labels path does not exist."
    if val_data is not None:
        assert val_data.exists(), "Validation data path does not exist."
        assert (
            val_labels is not None and val_labels.exists()
        ), "Validation labels path does not exist."

    train_paths = load_files_from_path(train_data)
    train_label_paths = load_files_from_path(train_labels)
    val_paths = (
        load_files_from_path(val_data) if val_data is not None else None
    )
    val_label_paths = (
        load_files_from_path(val_labels) if val_labels is not None else None
    )

    run_training(
        train_paths,
        train_label_paths,
        args.labels,
        model_type,
        model_name,
        args.label_key,
        result_path,
        val_paths,
        val_label_paths,
        num_epochs=args.num_epochs,
        log_training=args.log_training,
    )
