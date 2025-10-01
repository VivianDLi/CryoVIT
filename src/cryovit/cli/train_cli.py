from pathlib import Path
from typing import Annotated

from typer import Argument, Option

from cryovit.types import ModelType

from .cli import cli


@cli.command(name="train", no_args_is_help=True)
def train(
    train_data: Annotated[
        str,
        Argument(
            help="Path to the folder or .txt file containing the training tomograms.",
        ),
    ],
    train_labels: Annotated[
        str,
        Argument(
            help="Path to the folder or .txt file containing the training labels.",
        ),
    ],
    labels: Annotated[
        list[str],
        Option(help="List of available label names in ascending-value order."),
    ],
    label_key: Annotated[
        str,
        Argument(
            help="Label key to train on. Must be one of the provided labels.",
        ),
    ],
    validation_data: Annotated[
        str | None,
        Option(
            "--validation-data",
            "--val-data",
            help="Path to the folder or .txt file containing the validation tomograms.",
            show_default=False,
        ),
    ] = None,
    validation_labels: Annotated[
        str | None,
        Option(
            "--validation-labels",
            "--val-labels",
            help="Path to the folder or .txt file containing the validation labels.",
            show_default=False,
        ),
    ] = None,
    name: Annotated[
        str | None,
        Option(
            help="Name to identify the model. If not provided, a random name will be generated.",
            show_default=False,
            rich_help_panel="Model Customization",
        ),
    ] = None,
    model: Annotated[
        ModelType,
        Option(
            help="The type of model to train.",
            rich_help_panel="Model Customization",
        ),
    ] = ModelType.CRYOVIT,
    result_folder: Annotated[
        str | None,
        Option(
            help="Path to the folder where the trained model will be saved.",
            show_default="the current working directory",
            rich_help_panel="Model Customization",
        ),
    ] = None,
    ckpt: Annotated[
        str | None,
        Option(
            help="Path to a pre-trained .model file, or .ckpt/.pt weights to fine-tune a model from.",
            rich_help_panel="Model Customization",
        ),
    ] = None,
    num_epochs: Annotated[
        int,
        Option(
            min=1,
            help="Number of training epochs.",
            rich_help_panel="Training Customization",
        ),
    ] = 50,
    log_training: Annotated[
        bool,
        Option(
            "--log-training",
            "-l",
            help="Additionally log training metrics to TensorBoard?",
            rich_help_panel="Training Customization",
        ),
    ] = False,
):
    """Train a segmentation model on given training data and labels.

    Example
    -------
    cryovit train <path-to-training-data> <path-to-training-labels> --labels <first-label> <second-label> ... --label-key <label-to-train-on>
    """
    from cryovit._logging_config import setup_logging
    from cryovit.run.train_model import run_training
    from cryovit.utils import (
        id_generator,
        load_files_from_path,
    )

    setup_logging("INFO")

    ## Convert Arguments
    train_path = Path(train_data)
    label_path = Path(train_labels)
    ckpt_path = Path(ckpt) if ckpt is not None else None
    val_path = Path(validation_data) if validation_data else None
    val_label_path = Path(validation_labels) if validation_labels else None
    result_path = Path(result_folder) if result_folder else Path.cwd()
    model_name = name or model.value + "_" + id_generator()

    ## Sanity Checking
    assert train_path.exists(), "Training data path does not exist."
    assert label_path.exists(), "Training labels path does not exist."
    if val_path is not None:
        assert val_path.exists(), "Validation data path does not exist."
        assert (
            val_label_path is not None and val_label_path.exists()
        ), "Validation data provided but validation labels path does not exist."
    if ckpt_path is not None:
        assert ckpt_path.exists(), "Checkpoint path does not exist."

    train_files = load_files_from_path(train_path)
    train_label_files = load_files_from_path(label_path)
    val_files = (
        load_files_from_path(val_path) if val_path is not None else None
    )
    val_label_files = (
        load_files_from_path(val_label_path)
        if val_label_path is not None
        else None
    )
    result_path.mkdir(parents=True, exist_ok=True)

    run_training(
        train_files,
        train_label_files,
        labels,
        model,
        model_name,
        label_key,
        result_path,
        val_files,
        val_label_files,
        num_epochs=num_epochs,
        log_training=log_training,
        ckpt_path=ckpt_path,
    )
