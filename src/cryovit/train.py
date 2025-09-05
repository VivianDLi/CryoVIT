import argparse
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytorch_lightning.loggers import TensorBoardLogger

from cryovit.models.sam2 import create_sam_model_from_weights
from cryovit.utils import id_generator, load_files_from_path, save_model


def run_training(
    train_data: list[Path],
    train_labels: list[Path],
    labels: list[str],
    model_type: str,
    model_name: str,
    label_key: str,
    result_dir: Path,
    val_data: list[Path] | None = None,
    val_labels: list[Path] | None = None,
    num_epochs: int = 50,
    log_training: bool = False,
) -> None:
    ## Setup hydra config
    config_path = Path(__file__).parent / "configs"
    with initialize(
        version_base="1.2",
        config_path=str(config_path),
        job_name="cryovit_train",
    ):
        cfg = compose(
            config_name="train_model",
            overrides=[
                f"name={model_name}",
                f"label_key={label_key}",
                f"model={model_type}",
                "datamodule=file",
                f"trainer.max_epochs={num_epochs}",
            ],
        )
    cfg.paths.model_dir = Path(__file__).parent / "foundation_models"
    save_model_path = result_dir / f"{model_name}.model"

    # Check input key
    if cfg.model.input_key != "dino_features":
        cfg.model.input_key = None  # find available data instead

    ## Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    datamodule = instantiate(cfg.datamodule, _convert_="all")(
        data_paths=train_data,
        data_labels=train_labels,
        labels=labels,
        val_paths=val_data,
        val_labels=val_labels,
        dataloader_fn=dataloader_fn,
        dataset_fn=dataset_fn,
    )
    print("Setup dataset.")

    ## Setup training
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    logger = []
    if log_training:
        # tensorboard logger to avoid wandb account issues
        logger.append(TensorBoardLogger(save_dir=result_dir, name=model_name))
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    if cfg.model._target_ == "cryovit.models.SAM2":
        # Load SAM2 pre-trained models
        model = create_sam_model_from_weights(
            cfg.model, cfg.paths.model_dir / cfg.paths.sam_name
        )
    else:
        model = instantiate(cfg.model)
    print("Loaded model.")

    # Base SAM2 only supports image encoder compilation
    if cfg.model._target_ == "cryovit.models.SAM2":
        print("Compiling image encoder for SAM2 model.")
        try:
            model.compile()
        except Exception as e:  # noqa: BLE001
            print(f"Unable to compile image encoder for SAM2: {e}")
    else:
        print("Compiling model forward pass.")
        try:
            model.forward = torch.compile(model.forward)
        except Exception as e:  # noqa: BLE001
            print(f"Unable to compile forward pass: {e}")

    print("Starting training.")
    trainer.fit(model, datamodule=datamodule)

    # Save model
    print("Saving model.")
    save_model(model_name, label_key, model, cfg.model, save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model training given a training and optional validation folder (or text file specifying files)."
    )
    parser.add_argument(
        "train_data",
        type=str,
        required=True,
        help="Directory or .txt file of training tomograms",
    )
    parser.add_argument(
        "train_labels",
        type=str,
        required=True,
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
        "--model",
        type=str,
        required=False,
        default="cryovit",
        choices=["cryovit", "unet3d", "sam2", "medsam"],
        help="Type of model to train",
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
        "--label_key", type=str, required=True, help="Label key to train on."
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
    model_name = args.name or model_type + id_generator()
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
