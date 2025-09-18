"""Script for evaluating CryoVIT models based on configuration files."""

import logging
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from cryovit.models.callbacks import PredictionWriter
from cryovit.utils import load_model

torch.set_float32_matmul_precision("high")

## For Scripts


def run_inference(
    data_files: list[Path],
    model_path: Path,
    result_dir: Path,
    threshold: float = 0.5,
) -> list[Path]:
    """Run inference on the specified data files and saves the results.

    Args:
        data_files (list[Path]): List of paths to the input data files.
        model_path (Path): Path to the trained model file.
        result_dir (Path): Directory where the inference results will be saved.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        list[Path]: List of paths to the saved result files.
    """

    # Get model information
    model, model_type, model_name, label_key = load_model(model_path)
    assert model is not None, "Loaded model is None."
    ## Setup hydra config
    with initialize(
        version_base="1.2",
        config_path="../configs",
        job_name="infer_model",
    ):
        cfg = compose(
            config_name="infer_model",
            overrides=[
                f"name={model_name}",
                f"label_key={label_key}",
                f"model={model_type.value}",
                "datamodule=file",
            ],
        )
    cfg.paths.model_dir = Path(__file__).parent.parent / "foundation_models"
    cfg.paths.results_dir = result_dir

    # Check input key
    if cfg.model.input_key != "dino_features":
        cfg.model.input_key = None  # find available data instead

    ## Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    datamodule = instantiate(cfg.datamodule, _convert_="all")(
        data_paths=data_files,
        val_paths=None,
        dataloader_fn=dataloader_fn,
        dataset_fn=dataset_fn,
    )
    logging.info("Setup dataset.")

    ## Setup training
    pred_writer = PredictionWriter(
        results_dir=result_dir, label_key=label_key, threshold=threshold
    )
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    callbacks.append(pred_writer)
    loggers = []
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    logging.info("Starting prediction.")
    trainer.predict(model, datamodule=datamodule)

    result_paths = pred_writer.result_paths
    return result_paths
