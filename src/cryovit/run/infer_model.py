"""Script for evaluating CryoVIT models based on configuration files."""

import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything

from cryovit.config import BaseExperimentConfig
from cryovit.models import create_sam_model_from_weights
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


## For Experiments


def setup_exp_dir(cfg: BaseExperimentConfig) -> BaseExperimentConfig:
    # Convert paths to Paths
    cfg.paths.model_dir = Path(cfg.paths.model_dir)
    cfg.paths.data_dir = Path(cfg.paths.data_dir)
    cfg.paths.exp_dir = Path(cfg.paths.exp_dir)
    cfg.paths.results_dir = Path(cfg.paths.results_dir)

    if not isinstance(cfg.datamodule.sample, str) and isinstance(
        cfg.datamodule.sample, Iterable
    ):
        sample = "_".join(sorted(cfg.datamodule.sample))
    else:
        sample = cfg.datamodule.sample
    if not isinstance(cfg.datamodule.test_sample, str) and isinstance(
        cfg.datamodule.test_sample, Iterable
    ):
        test_sample = "_".join(sorted(cfg.datamodule.test_sample))
    else:
        test_sample = cfg.datamodule.test_sample

    new_exp_dir = cfg.paths.exp_dir / cfg.name / sample
    if cfg.datamodule.split_id is not None:
        new_exp_dir = new_exp_dir / f"split_{cfg.datamodule.split_id}"
    if test_sample is not None:
        new_exp_dir = new_exp_dir / f"{test_sample}"

    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    assert (
        new_exp_dir.exists()
    ), f"Experiment directory {new_exp_dir} does not exist. Run training first."
    cfg.paths.exp_dir = new_exp_dir

    if cfg.ckpt_path is None:
        cfg.ckpt_path = new_exp_dir / "weights.pt"
    return cfg


def run_trainer(cfg: BaseExperimentConfig) -> None:
    """Sets up and executes the model evaluation using the specified configuration.

    Args:
        cfg (BaseExperimentConfig): Configuration object containing all settings for the evaluation process.
    """
    seed_everything(cfg.random_seed, workers=True)

    # Setup experiment directories
    cfg = setup_exp_dir(cfg)
    assert (
        cfg.ckpt_path is not None and cfg.ckpt_path.exists()
    ), f"{cfg.paths.exp_dir} does not contain a checkpoint."

    # Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    split_file = cfg.paths.data_dir / cfg.paths.csv_name / cfg.paths.split_name
    datamodule = instantiate(cfg.datamodule, _convert_="all")(
        split_file=split_file,
        dataloader_fn=dataloader_fn,
        dataset_fn=dataset_fn,
    )
    logging.info("Setup dataset.")

    # Setup evaluation
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    loggers = [instantiate(lg_cfg) for lg_cfg in cfg.logger.values()]
    trainer: Trainer = instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )
    logging.info("Setup trainer.")
    if cfg.model._target_ == "cryovit.models.sam2.SAM2":
        # Load SAM2 pre-trained models
        model = create_sam_model_from_weights(
            cfg.model, cfg.paths.model_dir / cfg.paths.sam_name
        )
    else:
        model = instantiate(cfg.model)

    # Load model weights
    if cfg.ckpt_path.suffix == ".pt":
        model.load_state_dict(torch.load(cfg.ckpt_path))
    elif cfg.ckpt_path.suffix == ".ckpt":
        model = model.load_from_checkpoint(cfg.ckpt_path)
    else:
        raise ValueError(
            f"Unsupported checkpoint format: {cfg.ckpt_path.suffix}. Use .pt or .ckpt files."
        )

    logging.info("Setup model.")

    logging.info("Starting inference.")
    trainer.predict(model, datamodule=datamodule)
