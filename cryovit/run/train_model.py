"""Script for setting up and training CryoVIT models based on configuration files."""

import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything

from cryovit.config import BaseExperimentConfig
from cryovit.models import create_sam_model_from_weights

torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


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

    new_exp_dir = cfg.paths.exp_dir / cfg.name / sample
    if cfg.datamodule.split_id is not None:
        new_exp_dir = new_exp_dir / f"split_{cfg.datamodule.split_id}"

    new_exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.exp_dir = new_exp_dir

    # Setup WandB Logger
    for name, logger in cfg.logger.items():
        if name == "wandb":
            logger.name = (
                f"{sample}_{cfg.datamodule.split_id}"
                if cfg.datamodule.split_id is not None
                else sample
            )

    return cfg


def run_trainer(cfg: BaseExperimentConfig) -> None:
    """Sets up and runs the training process using the specified configuration.

    Args:
        cfg (TrainModelConfig): Configuration object containing all settings for the training process.
    """
    seed_everything(cfg.random_seed, workers=True)

    # Setup experiment directories
    cfg = setup_exp_dir(cfg)
    ckpt_path = (
        cfg.paths.exp_dir / "last.ckpt"
        if cfg.ckpt_path is None
        else cfg.ckpt_path
    )
    weights_path = cfg.paths.exp_dir / "weights.pt"

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

    # Setup training
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    logger = [instantiate(lg_cfg) for lg_cfg in cfg.logger.values()]
    trainer: Trainer = instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    logging.info("Setup trainer.")
    if cfg.model._target_ == "cryovit.models.SAM2":
        # Load SAM2 pre-trained models
        model = create_sam_model_from_weights(
            cfg.model, cfg.paths.model_dir / cfg.paths.sam_name
        )
    else:
        model = instantiate(cfg.model)
    logging.info("Setup model.")

    # Log hyperparameters
    if trainer.logger:
        hparams = {
            "datamodule_type": HydraConfig.get().runtime.choices["datamodule"],
            "model_name": cfg.model.name,
            "label_key": cfg.label_key,
            "experiment": cfg.name,
            "split_id": cfg.datamodule.split_id,
            "sample": (
                "_".join(sorted(cfg.datamodule.sample))
                if isinstance(cfg.datamodule.sample, Iterable)
                else cfg.datamodule.sample
            ),
            "cfg": cfg,
            "model": model,
            "model/params/total": sum(p.numel() for p in model.parameters()),
            "model/params/trainable": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "model/params/non_trainable": sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            ),
            "datamodule": datamodule,
            "trainer": trainer,
            "resume_ckpt": cfg.resume_ckpt,
            "ckpt_path": cfg.ckpt_path,
            "seed": cfg.random_seed,
        }
        for logger in trainer.loggers:
            logger.log_hyperparams(hparams)

    # Base SAM2 only supports image encoder compilation
    if cfg.model._target_ == "cryovit.models.SAM2":
        logging.info("Compiling image encoder for SAM2 model.")
        try:
            model.compile()
        except Exception as e:  # noqa: BLE001
            logging.warning("Unable to compile image encoder for SAM2: %s", e)
    else:
        logging.info("Compiling model forward pass.")
        try:
            model.forward = torch.compile(model.forward)
        except Exception as e:  # noqa: BLE001
            logging.warning("Unable to compile forward pass: %s", e)

    logging.info("Starting training.")
    if cfg.resume_ckpt and ckpt_path.exists():
        logging.info("Resuming training from checkpoint: %s", ckpt_path)
        trainer.fit(model, datamodule=datamodule, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, datamodule=datamodule)

    # Save model
    logging.info("Saving model.")
    torch.save(model.state_dict(), weights_path)
