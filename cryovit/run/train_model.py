"""Script for setting up and training CryoVIT models based on configuration files."""

import logging
from collections.abc import Iterable

import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything

from cryovit.config import BaseExperimentConfig

torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

def setup_exp_dir(cfg: BaseExperimentConfig) -> BaseExperimentConfig:
    if not isinstance(cfg.datamodule.sample, str) and isinstance(cfg.datamodule.sample, Iterable):
        sample = "_".join(sorted(cfg.datamodule.sample))
    else:
        sample = cfg.datamodule.sample
        
    new_exp_dir = cfg.paths.exp_dir / cfg.name / sample
    if cfg.datamodule.split_id is not None:
        new_exp_dir = new_exp_dir / f"split_{cfg.datamodule.split_id}"
    
    new_exp_dir.mkdir(parents=True, exist_ok=False)
    cfg.paths.exp_dir = new_exp_dir
    return cfg

def run_trainer(cfg: BaseExperimentConfig) -> None:
    """Sets up and runs the training process using the specified configuration.

    Args:
        cfg (TrainModelConfig): Configuration object containing all settings for the training process.
    """
    seed_everything(cfg.random_seed, workers=True)
    
    # Setup experiment directories
    cfg = setup_exp_dir(cfg)
    ckpt_path = cfg.paths.exp_dir / "weights.pt"

    # Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    split_file = cfg.paths.data_dir / cfg.paths.csv_name / cfg.paths.split_name
    datamodule = instantiate(cfg.datamodule)(split_file=split_file, dataloader_fn=dataloader_fn, dataset_fn=dataset_fn)

    # Setup training
    callbacks = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    model = instantiate(cfg.model)
    try:
        model.forward = torch.compile(model.forward)
    except Exception as e:
        logging.warning(f"Unable to compile forward pass: {e}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Save model
    torch.save(model.state_dict(), ckpt_path)
    torch.cuda.empty_cache()
