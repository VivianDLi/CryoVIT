"""Script for evaluating CryoVIT models based on configuration files."""

from collections.abc import Iterable
import logging
from pathlib import Path
from typing import Tuple

import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything

from cryovit.config import BaseExperimentConfig

torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def setup_exp_dir(cfg: BaseExperimentConfig) -> Tuple[BaseExperimentConfig, Path]:
    old_exp_dir = cfg.paths.exp_dir
    
    if not isinstance(cfg.datamodule.sample, str) and isinstance(cfg.datamodule.sample, Iterable):
        sample = "_".join(sorted(cfg.datamodule.sample))
    else:
        sample = cfg.datamodule.sample
        
    new_exp_dir = cfg.paths.exp_dir / cfg.name / sample
    if cfg.datamodule.split_id is not None:
        new_exp_dir = new_exp_dir / f"split_{cfg.datamodule.split_id}"
    
    new_exp_dir.mkdir(parents=True, exist_ok=False)
    cfg.paths.exp_dir = new_exp_dir
    
    if cfg.ckpt_path is None:
        cfg.ckpt_path = new_exp_dir / "weights.pt"
    return cfg, old_exp_dir

def run_trainer(cfg: BaseExperimentConfig) -> None:
    """Sets up and executes the model evaluation using the specified configuration.

    Args:
        cfg (EvalModelConfig): Configuration object containing all settings for the evaluation process.
    """
    seed_everything(cfg.random_seed, workers=True)
    
    # Setup experiment directories
    cfg, old_exp_dir = setup_exp_dir(cfg)
    prediction_result_dir = old_exp_dir / "predictions" / f"{cfg.name}"
    csv_result_path = old_exp_dir / "results" / f"{cfg.name}.csv"
    assert cfg.ckpt_path.exists(), f"{cfg.paths.exp_dir} does not contain a checkpoint."
    
    # Setup dataset
    dataset = instantiate(cfg.datamodule.dataset)
    dataloader = instantiate(cfg.datamodule.dataloader)
    split_file = cfg.paths.data_dir / cfg.paths.csv_name / cfg.paths.split_name
    datamodule = instantiate(cfg.datamodule)(split_file=split_file, dataloader_fn=dataloader, dataset_fn=dataset)
    
    # Setup evaluation
    callbacks = instantiate(cfg.callbacks, results_dir=prediction_result_dir, csv_result_path=csv_result_path)
    logger = instantiate(cfg.logger)
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    model = instantiate(cfg.model)
    
    trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
