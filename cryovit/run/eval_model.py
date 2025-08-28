"""Script for evaluating CryoVIT models based on configuration files."""

from collections.abc import Iterable
import logging
from pathlib import Path
from typing import Tuple

import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything

from cryovit.models import create_sam_model_from_weights
from cryovit.config import BaseExperimentConfig

torch.set_float32_matmul_precision("high")
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def setup_exp_dir(cfg: BaseExperimentConfig) -> Tuple[BaseExperimentConfig, Path]:
    # Convert paths to Paths
    cfg.paths.model_dir = Path(cfg.paths.model_dir)
    cfg.paths.data_dir = Path(cfg.paths.data_dir)
    cfg.paths.exp_dir = Path(cfg.paths.exp_dir)
    cfg.paths.results_dir = Path(cfg.paths.results_dir)
    
    if not isinstance(cfg.datamodule.sample, str) and isinstance(cfg.datamodule.sample, Iterable):
        sample = "_".join(sorted(cfg.datamodule.sample))
    else:
        sample = cfg.datamodule.sample
        
    new_exp_dir = cfg.paths.exp_dir / cfg.name / sample
    if cfg.datamodule.split_id is not None:
        new_exp_dir = new_exp_dir / f"split_{cfg.datamodule.split_id}"
    
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    assert new_exp_dir.exists(), f"Experiment directory {new_exp_dir} does not exist. Run training first."
    cfg.paths.exp_dir = new_exp_dir
    
    if cfg.ckpt_path is None:
        cfg.ckpt_path = new_exp_dir / "weights.pt"
    return cfg

def run_trainer(cfg: BaseExperimentConfig) -> None:
    """Sets up and executes the model evaluation using the specified configuration.

    Args:
        cfg (EvalModelConfig): Configuration object containing all settings for the evaluation process.
    """
    seed_everything(cfg.random_seed, workers=True)
    
    # Setup experiment directories
    cfg = setup_exp_dir(cfg)
    assert cfg.ckpt_path.exists(), f"{cfg.paths.exp_dir} does not contain a checkpoint."
    
    # Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    split_file = cfg.paths.data_dir / cfg.paths.csv_name / cfg.paths.split_name
    datamodule = instantiate(cfg.datamodule, _convert_="all")(split_file=split_file, dataloader_fn=dataloader_fn, dataset_fn=dataset_fn)
    logging.info("Setup dataset.")
    
    # Setup evaluation
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    logger = [instantiate(lg_cfg) for lg_cfg in cfg.logger.values()]
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    logging.info("Setup trainer.")
    if cfg.model._target_ == "cryovit.models.SAM2":
        # Load SAM2 pre-trained models
        model = create_sam_model_from_weights(cfg.model, cfg.paths.model_dir / cfg.paths.sam_name)
    else:
        model = instantiate(cfg.model)

    # Load model weights
    if cfg.ckpt_path.suffix == ".pt":
        model.load_state_dict(torch.load(cfg.ckpt_path))
    elif cfg.ckpt_path.suffix == ".ckpt":
        model = model.load_from_checkpoint(cfg.ckpt_path)
    else:
        raise ValueError(f"Unsupported checkpoint format: {cfg.ckpt_path.suffix}. Use .pt or .ckpt files.")
    
    logging.info("Setup model.")
    
    logging.info("Starting testing.")
    trainer.test(model, datamodule=datamodule)
