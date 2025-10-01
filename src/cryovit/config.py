"""Hydra configuration classes for CryoViT experiments."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from cryovit.types import Sample

samples: list[str] = [sample.name for sample in Sample]
tomogram_exts: list[str] = [".hdf", ".mrc"]

DINO_PATCH_SIZE = 14
DEFAULT_WINDOW_SIZE = (
    630  # Calculated assuming DINOv2 patch size of 14 and input size of 720
)


@dataclass
class BaseModel:
    """Base configuration for models used in CryoViT experiments.

    Attributes:
        name (str): Name of the model for identification purposes.
        input_key (str): Key to get the input data from a tomogram.
        model_dir (Optional[Path]): Optional directory to download model weights to (for SAMv2 models).
        lr (float): Learning rate for the model training.
        weight_decay (float): Weight decay (L2 penalty) rate. Default is 1e-3.
        losses (dict[str, Any]): Configurations for loss functions used in training.
        metrics (dict[str, Any]): Configurations for metrics used during model evaluation.
        custom_kwargs (Optional[dict[str, Any]]): Optional dictionary of custom keyword arguments to pass to the model.
    """

    _target_: str = MISSING
    name: str = MISSING

    input_key: str = MISSING
    model_dir: Path | None = None
    lr: float = MISSING
    weight_decay: float = 1e-3
    losses: dict = MISSING
    metrics: dict = MISSING

    custom_kwargs: dict | None = None


@dataclass
class BaseTrainer:
    """Base configuration for the trainer used in CryoViT experiments.

    Attributes:
        accelerator (str): Type of hardware acceleration. Default is 'gpu'.
        devices (str): Number of devices to use for training. Default is '1'.
        precision (str): Precision configuration for training (e.g., '16-mixed').
        default_root_dir (Optional[Path]): Default root directory for saving checkpoints and logs.
        max_epochs (Optional[int]): The maximum number of epochs to train for.
        enable_checkpointing (bool): Flag to enable or disable model checkpointing. Default is False.
        enable_model_summary (bool): Enable model summarization. Default is True.
        log_every_n_steps (Optional[int]): Frequency of logging in terms of training steps.
    """

    _target_: str = "pytorch_lightning.Trainer"

    accelerator: str = "gpu"
    devices: str = "1"
    precision: str = "16-mixed"
    default_root_dir: Path | None = None
    max_epochs: int | None = None
    enable_checkpointing: bool = False
    enable_model_summary: bool = True
    log_every_n_steps: int | None = None


@dataclass
class BaseDataModule:
    """Base configuration for datasets in CryoViT experiments.

    Attributes:
        sample (Union[Sample, tuple[Sample]]): Specific sample or samples used for training.
        split_id (Optional[int]): Optional split_id to use for validation.
        split_key (Optional[str]): Key in the sample .csv file to use for splitting the data. Default is "split_id".
        test_sample (Optional[Any]): Specific sample or samples used for testing.
        dataset (dict[str, Any]): Configuration for the dataset.
        dataloader (dict[str, Any]): Configuration for the dataloader.
    """

    _target_: str = ""
    _partial_: bool = True

    # OmegaConf doesn't support Union[Sample, tuple[Sample]] yet, so moved type-checking to config validation instead
    sample: Any = MISSING
    split_id: int | None = None
    split_key: str | None = "split_id"
    test_sample: Any | None = None

    dataset: dict = MISSING
    dataloader: dict = MISSING


@dataclass
class ExperimentPaths:
    """Configuration for managing experiment paths in CryoViT experiments.

    Attributes:
        model_dir (Path): Path to the folder containing downloaded models.
        data_dir (Path): Path to the parent directory containing tomogram data and .csv files.
        exp_dir (Path): Path to the parent directory for saving results from an experiment.
        results_dir (Path): Path to the parent directory for saving overall results.
        tomo_name (str): Name of the folder in data_dir with tomograms.
        feature_name (str): Name of the folder in data_dir with DINOv2 features.
        dino_name (str): Name of the folder in model_dir to save DINOv2 model.
        csv_name (str): Name of the folder in data_dir with .csv files.
        split_name (str): Name of the .csv file with training splits.
    """

    model_dir: Path = MISSING
    data_dir: Path = MISSING
    exp_dir: Path = MISSING
    results_dir: Path = MISSING

    tomo_name: str = "tomograms"
    feature_name: str = "dino_features"
    dino_name: str = "DINOv2"
    sam_name: str = "SAM2"
    csv_name: str = "csv"
    split_name: str = "splits.csv"


@dataclass
class DinoFeaturesConfig:
    """Base configuration for computing DINOv2 features in CryoViT experiments.

    Attributes:
        batch_size (int): Number of tomogram slices to process as one batch. Default is 128.
        dino_dir (Path): Path to the DINOv2 foundation model.
        paths (ExperimentPaths): Configuration for experiment paths.
        datamodule (dict[str, Any]): Configuration for the datamodule to use for loading tomograms.
        sample (Optional[Sample]): Sample to calculate features for. None means to calculate features for all samples.
        export_features (bool): Whether to additionally compute PCA colormaps for the calculated features.
    """

    batch_size: int = 128
    dino_dir: Path = MISSING
    paths: ExperimentPaths = MISSING
    datamodule: dict = MISSING
    sample: Sample | None = MISSING
    export_features: bool = False


@dataclass
class BaseExperimentConfig:
    """Base configuration for running CryoViT experiments.

    Attributes:
        name (str): Name of the experiment, should be unique for each configuration.
        label_key (str): Key used to specify the training labels.
        additional_keys (tuple[str]): Keys to pass through additional data from the dataset.
        random_seed (int): Random seed set for reproducibility. Default is 42.
        paths (ExperimentPaths): Configuration for experiment paths.
        model (BaseModel): Configuration for the model to use.
        trainer (BaseTrainer): Configuration for the trainer to use.
        callbacks (Optional[list]): List of callback functions for the trainer.
        logger (Optional[list]): List of logging functions for the trainer.
        datamodule (BaseDataModule): Configuration for the datamodule to use.
        ckpt_path (Optional[Path]): Optional path to a checkpoint file to resume training from.
        resume_ckpt (bool): Whether to resume training from the checkpoint. Default is False.
    """

    name: str = MISSING
    label_key: str = MISSING
    additional_keys: tuple[str] = ()  # type: ignore
    random_seed: int = 42
    paths: ExperimentPaths = MISSING
    model: BaseModel = MISSING
    trainer: BaseTrainer = MISSING
    callbacks: dict[str, Any] = MISSING
    logger: dict[str, Any] = MISSING
    datamodule: BaseDataModule = MISSING
    ckpt_path: Path | None = None
    resume_ckpt: bool = False


cs = ConfigStore.instance()

cs.store(group="model", name="base_model", node=BaseModel)
cs.store(group="trainer", name="base_trainer", node=BaseTrainer)
cs.store(group="datamodule", name="base_datamodule", node=BaseDataModule)
cs.store(group="paths", name="base_env", node=ExperimentPaths)

cs.store(name="dino_features_config", node=DinoFeaturesConfig)
cs.store(name="base_experiment_config", node=BaseExperimentConfig)

#### Utility Functions for Configs ####\


def validate_dino_config(cfg: DinoFeaturesConfig) -> None:
    """Validates the configuration for DINOv2 feature extraction.

    Checks if all necessary parameters are present in the configuration. If any required parameters are
    missing, it logs an error message and exits the script.

    Args:
        cfg (DinoFeaturesConfig): The configuration object containing settings for feature extraction.

    Raises:
        SystemExit: If any configuration parameters are missing.
    """
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = [
        "The following parameters were missing from dino_features.yaml"
    ]

    for i, key in enumerate(missing_keys, 1):
        param_dict = DinoFeaturesConfig.__annotations__
        error_str = f"{i}. {key}: {param_dict.get(key, Any).__name__}"
        error_msg.append(error_str)

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)

    OmegaConf.set_struct(cfg, False)  # type: ignore


def validate_experiment_config(cfg: BaseExperimentConfig) -> None:
    """Validates an experiment configuration.

    Checks if all necessary parameters are present in the configuration. If any required parameters are
    missing, it logs an error message and exits the script.

    Additionally, checks that all Samples specified are valid, and logs an error and exits if any samples are not valid.

    Args:
        cfg (BaseExperimentConfig): The configuration object to validate.

    Raises:
        SystemExit: If any configuration parameters are missing, or any samples are not valid, terminating the script.
    """
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = ["The following parameters were missing from config:"]

    for i, key in enumerate(missing_keys, 1):
        error_msg.append(f"{i}. {key}")

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)

    # Check datamodule samples are valid
    error_msg = ["The following datamodule parameters are not valid samples:"]
    invalid_samples = []
    if isinstance(cfg.datamodule.sample, str):
        cfg.datamodule.sample = [cfg.datamodule.sample]
    if isinstance(cfg.datamodule.test_sample, str):
        cfg.datamodule.test_sample = [cfg.datamodule.test_sample]

    for sample in cfg.datamodule.sample:
        if sample not in samples:
            invalid_samples.append(sample)

    if cfg.datamodule.test_sample is not None and isinstance(
        cfg.datamodule.test_sample, list
    ):
        for sample in cfg.datamodule.test_sample:
            if sample not in samples:
                invalid_samples.append(sample)

    for i, sample in enumerate(invalid_samples, 1):
        error_msg.append(f"{i}. {sample}")

    if invalid_samples:
        logging.error("\n".join(error_msg))
        sys.exit(1)

    OmegaConf.set_struct(cfg, False)  # type: ignore
