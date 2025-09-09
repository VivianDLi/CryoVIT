"""Config file for CryoVIT experiments."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from cryovit.types import Sample

samples: list[str] = [sample.name for sample in Sample]
hd_samples: list[str] = [
    "BACHD",
    "dN17_BACHD",
    "Q109",
    "Q18",
    "Q20",
    "Q53",
    "Q53_KD",
    "Q66",
    "Q66_GRFS1",
    "Q66_KD",
    "WT",
]
tomogram_exts: list[str] = [".hdf", ".mrc"]


@dataclass
class BaseModel:
    """Base class for model configurations used in CryoVIT experiments.

    Attributes:
        name (str): Name of the model for identification purposes.
        input_key (str): Key to get the input data from a tomogram.
        model_dir (Optional[Path]): Optional directory to download model weights to (for SAMv2 models).
        lr (float): Learning rate for the model training.
        weight_decay (float): Weight decay (L2 penalty) rate. Default is 1e-3.
        losses (tuple[dict]): Configuration for loss functions used in training.
        metrics (tuple[dict]): Configuration for metrics used during model evaluation.
        custom_kwargs (InitVar[dict]): Optional dictionary of custom keyword arguments to pass to the model.
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

    def __post_init__(self) -> None:
        if self.custom_kwargs is not None:
            for key, value in self.custom_kwargs.items():
                setattr(self, key, value)

        delattr(
            self, "custom_kwargs"
        )  # Remove custom_kwargs from the dataclass after initialization


@dataclass
class BaseTrainer:
    """Base class for trainer configurations used in CryoVIT experiments.

    Attributes:
        accelerator (str): Type of hardware acceleration ('gpu' for this configuration).
        devices (str): Number of devices to use for training.
        precision (str): Precision configuration for training (e.g., '16-mixed').
        default_root_dir (Path): Default root directory for saving checkpoints and logs.
        max_epochs (Optional[int]): The maximum number of epochs to train for.
        enable_checkpointing (bool): Flag to enable or disable model checkpointing.
        enable_model_summary (bool): Enable model summarization.
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
    """Base class for dataset configurations in CryoVIT experiments.

    Attributes:
        sample (Union[Sample, tuple[Sample]]): Specific sample or samples used for training.
        split_id (Optional[int]): Optional split_id to use for validation.
        test_sample (Union[Sample, tuple[Sample]]): Specific sample or samples used for testing.
        dataset (dict): Configuration options for the dataset.
        dataloader (dict): Configuration options for the dataloader.
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
    """Configuration for managing experiment paths in CryoVIT experiments.

    Attributes:
        model_dir (Path): Directory path for downloaded models.
        data_dir (Path): Directory path for tomogram data and .csv files.
        exp_dir (Path): Directory path for saving results from an experiment.
        results_dir (Path): Directory path for saving overall results.
        tomo_name (str): Name of the directory in data_dir with tomograms.
        feature_name (str): Name of the directory in data_dir with DINOv2 features.
        dino_name (str): Name of the directory in model_dir to save DINOv2 model.
        csv_name (str): Name of the directory in data_dir with .csv files.
        split_name(str): Name of the .csv file with training splits.
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
    """Configuration for managing DINOv2 features within CryoVIT experiments.

    Attributes:
        batch_size (int): Number of tomogram slices to process as one batch.
        dino_dir (Path): Path to the DINOv2 foundation model.
        envs (Path): Path to the directory containing tomograms.
        csv_dir (Optional[Path]): Path to the directory containing .csv files.
        feature_dir (Path): Destination to save the generated DINOv2 features.
        sample (Optional[Sample]): Sample to calculate features for. None means to calculate features for all samples.
        export_features (bool): Whether to additionally save calculated features as PCA color-maps for investigation.
    """

    batch_size: int = 128
    dino_dir: Path = MISSING
    paths: ExperimentPaths = MISSING
    datamodule: dict = MISSING
    sample: Sample | None = MISSING
    export_features: bool = False


@dataclass
class BaseExperimentConfig:
    """Base configuration for running experiment scripts.

    Attributes:
        name (str): Name of the experiment, must be unique for each configuration.
        label_key (str): Key used to specify the training label.
        additional_keys (tuple[str]): Additional keys to load auxiliary data from tomograms.
        random_seed (int): Random seed set for reproducibility.
        paths (ExperimentPaths): Configuration paths relevant to the experiment.
        model (BaseModel): Model configuration to use for the experiment.
        trainer (BaseTrainer): Trainer configuration to use for the experiment.
        callbacks (Optional[list]): list of callback functions for training sessions.
        logger (Optional[list]): list of logging functions for training sessions.
        dataset (BaseDataset): Dataset configuration to use for the experiment.
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

    Checks if all necessary parameters are present in the configuration. Logs an error and exits if any required parameters are missing.

    Also checks that all Samples specified are valid, and logs an error and exits if any samples are not valid.

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

    if cfg.datamodule.test_sample is not None:
        for sample in cfg.datamodule.test_sample:
            if sample not in samples:
                invalid_samples.append(sample)

    for i, sample in enumerate(invalid_samples, 1):
        error_msg.append(f"{i}. {sample}")

    if invalid_samples:
        logging.error("\n".join(error_msg))
        sys.exit(1)

    OmegaConf.set_struct(cfg, False)  # type: ignore
