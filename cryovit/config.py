"""Config file for CryoVIT experiments."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import logging
import sys

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING

class Sample(Enum):
    """Enum of all valid CryoET Samples."""

    BACHD = "BACHD"
    BACHD_Microtubules = "BACHD Microtubules"
    dN17_BACHD = "dN17 BACHD"
    Q109 = "Q109"
    Q109_Microtubules = "Q109 Microtubules"
    Q18 = "Q18"
    Q18_Microtubules = "Q18 Microtubules"
    Q20 = "Q20"
    Q53 = "Q53"
    Q53_KD = "Q53 PIAS1"
    Q66 = "Q66"
    Q66_GRFS1 = "Q66 GRFS1"
    Q66_KD = "Q66 PIAS1"
    WT = "Wild Type"
    WT_Microtubules = "Wild Type Microtubules"
    cancer = "Cancer"
    AD = "AD"
    AD_Abeta = "AD Abeta"
    Aged = "Aged"
    Young = "Young"
    RGC_CM = "RGC CM"
    RGC_control = "RGC Control"
    RGC_naPP = "RGC naPP"
    RGC_PP = "RGC PP"
    CZI_Algae = "Algae"
    CZI_Campy_C = "Campy C"
    CZI_Campy_CDel = "Campy C-Deletion"
    CZI_Campy_F = "Campy F"
    

samples: List[str] = [sample.name for sample in Sample]
tomogram_exts: List[str] = [".hdf", ".mrc"]


@dataclass
class BaseModel:
    """Base class for model configurations used in CryoVIT experiments.

    Attributes:
        input_key (str): Key to get the input data from a tomogram.
        lr (float): Learning rate for the model training.
        weight_decay (float): Weight decay (L2 penalty) rate. Default is 1e-3.
        losses (Tuple[Dict]): Configuration for loss functions used in training.
        metrics (Tuple[Dict]): Configuration for metrics used during model evaluation.
    """

    _target_: str = MISSING

    name: str = MISSING
    input_key: str = MISSING
    lr: float = MISSING
    weight_decay: float = 1e-3
    losses: Tuple[Dict] = (
        dict(
            _target_="cryovit.models.losses.DiceLoss",
            name="DiceLoss"
        ),
    )
    metrics: Tuple[Dict] = (
        dict(
            _target_="cryovit.models.metrics.DiceMetric",
            name="DiceMetric",
            threshold=0.5
        ),
    )


@dataclass
class BaseTrainer:
    """Base class for trainer configurations used in CryoVIT experiments.

    Attributes:
        accelerator (str): Type of hardware acceleration ('gpu' for this configuration).
        devices (str): Number of devices to use for training.
        precision (str): Precision configuration for training (e.g., '16-mixed').
        max_epochs (Optional[int]): The maximum number of epochs to train for.
        enable_checkpointing (bool): Flag to enable or disable model checkpointing.
        enable_model_summary (bool): Enable model summarization.
    """

    _target_: str = "pytorch_lightning.Trainer"
    
    accelerator: str = "gpu"
    devices: str = 1
    precision: str = "16-mixed"
    max_epochs: Optional[int] = None
    enable_checkpointing: bool = False
    enable_model_summary: bool = True


@dataclass
class BaseDataModule:
    """Base class for dataset configurations in CryoVIT experiments.

    Attributes:
        sample (Union[Sample, Tuple[Sample]]): Specific sample or samples used for training.
        split_id (Optional[int]): Optional split_id to use for validation.
        test_sample (Union[Sample, Tuple[Sample]]): Specific sample or samples used for testing.
        dataset (Dict): Configuration options for the dataset.
        dataloader (Dict): Configuration options for the dataloader.
    """

    _target_: str = MISSING
    _partial_: bool = True

    # OmegaConf doesn't support Union[Sample, Tuple[Sample]] yet, so moved type-checking to config validation instead
    sample: Any = MISSING
    split_id: Optional[int] = None
    split_key: Optional[str] = "split_id"
    test_sample: Optional[Any] = None
    
    dataset: Dict = MISSING
    dataloader: Dict = MISSING


@dataclass
class ExperimentPaths:
    """Configuration for managing experiment paths in CryoVIT experiments.

    Attributes:
        project_dir (Path): Directory path for code projects.
        data_dir (Path): Directory path for tomogram data and .csv files.
        exp_dir (Path): Directory path for saving results from an experiment.
        tomo_name (str): Name of the directory in data_dir with tomograms.
        feature_name (str): Name of the directory in data_dir with DINOv2 features.
        dino_name (str): Name of the directory in project_dir to save DINOv2 model.
        csv_name (str): Name of the directory in data_dir with .csv files.
        split_name(str): Name of the .csv file with training splits.
    """
    project_dir: Path = MISSING
    data_dir: Path = MISSING
    exp_dir: Path = MISSING
    
    tomo_name: str = "tomo_annot"
    feature_name: str = "dino_features"
    dino_name: str = "foundation_models"
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
    datamodule: BaseDataModule = MISSING
    sample: Optional[Sample] = MISSING
    export_features: bool = False


@dataclass
class BaseExperimentConfig:
    """Base configuration for running experiment scripts.
    
    Attributes:
        name (str): Name of the experiment, must be unique for each configuration.
        label_key (str): Key used to specify the training label.
        additional_keys (Tuple[str]): Additional keys to load auxiliary data from tomograms.
        random_seed (int): Random seed set for reproducibility.
        paths (ExperimentPaths): Configuration paths relevant to the experiment.
        model (BaseModel): Model configuration to use for the experiment.
        trainer (BaseTrainer): Trainer configuration to use for the experiment.
        callbacks (Optional[List]): List of callback functions for training sessions.
        logger (Optional[List]): List of logging functions for training sessions.
        dataset (BaseDataset): Dataset configuration to use for the experiment.
    """
    
    name: str = MISSING
    label_key: str = MISSING
    additional_keys: Tuple[str] = ()
    random_seed: int = 42
    paths: ExperimentPaths = MISSING
    model: BaseModel = MISSING
    trainer: BaseTrainer = MISSING
    callbacks: Optional[List] = None
    logger: Optional[List] = None
    datamodule: BaseDataModule = MISSING
    ckpt_path: Optional[Path] = None

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
    error_msg = ["The following parameters were missing from dino_features.yaml"]

    for i, key in enumerate(missing_keys, 1):
        param_dict = DinoFeaturesConfig.__annotations__
        error_str = f"{i}. {key}: {param_dict.get(key, Any).__name__}"
        error_msg.append(error_str)

    if missing_keys:
        logging.error("\n".join(error_msg))
        sys.exit(1)

def validate_experiment_config(cfg: BaseExperimentConfig) -> None:
    """Validates an experiment configuration.
    
    Checks if all necessary parameters are present in the configuration. Logs an error and exits if any required parameters are missing.
    
    Also checks that all Samples specified are valid, and logs an error and exits if any samples are not valid.
    
    Args:
        cfg (BaseExperimentConfig): The configuration object to validate.
        
    Raises:
        SystemExit: If any configuration parameters are missing, or any samples are not valid, terminating the script."""
    missing_keys = OmegaConf.missing_keys(cfg)
    error_msg = ["The following parameters were missing from train_model.yaml:"]

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
    for sample in cfg.datamodule.sample:
        if sample not in samples:
            invalid_samples.append(sample)
    
    for i, sample in enumerate(invalid_samples, 1):
        error_msg.append(f"{i}. {sample}")
    
    if invalid_samples:
        logging.error("\n".join(error_msg))
        sys.exit(1)

    OmegaConf.set_struct(cfg, False)