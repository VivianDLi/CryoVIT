"""Script to train segmentation models for CryoET data."""

import logging
import warnings

import hydra

from cryovit.config import BaseExperimentConfig, validate_experiment_config
from cryovit.run import train_model


warnings.simplefilter("ignore") 


@hydra.main(
    config_path="configs",
    config_name="train_model",
    version_base="1.2",
)
def main(cfg: BaseExperimentConfig) -> None:
    """Main function to orchestrate the training of segmentation models.

    Validates the provided configuration, then initializes and runs the training process using the
    specified settings. Catches and logs any exceptions that occur during training.

    Args:
        cfg (TrainModelConfig): Configuration object loaded from train_model.yaml.

    Raises:
        BaseException: Captures and logs any exceptions that occur during the training process.
    """
    validate_experiment_config(cfg)

    try:
        train_model.run_trainer(cfg)
    except BaseException as err:
        logging.error(f"{type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
