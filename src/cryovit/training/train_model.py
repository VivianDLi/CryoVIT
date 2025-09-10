"""Script to train segmentation models for CryoET data."""

import traceback
import warnings

import hydra
import torch

from cryovit._logging_config import setup_logging
from cryovit.config import BaseExperimentConfig, validate_experiment_config
from cryovit.run import train_model

setup_logging("DEBUG")
import logging  # noqa: E402

warnings.simplefilter("ignore")


@hydra.main(
    config_path="../configs",
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
    result = 0
    try:
        train_model.run_trainer(cfg)
    except BaseException as err:  # noqa: BLE001
        logging.error("%s: %s", type(err).__name__, err)
        logging.error(traceback.format_exc())
        result = -1
    finally:
        import wandb

        # Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Ensure W&B finishes properly
        wandb.Settings(quiet=True)  # Disable W&B output
        if wandb.run is not None:
            wandb.finish(exit_code=result)


if __name__ == "__main__":
    main()
