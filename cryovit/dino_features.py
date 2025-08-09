"""Script to extract DINOv2 features from tomograms."""

import logging
import warnings
import traceback

import hydra

from cryovit.config import DinoFeaturesConfig, validate_dino_config
from cryovit.run import dino_features


warnings.simplefilter("ignore")

@hydra.main(
    config_path="configs",
    config_name="dino_features",
    version_base="1.2",
)
def main(cfg: DinoFeaturesConfig) -> None:
    """Main function to process DINOv2 feature extraction.

    Validates the configuration and processes the sample as per the specified settings. Errors during
    processing are caught and logged.

    Args:
        cfg (DinoFeaturesConfig): Configuration object loaded from dino_features.yaml.

    Raises:
        BaseException: Captures and logs any exceptions that occur during the processing of the sample.
    """
    validate_dino_config(cfg)

    try:
        dino_features.run_dino(cfg)
    except BaseException as err:
        logging.error(f"{type(err).__name__}: {err}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
