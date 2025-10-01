import traceback
import warnings

import hydra

from cryovit._logging_config import setup_logging
from cryovit.config import DinoFeaturesConfig, validate_dino_config
from cryovit.run import dino_features

setup_logging("DEBUG")
import logging  # noqa: E402

warnings.simplefilter("ignore")


@hydra.main(
    config_path="../configs",
    config_name="dino_features",
    version_base="1.2",
)
def main(cfg: DinoFeaturesConfig) -> None:
    """Main function to process DINOv2 feature extraction.

    Validates the configuration and processes the sample as per the specified settings. Errors during
    processing are caught and logged.

    Args:
        cfg (DinoFeaturesConfig): Configuration object loaded from dino_features.yaml.
    """

    validate_dino_config(cfg)

    try:
        dino_features.run_trainer(cfg)
    except BaseException as err:  # noqa: BLE001
        logging.error("%s: %s", type(err).__name__, err)
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
