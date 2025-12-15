from pathlib import Path
from typing import Annotated

from typer import Argument, Option

from .cli import cli


@cli.command(name="features", no_args_is_help=True)
def features(
    tomograms: Annotated[
        str,
        Argument(
            help="Path to the folder or .txt file containing the tomograms to process.",
        ),
    ],
    result_folder: Annotated[
        str,
        Argument(
            help="Path to the folder where the DINO features will be saved.",
        ),
    ],
    batch_size: Annotated[
        int,
        Option(
            min=1,
            help="Batch size for DINO feature extraction.",
            rich_help_panel="Customization and Utils",
        ),
    ] = 64,
    visualize: Annotated[
        bool,
        Option(
            "--visualize",
            "-v",
            help="Save PCA visualization of DINO features? This will increase the runtime.",
            rich_help_panel="Customization and Utils",
        ),
    ] = False,
):
    """Compute high-level features using DINOv2 for a set of tomograms.

    Example
    -------
    cryovit features <path-to-tomograms> <path-to-result-folder>
    """
    from cryovit._logging_config import setup_logging
    from cryovit.run.dino_features import run_dino
    from cryovit.utils import load_files_from_path

    setup_logging("INFO")

    ## Convert Arguments
    tomograms_path = Path(tomograms)
    result_path = Path(result_folder)

    ## Sanity Checking
    assert tomograms_path.exists(), "Tomograms path does not exist."
    result_path.mkdir(parents=True, exist_ok=True)

    ## Run DINO feature extraction
    tomogram_files = load_files_from_path(tomograms_path)
    run_dino(
        tomogram_files,
        result_path,
        batch_size=batch_size,
        visualize=visualize,
    )
