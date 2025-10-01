from pathlib import Path
from typing import Annotated

from typer import Argument, Option

from .cli import cli


@cli.command(name="infer", no_args_is_help=True)
def infer(
    tomograms: Annotated[
        str,
        Argument(
            help="Path to the folder or .txt file containing the tomograms to process.",
        ),
    ],
    model: Annotated[
        str,
        Option(
            help="Path to the .model file containing the pre-trained model.",
        ),
    ],
    result_folder: Annotated[
        str | None,
        Option(
            help="Path to the folder where the inference results will be saved.",
            show_default="the current working directory",
        ),
    ] = None,
    threshold: Annotated[
        float,
        Option(
            min=0.0,
            max=1.0,
            help="Threshold for binary segmentation.",
        ),
    ] = 0.5,
):
    """Segment tomograms using a pre-trained model.

    Example
    -------
    cryovit infer <path-to-tomograms> --model <path-to-model> --result-folder <path-to-result-folder>
    """
    from cryovit._logging_config import setup_logging
    from cryovit.run.infer_model import run_inference
    from cryovit.utils import load_files_from_path

    setup_logging("INFO")

    ## Convert Arguments
    tomograms_path = Path(tomograms)
    model_path = Path(model)
    result_path = (
        Path(result_folder) if result_folder else Path.cwd() / "predictions"
    )

    ## Sanity Checking
    assert tomograms_path.exists(), "Tomograms path does not exist."
    assert (
        model_path.exists() and model_path.suffix == ".model"
    ), "Model path does not exist or is not a .model file."
    result_path.mkdir(parents=True, exist_ok=True)

    tomogram_files = load_files_from_path(tomograms_path)

    run_inference(tomogram_files, model_path, result_path, threshold=threshold)
