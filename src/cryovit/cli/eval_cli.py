from pathlib import Path
from typing import Annotated

from typer import Argument, Option

from .cli import cli


@cli.command(name="evaluate", no_args_is_help=True)
def evaluate(
    test_data: Annotated[
        str,
        Argument(
            help="Path to the folder or .txt file containing the test tomograms.",
        ),
    ],
    test_labels: Annotated[
        str,
        Argument(
            help="Path to the folder or .txt file containing the test labels.",
        ),
    ],
    labels: Annotated[
        list[str],
        Option(help="List of available label names in ascending-value order."),
    ],
    model: Annotated[
        str,
        Argument(
            help="Path to the .model file containing the pre-trained model.",
        ),
    ],
    result_folder: Annotated[
        str | None,
        Option(
            help="Path to the directory to save the evaluation results. Evaluation metrics will be saved to a .csv file in a folder named 'results' inside the result folder.",
            show_default="the current working directory",
        ),
    ] = None,
    visualize: Annotated[
        bool,
        Option(
            "--visualize",
            "-v",
            help="Save visualizations of model predictions?. This will slightly increase the runtime. Results will be saved in a folder named `visualizations` inside the result folder.",
        ),
    ] = False,
):
    """Evaluate a pre-trained model on a test dataset.

    Example
    -------
    cryovit evaluate <path-to-test-data> <path-to-test-labels> --labels <first-label> ... --model <path-to-model-file> --result-folder <path-to-result-folder>
    """
    from cryovit._logging_config import setup_logging
    from cryovit.run.eval_model import run_evaluation
    from cryovit.utils import load_files_from_path, load_model

    setup_logging("INFO")

    ## Convert Arguments
    test_path = Path(test_data)
    label_path = Path(test_labels)
    model_path = Path(model)
    result_path = Path(result_folder) if result_folder else Path.cwd()

    ## Sanity Checking
    assert test_path.exists(), "Test data path does not exist."
    assert label_path.exists(), "Test labels path does not exist."
    assert (
        model_path.exists() and model_path.suffix == ".model"
    ), "Model path does not exist, or is not a .model file."
    _, _, _, label_key = load_model(model_path, load_model=False)
    assert (
        label_key in labels
    ), f"The label key {label_key} used to train the model is not in the provided labels."
    result_path.mkdir(parents=True, exist_ok=True)

    test_files = load_files_from_path(test_path)
    label_files = load_files_from_path(label_path)

    run_evaluation(
        test_files,
        label_files,
        labels,
        model_path,
        result_path,
        visualize=visualize,
    )
