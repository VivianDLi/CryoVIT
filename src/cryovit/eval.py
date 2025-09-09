import argparse
from pathlib import Path

from hydra import compose, initialize
from hydra.utils import instantiate

from cryovit.utils import load_files_from_path, load_model


def run_evaluation(
    test_data: list[Path],
    test_labels: list[Path],
    labels: list[str],
    model_path: Path,
    result_dir: Path,
    visualize: bool = True,
) -> Path:
    ## Get model information
    model, model_type, model_name, label_key = load_model(model_path)
    ## Setup hydra config
    with initialize(
        version_base="1.2",
        config_path="configs",
        job_name="cryovit_eval",
    ):
        cfg = compose(
            config_name="eval_model",
            overrides=[
                f"name={model_name}",
                f"label_key={label_key}",
                f"model={model_type}",
                "datamodule=file",
            ],
        )
    cfg.paths.model_dir = Path(__file__).parent / "foundation_models"
    cfg.paths.results_dir = result_dir

    # Check input key
    if cfg.model.input_key != "dino_features":
        cfg.model.input_key = None  # find available data instead

    ## Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    datamodule = instantiate(cfg.datamodule, _convert_="all")(
        data_paths=test_data,
        data_labels=test_labels,
        labels=labels,
        val_path=None,
        val_labels=None,
        dataloader_fn=dataloader_fn,
        dataset_fn=dataset_fn,
    )
    print("Setup dataset.")

    ## Setup training
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    # Remove pred_writer if visualize is False
    logger = [
        instantiate(lg_cfg)
        for lg_name, lg_cfg in cfg.logger.items()
        if (visualize or lg_name != "test_pred_writer")
    ]
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    print("Starting testing.")
    trainer.test(model, datamodule=datamodule)

    # Load and return metrics path
    metrics_path = cfg.paths.results_dir / "results" / f"{model_name}.csv"
    return metrics_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model evaluation given a test folder (or text file specifying files)."
    )
    parser.add_argument(
        "test_data",
        type=str,
        help="Directory or .txt file of test tomograms",
    )
    parser.add_argument(
        "test_labels",
        type=str,
        help="Directory or .txt file of test labels",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="list of label names in ascending-value order..",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        required=False,
        help="Path to the directory to save the evaluation results: a .csv file containing evaluation metrics. Defaults to the current directory.",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="If set, will save visualizations of the predictions in the result_dir folder.",
    )

    args = parser.parse_args()
    test_data = Path(args.test_data)
    test_labels = Path(args.test_labels)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else Path.cwd()

    ## Sanity Checking
    assert test_data.exists(), "Test data path does not exist."
    assert test_labels.exists(), "Test labels path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert (
        model_path.suffix == ".model"
    ), "Model path must point to a .model file."
    assert (
        result_dir.exists() and result_dir.is_dir()
    ), "Result directory either does not exist or isn't a directory."

    test_paths = load_files_from_path(test_data)
    test_label_paths = load_files_from_path(test_labels)

    run_evaluation(
        test_paths,
        test_label_paths,
        args.labels,
        model_path,
        result_dir,
        visualize=args.visualize,
    )
