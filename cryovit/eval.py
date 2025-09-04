import argparse
from pathlib import Path
from typing import List

from hydra import initialize, compose
from hydra.utils import instantiate
import pandas as pd

from cryovit.utils import load_model

def run_evaluation(test_data: Path, test_labels: Path, labels: List[str], model_path: Path, result_dir: Path, visualize: bool = True) -> pd.DataFrame:
    ## Get model information
    model, model_type, model_name, label_key = load_model(model_path)
    ## Setup hydra config
    config_path = Path(__file__).parent / "configs"
    with initialize(version_base="1.2", config_path=str(config_path), job_name="cryovit_eval"):
        cfg = compose(config_name="eval_model", overrides=[f"name={model_name}", f"label_key={label_key}", f"model={model_type}", "datamodule=file"])
    cfg.paths.model_dir = Path(__file__).parent / "foundation_models"
    cfg.paths.results_dir = result_dir
        
    # Check input key
    if cfg.model.input_key != "dino_features":
        cfg.model.input_key = None # find available data instead

    ## Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    datamodule = instantiate(cfg.datamodule, _convert_="all")(data_path=test_data, data_labels=test_labels, labels=labels, val_path=None, val_labels=None, dataloader_fn=dataloader_fn, dataset_fn=dataset_fn)
    print("Setup dataset.")

    ## Setup training
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    # Remove pred_writer if visualize is False
    logger = [instantiate(lg_cfg) for lg_name, lg_cfg in cfg.logger.items() if (visualize or lg_name != "test_pred_writer")]
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    print("Starting testing.")
    trainer.test(model, datamodule=datamodule)
    
    # Load and return metrics
    metrics_path = cfg.paths.results_dir / "results" / f"{model_name}.csv"
    metrics = pd.read_csv(metrics_path)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation given a test folder (or text file specifying files).")
    parser.add_argument("test_data", type=str, required=True, help="Directory or .txt file of test tomograms")
    parser.add_argument("test_labels", type=str, required=True, help="Directory or .txt file of test labels")
    parser.add_argument("--labels", type=str, nargs='+', required=True, help="List of label names in ascending-value order..")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--result_dir", type=str, default=None, required=False, help="Path to the directory to save the evaluation results: a .csv file containing evaluation metrics. Defaults to the current directory.")
    parser.add_argument("--visualize", "-v", action="store_true", help="If set, will save visualizations of the predictions in the result_dir folder.")

    args = parser.parse_args()
    test_data = Path(args.test_data)
    test_labels = Path(args.test_labels)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else Path.cwd()

    ## Sanity Checking
    assert test_data.exists(), "Test data path does not exist."
    assert test_labels.exists(), "Test labels path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert model_path.suffix == ".model", "Model path must point to a .model file."
    assert result_dir.exists() and result_dir.is_dir(), "Result directory either does not exist or isn't a directory."

    run_evaluation(test_data, test_labels, args.labels, model_path, result_dir, visualize=args.visualize)