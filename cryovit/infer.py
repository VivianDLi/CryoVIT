import argparse
from pathlib import Path

from hydra import initialize, compose
from hydra.utils import instantiate

from cryovit.utils import load_model

def run_inference(data: Path, model_path: Path, result_dir: Path) -> None:
    ## Get model information
    model, model_type, model_name, label_key = load_model(model_path)
    ## Setup hydra config
    config_path = Path(__file__).parent / "configs"
    with initialize(version_base="1.2", config_path=str(config_path), job_name="infer_model"):
        cfg = compose(config_name="eval_model", overrides=[f"name={model_name}", f"label_key={label_key}", f"model={model_type}", "datamodule=file"])
    cfg.paths.model_dir = Path(__file__).parent / "foundation_models"
    cfg.paths.results_dir = result_dir
        
    # Check input key
    if cfg.model.input_key != "dino_features":
        cfg.model.input_key = None # find available data instead

    ## Setup dataset
    dataset_fn = instantiate(cfg.datamodule.dataset)
    dataloader_fn = instantiate(cfg.datamodule.dataloader)
    datamodule = instantiate(cfg.datamodule, _convert_="all")(data_path=data, val_path=None, dataloader_fn=dataloader_fn, dataset_fn=dataset_fn)
    print("Setup dataset.")

    ## Setup training
    callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    # Remove pred_writer if visualize is False
    logger = [instantiate(lg_cfg) for lg_cfg in cfg.logger.values()]
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    print("Starting inference.")
    trainer.predict(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference given a data folder (or text file specifying files).")
    parser.add_argument("data", type=str, required=True, help="Directory or .txt file of data tomograms")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--result_dir", type=str, default=None, required=False, help="Path to the directory to save the inference results. Defaults to the current directory.")

    args = parser.parse_args()
    data = Path(args.data)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else Path.cwd()

    ## Sanity Checking
    assert data.exists(), "Data path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert model_path.suffix == ".model", "Model path must point to a .model file."
    assert result_dir.exists() and result_dir.is_dir(), "Result directory either does not exist or isn't a directory."

    run_inference(data, model_path, result_dir)