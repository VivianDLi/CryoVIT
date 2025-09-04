import argparse
from pathlib import Path

from rich.progress import track
from hydra import initialize, compose
from hydra.utils import instantiate
import h5py
import numpy as np
import torch

from cryovit.utils import load_model

@torch.inference_mode()
def predict_model(data: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    data = data.cuda()
    preds = model(data)
    return preds.cpu().numpy()

def run_inference(data_path: Path, model_path: Path, result_dir: Path) -> None:
    ## Setup hydra config
    model, model_type, model_name, label_key = load_model(model_path)
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
    datamodule = instantiate(cfg.datamodule, _convert_="all")(data_path=data_path, val_path=None, dataloader_fn=dataloader_fn, dataset_fn=dataset_fn)
    dataloader = datamodule.predict_dataloader()
    print("Setup dataset.")

    for i, x in track(
            enumerate(dataloader),
            description=f"[green]Predicting {label_key} with {model_name}",
            total=len(dataloader),
    ):
        preds = predict_model(x, model)
        datas = x.tomo_batch.cpu().numpy()
        tomo_names = x.metadata.identifiers[1]
        
        # Save or process preds as needed
        for name, data, pred in zip(tomo_names, datas, preds):
            result_path = (result_dir / name).withsuffix(".hdf")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            # Save data and pred to HDF5 or other format
            with h5py.File(result_path, "w") as fh:
                fh.create_dataset("data", data=data, shape=data.shape, dtype=data.dtype, compression="gzip")
                fh.create_dataset("predictions", data=pred, shape=pred.shape, dtype=pred.dtype, compression="gzip")

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