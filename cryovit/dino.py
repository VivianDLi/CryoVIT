import argparse
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from rich.progress import track

from cryovit.datamodules import FileDataModule
from cryovit.run.dino_features import _dino_features, _save_data, dino_model


def run_dino(train_data: Path, result_data: Path, batch_size: int) -> None:
    ## Setup hydra config
    config_path = Path(__file__).parent / "configs"
    with initialize(
        version_base="1.2",
        config_path=str(config_path),
        job_name="cryovit_features",
    ):
        cfg = compose(
            config_name="dino_features",
            overrides=[
                f"batch_size={batch_size}, sample=null",
                "export_features=False",
                "datamodule.dataset=file",
            ],
        )
    cfg.paths.model_dir = Path(__file__).parent / "foundation_models"

    ## Load DINOv2 model
    torch.hub.set_dir(cfg.dino_dir)
    model = torch.hub.load(*dino_model, verbose=False).cuda()  # type: ignore
    model.eval()

    ## Setup dataset
    file_list = FileDataModule._load_files(train_data)
    assert (
        file_list is not None and len(file_list) > 0
    ), "No valid tomogram files found in the specified training data path."
    dataset = instantiate(cfg.datamodule.dataset)(
        file_list, input_key=None, label_key=None, for_dino=True
    )
    dataloader = instantiate(cfg.datamodule.dataloader)(dataset=dataset)

    if result_data.suffix == ".txt":
        result_list = [
            fd.tomo_path for fd in FileDataModule._load_files(result_data)
        ]
    else:
        result_list = [result_data / f"{f.stem}_dino.hdf" for f in file_list]

    ## Iterate through dataloader and extract features
    try:
        for i, x in track(
            enumerate(dataloader),
            description="[green]Computing DINO features for training data",
            total=len(dataloader),
        ):
            features = _dino_features(x, model, cfg.batch_size)

            result_path = result_list[i].with_suffix(".hdf")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_dir, tomo_name = result_path.parent, result_path.name
            data = {"data": x.cpu().numpy()}
            _save_data(data, features, tomo_name, result_dir)
    except torch.OutOfMemoryError:
        print(
            f"Ran out of GPU memory during DINO feature extraction. Try reducing the batch size. Current batch size is {cfg.batch_size}."
        )
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate DINOv2 features for a given training dataset folder (or text file specifying files)."
    )
    parser.add_argument(
        "train_data",
        type=str,
        required=True,
        help="Directory or .txt file of training tomograms",
    )
    parser.add_argument(
        "result_data",
        type=str,
        required=True,
        help="Directory or .txt file of file paths to save the DINO features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for DINO feature extraction (default: 128)",
    )

    args = parser.parse_args()
    train_data = Path(args.train_data)
    result_data = Path(args.result_data)
    batch_size = args.batch_size

    ## Sanity Checking
    assert train_data.exists(), "Training data path does not exist."
    if result_data.suffix == ".txt":
        assert result_data.exists(), "Result data .txt file does not exist."

    run_dino(train_data, result_data, batch_size)
