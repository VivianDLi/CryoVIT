"""Functions to load data and run DINOv2 feature extraction."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from numpy.typing import NDArray
from rich.progress import track

from cryovit.config import (
    DEFAULT_WINDOW_SIZE,
    BaseDataModule,
    DinoFeaturesConfig,
    samples,
    tomogram_exts,
)
from cryovit.types import FileData
from cryovit.visualization.dino_pca import export_pca

torch.set_float32_matmul_precision("high")  # ensures tensor cores are used
dino_model = (
    "facebookresearch/dinov2",
    "dinov2_vitg14_reg",
)  # the giant variant of DINOv2


@torch.inference_mode()
def _dino_features(
    data: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
) -> NDArray[np.float16]:
    """Extract patch features from a tomogram using a DINOv2 model.

    Args:
        data (torch.Tensor): The input data tensors containing the tomogram's data.
        model (nn.Module): The pre-loaded DINOv2 model used for feature extraction.
        batch_size (int): The maximum number of 2D slices of a tomograms processed in each batch.

    Returns:
        NDArray[np.float16]: A numpy array containing the extracted features in reduced precision.
    """
    data = data.cuda()
    w, h = (
        np.array(data.shape[-2:]) // 14
    )  # the number of patches extracted per slice
    all_features = []

    for i in range(0, len(data), batch_size):
        # Check for overflows
        if i + batch_size > len(data):
            batch_size = len(data) - i
        vec = data[i : i + batch_size]
        features = model.forward_features(vec)["x_norm_patchtokens"]  # type: ignore
        features = features.reshape(features.shape[0], w, h, -1)
        features = features.permute([3, 0, 1, 2]).contiguous()
        features = features.to("cpu").half().numpy()
        all_features.append(features)

    return np.concatenate(all_features, axis=1)  # type: ignore

@torch.inference_mode()
def _sam_features(
    data: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
) -> list[NDArray[np.float32]]:
    """Extract patch features from a tomogram using a SAM model.

    Args:
        data (torch.Tensor): The input data tensors containing the tomogram's data.
        model (nn.Module): The pre-loaded SAM model used for feature extraction.
        batch_size (int): The maximum number of 2D slices of a tomograms processed in each batch.

    Returns:
        list[NDArray[np.float32]]: A list of numpy arrays containing the extracted features of different levels.
    """
    data = data.cuda()
    all_features = []

    for i in range(0, len(data), batch_size):
        # Check for overflows
        if i + batch_size > len(data):
            batch_size = len(data) - i
        vec = data[i : i + batch_size]
        features = model.forward_features(vec)  # type: ignore
        for i, feature in enumerate(features):
            feature = feature.to("cpu").numpy()
            if len(all_features) <= i:
                all_features.append(feature)
            else:
                all_features[i] = np.concatenate(
                    [all_features[i], feature], axis=0
                )  # type: ignore

    return all_features

def _save_data(
    data: dict[str, NDArray[np.uint8]],
    features: NDArray[np.float16] | list[NDArray[np.float32]],
    tomo_name: str,
    dst_dir: Path,
) -> None:
    """Save extracted features to a specified directory as an .hdf file, copying the source tomogram data."""

    dst_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dst_dir / tomo_name, "w") as fh:
        for key in data:
            if key != "data" and key != "dino_features":
                if "labels" not in fh:
                    fh.create_group("labels")
                fh["labels"].create_dataset(key, data=data[key], shape=data[key].shape, dtype=data[key].dtype, compression="gzip")  # type: ignore
            elif key == "dino_features":
                continue # skip original dino_features
            else:
                fh.create_dataset(
                    "data",
                    data=data[key],
                    shape=data[key].shape,
                    dtype=data[key].dtype,
                    compression="gzip",
                )
        if isinstance(features, list):
            if "dino_features" in data:
                # Save original dino_features if SAM features were computed
                fh.create_dataset("dino_features", data=data["dino_features"], shape=data["dino_features"].shape, dtype=data["dino_features"].dtype, compression="gzip")  # type: ignore
            for i, feat in enumerate(features):
                fh.create_dataset(
                    f"sam_features/{i}",
                    data=feat,
                    shape=feat.shape,
                    dtype=feat.dtype,
                )
        else:
            fh.create_dataset(
                "dino_features",
                data=features,
                shape=features.shape,
                dtype=features.dtype,
            )


def _process_sample(
    src_dir: Path,
    dst_dir: Path,
    csv_dir: Path,
    model: torch.nn.Module,
    sample: str,
    datamodule: BaseDataModule,
    batch_size: int,
    use_sam: bool = False,
    image_dir: Path | None,
):
    """Process all tomograms in a single sample by extracting and saving their DINOv2 features."""

    tomo_dir = src_dir / sample
    result_dir = dst_dir / sample
    csv_file = csv_dir / f"{sample}.csv"
    # If no .csv file, use all tomograms in the dataset
    if not csv_file.exists():
        records = [
            f.name for f in tomo_dir.glob("*") if f.suffix in tomogram_exts
        ]
    else:
        records = pd.read_csv(csv_file)["tomo_name"].to_list()

    dataset = instantiate(datamodule.dataset, data_root=tomo_dir, use_sam=use_sam)(
        records=records
    )
    dataloader = instantiate(datamodule.dataloader)(dataset=dataset)

    for i, x in track(
        enumerate(dataloader),
        description=f"[green]Computing features for {sample}",
        total=len(dataloader),
    ):
        feature_fn = _sam_features if use_sam else _dino_features
        features = feature_fn(x, model, batch_size)

        data = {}
        with h5py.File(tomo_dir / records[i]) as fh:
            for key in fh:
                data[key] = fh[key][()]  # type: ignore

        _save_data(data, features, records[i], result_dir)
        # Save PCA calculation of features
        if image_dir is not None:
            export_pca(data["data"], features.astype(np.float32), records[i][:-4], image_dir / sample)  # type: ignore


## For Scripts


def run_dino(
    train_data: list[Path],
    result_dir: Path,
    batch_size: int,
    use_sam: bool = False,
    visualize: bool = False,
) -> None:
    """Run DINO feature extraction on the specified training data, and saves the results as .hdf files.
    The saved result file will contain `data`, `dino_features`, and any labels present in the source tomogram in the `labels/` group.

    Args:
        train_data (list[Path]): List of paths to the training tomograms.
        result_dir (Path): Directory where the results will be saved.
        batch_size (int): Number of samples to process in each batch.
        use_sam (bool, optional): Whether to use a SAM2 model for feature extraction instead of DINOv2. Defaults to False.
        visualize (bool, optional): Whether to visualize the extracted features. Defaults to False.
    """

    ## Setup hydra config
    with initialize(
        version_base="1.2",
        config_path="../configs",
        job_name="cryovit_features",
    ):
        cfg = compose(
            config_name="dino_features" if not use_sam else "sam_features",
            overrides=[
                f"batch_size={batch_size}",
                "sample=null",
                "export_features=False",
                "datamodule/dataset=file",
            ],
        )
        sam_cfg = compose(config_name="model/sam2")

    if use_sam:
        ## Load SAM2 model
        model = create_sam_model_from_weights(sam_cfg, cfg.model_dir)
        model.eval()
    else:
        ## Load DINOv2 model
        torch.hub.set_dir(cfg.model_dir)
        model = torch.hub.load(*dino_model, verbose=False).cuda()  # type: ignore
        model.eval()

    ## Setup dataset
    assert (
        len(train_data) > 0
    ), "No valid tomogram files found in the specified training data path."
    train_file_datas = [FileData(tomo_path=f) for f in train_data]
    dataset = instantiate(
        cfg.datamodule.dataset, input_key=None, label_key=None
    )(train_file_datas, for_dino=True, use_sam=use_sam)
    dataloader = instantiate(cfg.datamodule.dataloader)(dataset=dataset)

    result_list = [result_dir / f"{f.stem}.hdf" for f in train_data]

    ## Iterate through dataloader and extract features
    try:
        for i, x in track(
            enumerate(dataloader),
            description="[green]Computing DINO features for training data",
            total=len(dataloader),
        ):
            feature_fn = _sam_features if use_sam else _dino_features
            features = feature_fn(x.data, model, cfg.batch_size)

            result_path = result_list[i].with_suffix(".hdf")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_dir, tomo_name = result_path.parent, result_path.name
            _save_data(x.aux_data, features, tomo_name, result_dir)
            if visualize:
                export_pca(
                    x.aux_data["data"],
                    features,
                    tomo_name[:-4],
                    result_dir.parent / "dino_images" / tomo_name[:-4],
                )
    except torch.OutOfMemoryError:
        print(
            f"Ran out of GPU memory during DINO feature extraction. Try reducing the batch size. Current batch size is {cfg.batch_size}."
        )
        return


## For Experiments


def run_trainer(cfg: DinoFeaturesConfig) -> None:
    """Sets up and executes the DINO feature computation using the specified configuration.

    Args:
        cfg (DinoFeaturesConfig): Configuration object containing all settings for the DINO feature computation.
    """

    # Convert paths to Paths
    cfg.paths.model_dir = Path(cfg.paths.model_dir)
    cfg.paths.data_dir = Path(cfg.paths.data_dir)
    cfg.paths.exp_dir = Path(cfg.paths.exp_dir)

    src_dir = cfg.paths.data_dir / cfg.paths.feature_name
    dst_dir = cfg.paths.data_dir / cfg.paths.tomo_name
    csv_dir = cfg.paths.data_dir / cfg.paths.csv_name
    image_dir = cfg.paths.exp_dir / "dino_images"
    sample_names = (
        [cfg.sample.name]
        if cfg.sample is not None
        else [s for s in samples if (src_dir / s).exists()]
    )

    if cfg.use_sam:
        ## Load SAM2 model
        model = create_sam_model_from_weights(sam_cfg, cfg.model_dir)
        model.eval()
    else:
        ## Load DINOv2 model
        torch.hub.set_dir(cfg.model_dir)
        model = torch.hub.load(*dino_model, verbose=False).cuda()  # type: ignore
        model.eval()

    for sample_name in sample_names:
        _process_sample(
            src_dir,
            dst_dir,
            csv_dir,
            model,
            sample_name,
            cfg.datamodule,  # type: ignore
            cfg.batch_size,
            cfg.use_sam,
            image_dir if cfg.export_features else None,
        )
