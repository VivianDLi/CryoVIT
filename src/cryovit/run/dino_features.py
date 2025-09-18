"""Functions to load data and run DINOv2 feature extraction."""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize
from hydra.utils import instantiate
from numpy.typing import NDArray
from rich.progress import track

from cryovit.config import (
    DEFAULT_WINDOW_SIZE,
    DINO_PATCH_SIZE,
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


def _folded_to_patch(
    x: torch.Tensor, C: int, window_size: tuple[int, int]
) -> tuple[torch.Tensor, int, int]:
    """Convert unfolded tensor to patch features tensor shape."""

    B, _, L = x.shape  # B, C * window_size * window_size, L
    x = x.permute(0, 2, 1).contiguous()  # B, L, C * window_size * window_size
    x = x.reshape(
        -1, C, window_size[0], window_size[1]
    )  # B * L, C, window_size, window_size
    return x, B, L


def _patch_to_folded(x: torch.Tensor, B: int, L: int) -> torch.Tensor:
    """Convert patch features tensor to folded tensor shape."""

    x = x.reshape(
        B, L, x.shape[1], x.shape[2]
    )  # B, L, window_size * window_size, C2
    x = x.permute(
        0, 3, 2, 1
    ).contiguous()  # B, C2, window_size * window_size, L
    x = x.reshape(B, -1, L)  # B, C2 * window_size * window_size, L
    return x


@torch.inference_mode()
def _dino_features(
    data: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    window_size: tuple[int, int] | int | None = DEFAULT_WINDOW_SIZE,
) -> NDArray[np.float16]:
    """Extract patch features from a tomogram using a DINOv2 model. Uses windows to fit everything in memory.

    Args:
        data (torch.Tensor): The input data tensors containing the tomogram's data.
        model (nn.Module): The pre-loaded DINOv2 model used for feature extraction.
        batch_size (int): The maximum number of 2D slices of a tomograms processed in each batch.
        window_size (Optional[tuple[int, int] | int]): The size of the windows to use for feature extraction.

    Returns:
        NDArray[np.float16]: A numpy array containing the extracted features in reduced precision.
    """

    if window_size is None:
        window_size = (DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE)
    elif isinstance(window_size, int):
        window_size = (window_size, window_size)
    C, H, W = data.shape[1:]
    ph, pw = (
        H // DINO_PATCH_SIZE,
        W // DINO_PATCH_SIZE,
    )  # the number of patches extracted per slice
    window_size = (min(window_size[0], H), min(window_size[1], W))
    stride = tuple(
        (ws // 2) // DINO_PATCH_SIZE * DINO_PATCH_SIZE for ws in window_size
    )  # 50% overlap, multiple of 14
    fold = nn.Fold(
        output_size=(ph, pw),
        kernel_size=tuple(ws // DINO_PATCH_SIZE for ws in window_size),
        stride=tuple(s // DINO_PATCH_SIZE for s in stride),
    )
    unfold = nn.Unfold(
        kernel_size=window_size,
        stride=stride,
    )
    # divisor tensor for averaging overlapping patches
    divisor = torch.ones(1, 1, H, W).cuda()
    divisor = unfold(divisor)
    divisor, B, L = _folded_to_patch(divisor, 1, window_size)
    divisor = F.max_pool2d(
        divisor, DINO_PATCH_SIZE, stride=DINO_PATCH_SIZE
    )  # simulate patch extraction
    divisor = divisor.flatten(2).transpose(1, 2)  # B, L, 1
    divisor = _patch_to_folded(divisor, B, L)
    divisor = fold(divisor)

    # Fold data into overlapping windows
    data = unfold(data)  # B, C * window_size * window_size, L
    data, B, L = _folded_to_patch(
        data, C, window_size
    )  # B * L, C, window_size, window_size
    # Calculate features in batches
    all_features = []
    num_batches = (len(data) + batch_size - 1) // batch_size
    for i, batch_idx in enumerate(range(0, len(data), batch_size)):
        # Check for overflows
        if batch_idx + batch_size > len(data):
            batch_size = len(data) - batch_idx
        logging.debug(
            "Processing batch %d/%d: %d -> %d",
            i + 1,
            num_batches,
            batch_idx,
            (batch_idx + batch_size),
        )
        vec = data[batch_idx : batch_idx + batch_size].cuda().float()
        # Convert to RGB by repeating channels
        if vec.shape[1] == 1:
            vec = vec.repeat(1, 3, 1, 1)
        # Extract features for each patch
        patch_features = model.forward_features(vec)[  # type: ignore
            "x_norm_patchtokens"
        ]  # B', ph * pw, C2
        all_features.append(patch_features.half())
    features = torch.cat(all_features, dim=0)  # B * L, ph * pw, C2
    features = _patch_to_folded(features, B, L)  # B, C2 * ph * pw, L
    features = fold(features) / divisor  # B, C2, ph, pw
    features = features.permute([1, 0, 2, 3]).contiguous()  # C2, D, W, H
    features = features.cpu().numpy()
    return features


def _save_data(
    data: dict[str, NDArray[np.uint8]],
    features: NDArray[np.float16],
    tomo_name: str,
    dst_dir: Path,
) -> None:
    """Save extracted features to a specified directory as an .hdf file, copying the source tomogram data."""

    dst_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dst_dir / tomo_name, "w") as fh:
        for key in data:
            if key != "data":
                if "labels" not in fh:
                    fh.create_group("labels")
                fh["labels"].create_dataset(key, data=data[key], shape=data[key].shape, dtype=data[key].dtype, compression="gzip")  # type: ignore
            else:
                fh.create_dataset(
                    "data",
                    data=data[key],
                    shape=data[key].shape,
                    dtype=data[key].dtype,
                    compression="gzip",
                )
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

    dataset = instantiate(datamodule.dataset, data_root=tomo_dir)(
        records=records
    )
    dataloader = instantiate(datamodule.dataloader)(dataset=dataset)

    for i, x in track(
        enumerate(dataloader),
        description=f"[green]Computing features for {sample}",
        total=len(dataloader),
    ):
        features = _dino_features(x, model, batch_size)

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
    window_size: int | None = DEFAULT_WINDOW_SIZE,
    visualize: bool = False,
) -> None:
    """Run DINO feature extraction on the specified training data, and saves the results as .hdf files.
    The saved result file will contain `data`, `dino_features`, and any labels present in the source tomogram in the `labels/` group.

    Args:
        train_data (list[Path]): List of paths to the training tomograms.
        result_dir (Path): Directory where the results will be saved.
        batch_size (int): Number of samples to process in each batch.
        window_size (Optional[int], optional): Size of the sliding window for feature extraction. If None, uses the default size.
        visualize (bool, optional): Whether to visualize the extracted features. Defaults to False.
    """

    ## Setup hydra config
    with initialize(
        version_base="1.2",
        config_path="../configs",
        job_name="cryovit_features",
    ):
        cfg = compose(
            config_name="dino_features",
            overrides=[
                f"batch_size={batch_size}",
                "sample=null",
                "export_features=False",
                "datamodule/dataset=file",
            ],
        )
    cfg.paths.model_dir = Path(__file__).parent.parent / "foundation_models"

    ## Load DINOv2 model
    torch.hub.set_dir(cfg.dino_dir)
    model = torch.hub.load(*dino_model, verbose=False).cuda()  # type: ignore
    model.eval()

    ## Setup dataset
    assert (
        len(train_data) > 0
    ), "No valid tomogram files found in the specified training data path."
    train_file_datas = [FileData(tomo_path=f) for f in train_data]
    dataset = instantiate(
        cfg.datamodule.dataset, input_key=None, label_key=None
    )(train_file_datas, for_dino=True)
    dataloader = instantiate(cfg.datamodule.dataloader)(dataset=dataset)

    result_list = [result_dir / f"{f.stem}.hdf" for f in train_data]

    ## Iterate through dataloader and extract features
    try:
        for i, x in track(
            enumerate(dataloader),
            description="[green]Computing DINO features for training data",
            total=len(dataloader),
        ):
            features = _dino_features(
                x.data, model, cfg.batch_size, window_size=window_size
            )

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
            f"Ran out of GPU memory during DINO feature extraction. Try reducing the batch size or window size. Current batch size is {cfg.batch_size} and window size is {window_size}."
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

    src_dir = cfg.paths.data_dir / cfg.paths.tomo_name
    dst_dir = cfg.paths.data_dir / cfg.paths.feature_name
    csv_dir = cfg.paths.data_dir / cfg.paths.csv_name
    image_dir = cfg.paths.exp_dir / "dino_images"
    sample_names = (
        [cfg.sample.name]
        if cfg.sample is not None
        else [s for s in samples if (src_dir / s).exists()]
    )

    torch.hub.set_dir(cfg.dino_dir)
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
            image_dir if cfg.export_features else None,
        )
