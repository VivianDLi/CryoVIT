"""Script for extracting DINOv2 features from tomograms."""

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

DEFAULT_WINDOW_SIZE = 630  # 720 // 16 * 14


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
        batch_size (int): The number of 2D slices of a tomograms processed in each batch.

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
    dilation = (
        max((W - 1) // window_size[1], 1),
        max((H - 1) // window_size[0], 1),
    )  # at least 1 and not equal to image size
    stride = tuple(
        (((d * ws) // 4) // DINO_PATCH_SIZE) * DINO_PATCH_SIZE
        for d, ws in zip(dilation, window_size, strict=True)
    )  # 75% overlap, multiple of 14
    fold = nn.Fold(
        output_size=(ph, pw),
        kernel_size=tuple(ws // DINO_PATCH_SIZE for ws in window_size),
        stride=tuple(s // DINO_PATCH_SIZE for s in stride),
        dilation=dilation,
    )
    unfold = nn.Unfold(
        kernel_size=window_size,
        stride=stride,
        dilation=dilation,
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
    for i in range(0, len(data), batch_size):
        # Check for overflows
        if i + batch_size > len(data):
            batch_size = len(data) - i
        logging.info(
            "Processing batch %d/%d: %d -> %d",
            (i // batch_size + 1),
            (len(data) // batch_size),
            i,
            (i + batch_size),
        )
        vec = data[i : i + batch_size].cuda().float()
        # Convert to RGB by repeating channels
        if vec.shape[1] == 1:
            vec = vec.repeat(1, 3, 1, 1)
        # Extract features for each patch
        patch_features = model.forward_features(vec)[  # type: ignore
            "x_norm_patchtokens"
        ]  # B', ph * pw, C2
        patch_features: torch.Tensor = patch_features.half()
        all_features.append(patch_features)
    features = torch.cat(all_features, dim=0)  # B * L, ph * pw, C2
    features = _patch_to_folded(features, B, L)  # B, C2 * ph * pw, L
    features = fold(features)  # B, C2, ph, pw
    features /= divisor  # average overlapping patches
    features = features.permute([1, 0, 2, 3]).contiguous()  # C2, D, W, H
    features = features.cpu().numpy()
    return features


def _save_data(
    data: dict[str, NDArray[np.uint8]],
    features: NDArray[np.float16],
    tomo_name: str,
    dst_dir: Path,
) -> None:
    """Save extracted features to a specified directory, and copy the source tomogram file.

    Args:
        data (dict[str, NDArray]): A dictionary containing tomogram data and labels.
        features (NDArray[np.float16]): Extracted features to be saved.
        tomo_name (str): The name of the tomogram file.
        src_dir (Path): The source directory containing the original tomogram file.
        dst_dir (Path): The destination directory where the tomogram and features are saved.
    """
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


def run_dino(cfg: DinoFeaturesConfig) -> None:
    """Process all tomograms in a sample or samples by extracting and saving their DINOv2 features."""
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
