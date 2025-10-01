"""Extract visualizations of DINO features with PCA."""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

from cryovit.config import tomogram_exts


def _calculate_pca(features: NDArray[np.float16]) -> NDArray[np.float32]:
    from sklearn.decomposition import PCA
    from umap import UMAP

    float_features = features.astype(np.float32)  # type: ignore # PCA expects float32
    x = float_features.transpose((1, 2, 3, 0))
    x = x.reshape((-1, x.shape[-1]))  # N, C
    # Reduce dimensionality to 3 colors
    pca = PCA(n_components=min(1024, x.shape[0]))
    x = pca.fit_transform(x)
    umap = UMAP(n_components=3, verbose=False, n_jobs=16)
    umap.fit(x)
    # Upscale features
    torch_features = F.interpolate(
        torch.from_numpy(features), scale_factor=2, mode="bicubic"
    )
    D, W, H = torch_features.shape[1:]  # D, W, H
    torch_features = torch_features.permute(1, 2, 3, 0).contiguous()
    np_features = torch_features.view(-1, torch_features.shape[-1]).numpy()
    np_features = pca.transform(np_features)
    np_features = umap.transform(np_features)
    return np_features.reshape(D, W, H, 3)  # type: ignore


def _color_features(
    features: NDArray[np.float32], alpha: float = 0.0
) -> NDArray[np.uint8]:
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    # Normalize
    features = features - features.min(axis=(0, 1, 2))
    features = features / features.max(axis=(0, 1, 2))

    # Normalize colors
    hsv = rgb_to_hsv(features)
    hsv[..., 1] = 0.9
    hsv[..., 2] = 0.75
    hsv[..., 0] = (alpha + hsv[..., 0]) % 1.0  # alpha = 0
    rgb = hsv_to_rgb(hsv)
    rgb = (255 * rgb).astype(np.uint8)

    # Upscale to full image size
    rgb = np.repeat(rgb, 8, axis=1)
    rgb = np.repeat(rgb, 8, axis=2)
    return rgb  # type: ignore


def export_pca(
    data: NDArray[np.float32],
    features: NDArray[np.float16],
    tomo_name: str,
    result_dir: Path,
    frame_id: int | None = None,
) -> None:
    """Extract PCA colormap from features and save to a specified directory."""

    from PIL import Image

    # Save as Images
    image_dir = result_dir / tomo_name
    image_dir.mkdir(parents=True, exist_ok=True)

    if frame_id is None:
        idxs = list(np.arange(0, data.shape[0], step=10, dtype=int))
    else:
        idxs = [frame_id]
    np_features = features[:, idxs]
    np_features = _calculate_pca(np_features)
    np_features = _color_features(np_features)
    for i, idx in enumerate(idxs):  # type: ignore
        img_path = image_dir / f"{idx}.png"

        data = data - data.min()
        data = data / data.max()
        int_data = (data * 255.0).astype(np.uint8)
        f_img = Image.fromarray(np_features[i][::-1])
        d_img = Image.fromarray(int_data[idx][::-1])

        img = Image.new(
            "RGB", (2 * f_img.size[0], f_img.size[1])
        )  # concat images
        img.paste(d_img)
        img.paste(f_img, box=(d_img.size[0], 0))
        logging.debug("Saving PCA visualization to %s", img_path)
        img.save(img_path)


def process_samples(
    exp_dir: Path, result_dir: Path, sample: str | None = None
):
    """Process all samples in an experiment directory and save PCA visualizations."""

    result_dir.mkdir(parents=True, exist_ok=True)
    samples = (
        [s.name for s in exp_dir.iterdir() if s.is_dir()]
        if sample is None
        else [sample]
    )
    logging.info(
        "Found %d samples in experiment directory %s: %s",
        len(samples),
        exp_dir,
        samples,
    )

    for sample in samples:
        tomo_dir = exp_dir / sample
        tomo_names = [
            f.name for f in tomo_dir.glob("*") if f.suffix in tomogram_exts
        ]
        for tomo_name in track(
            tomo_names,
            description=f"[green]Processing {sample}",
            total=len(tomo_names),
        ):
            with h5py.File(tomo_dir / tomo_name) as fh:
                data: NDArray[np.float32] = fh["data"][()]  # type: ignore
                if data.dtype == np.uint8:
                    data = data.astype(np.float32) / 255.0
                features: NDArray[np.float16] = fh["dino_features"][()].astype(np.float32)  # type: ignore
                export_pca(data, features, tomo_name[:-4], result_dir / sample)
