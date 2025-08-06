"""Script for extracting DINOv2 features from tomograms."""

from pathlib import Path
from typing import Optional

from hydra.utils import instantiate
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from umap import UMAP
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from numpy.typing import NDArray
from rich.progress import track

from cryovit.config import DinoFeaturesConfig, BaseDataModule, samples, tomogram_exts


torch.set_float32_matmul_precision("high")  # ensures tensor cores are used
dino_model = ("facebookresearch/dinov2", "dinov2_vitg14")  # the giant variant of DINOv2

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
        batch_size (int): The number of 2D slices of a tomograms processed in each batch.

    Returns:
        NDArray[np.float16]: A numpy array containing the extracted features in reduced precision.
    """
    data = data.cuda()
    w, h = np.array(data.shape[2:]) // 14  # the number of patches extracted per slice
    all_features = []

    for i in range(0, len(data), batch_size):
        # Check for overflows
        if i + batch_size > len(data):
            batch_size = len(data) - i
        vec = data[i : i + batch_size]
        features = model.forward_features(vec)["x_norm_patchtokens"]
        features = features.reshape(features.shape[0], w, h, -1)
        features = features.permute([3, 0, 1, 2]).contiguous()
        features = features.to("cpu").half().numpy()
        all_features.append(features)

    return np.concatenate(all_features, axis=1)

def _save_data(
    data: NDArray[np.float32],
    features: NDArray[np.float16],
    tomo_name: str,
    dst_dir: Path,
) -> None:
    """Save extracted features to a specified directory, and copy the source tomogram file.

    Args:
        features (NDArray[np.float16]): Extracted features to be saved.
        tomo_name (str): The name of the tomogram file.
        src_dir (Path): The source directory containing the original tomogram file.
        dst_dir (Path): The destination directory where the tomogram and features are saved.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dst_dir / tomo_name, "r+") as fh:
        fh.create_dataset("data", data=data)
        fh.create_dataset("dino_features", data=features)  # save the features

def _calculate_pca(features: NDArray[np.float16]) -> NDArray:
    features = features.astype(np.float32) # PCA expects float32
    x = features.transpose((1, 2, 3, 0))
    x = x.reshape((-1, x.shape[-1])) # N, C
    # Reduce dimensionality to 3 colors
    pca = PCA(n_components=1024)
    x = pca.fit_transform(x)
    umap = UMAP(n_components=3, verbose=True, n_jobs=16)
    umap.fit(x)
    # Upscale features
    features = F.interpolate(torch.from_numpy(features), scale_factor=2, mode="bicubic")
    D, W, H = features.shape[1:] # D, W, H
    features = features.permute(1, 2, 3, 0).contiguous()
    features = features.view(-1, features.shape[-1]).numpy()
    features = pca.transform(features)
    features = umap.transform(features)
    return features.reshape(D, W, H, 3)

def _color_features(features: NDArray, alpha: float = 0.0):
    # Normalize
    features = features - features.min(axis=(0, 1, 2))
    features = features / features.max(axis= (0, 1, 2))
    
    # Normalize colors
    hsv = rgb_to_hsv(features)
    hsv[..., 1] = 0.9
    hsv[..., 2] = 0.75
    hsv[..., 0] = (alpha + hsv[..., 0]) % 1.0 # alpha = 0
    rgb = hsv_to_rgb(hsv)
    rgb = (255 * rgb).astype(np.uint8)
    
    # Upscale to full image size
    rgb = np.repeat(rgb, 8, axis=1)
    rgb = np.repeat(rgb, 8, axis=2)
    return rgb

def _export_pca(
    data: NDArray[np.float32],
    features: NDArray[np.float32],
    tomo_name: str,
    result_dir: Path,
) -> None:
    """Extract PCA colormap from features and save to a specified directory."""
    features = _calculate_pca(features)
    features = _color_features(features)
    
    # Save as Images
    image_dir = result_dir / tomo_name
    image_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in range(data.shape[0], step=10):
        img_path = image_dir / f"{idx}.png"
        
        f_img = Image.fromarray(features[idx][::-1])
        d_img = Image.fromarray(data[idx][::-1])
    
        img = Image.new("RGB", (2 * f_img.size[0], f_img.size[1])) # concat images
        img.paste(d_img)
        img.paste(f_img, box=(d_img.size[0], 0))
        img.save(img_path)

def _process_sample(src_dir: Path, dst_dir: Path, csv_dir: Path, dino_dir: Path, sample: str, datamodule: BaseDataModule, batch_size: int, image_dir: Optional[Path]):
    """Process all tomograms in a single sample by extracting and saving their DINOv2 features."""
    tomo_dir = src_dir / sample
    result_dir = dst_dir / sample
    csv_file = csv_dir / f"{sample}.csv"
    # If no .csv file, use all tomograms in the dataset
    if not csv_file.exists():
        records = list([f.name for f in tomo_dir.glob("*") if f.suffix in tomogram_exts])
    else:
        records = pd.read_csv(csv_file)["tomo_name"].to_list()

    dataset = instantiate(datamodule.dataset, data_root=tomo_dir)(records)
    dataloader = instantiate(datamodule.dataloader)(dataset)

    torch.hub.set_dir(dino_dir)
    model = torch.hub.load(*dino_model, verbose=False).cuda()
    model.eval()

    for i, x in track(
        enumerate(dataloader),
        description=f"[green]Computing features for {sample}",
        total=len(dataloader),
    ):
        features = _dino_features(x, model, batch_size)
        
        with h5py.File(tomo_dir / records[i]) as fh:
            data = fh["data"][()]
        _save_data(data, features, records[i], result_dir)
        # Save PCA calculation of features
        if image_dir is not None:
            _export_pca(data, features.astype(np.float32), records[i][:-4], image_dir / sample)

def run_dino(cfg: DinoFeaturesConfig) -> None:
    """Process all tomograms in a sample or samples by extracting and saving their DINOv2 features."""
    src_dir = cfg.paths.data_dir / cfg.paths.tomo_name
    dst_dir = cfg.paths.data_dir / cfg.paths.feature_name
    csv_dir = cfg.paths.data_dir / cfg.paths.csv_name
    image_dir = cfg.paths.exp_dir / "dino_images"
    sample_names = cfg.sample if cfg.sample is not None else [s for s in samples if (src_dir / s).exists()]

    for sample_name in sample_names:
        _process_sample(
            src_dir,
            dst_dir,
            csv_dir,
            cfg.dino_dir,
            sample_name,
            cfg.datamodule,
            cfg.batch_size,
            image_dir if cfg.export_features else None
        )
