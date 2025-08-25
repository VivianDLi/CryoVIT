"""Extract visualizations of DINO features with PCA."""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from umap import UMAP

from cryovit.config import tomogram_exts
from cryovit.types import DinoFeaturesData, FloatTomogramData, IntTomogramData

def _calculate_pca(features: DinoFeaturesData) -> FloatTomogramData:
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

def _color_features(features: FloatTomogramData, alpha: float = 0.0) -> IntTomogramData:
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

def export_pca(
    data: FloatTomogramData,
    features: FloatTomogramData,
    tomo_name: str,
    result_dir: Path,
) -> None:
    """Extract PCA colormap from features and save to a specified directory."""
    features = _calculate_pca(features)
    features = _color_features(features)
    
    # Save as Images
    image_dir = result_dir / tomo_name
    image_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in np.arange(data.shape[0], step=10):
        img_path = image_dir / f"{idx}.png"
        
        f_img = Image.fromarray(features[idx][::-1])
        d_img = Image.fromarray(data[idx][::-1])
    
        img = Image.new("RGB", (2 * f_img.size[0], f_img.size[1])) # concat images
        img.paste(d_img)
        img.paste(f_img, box=(d_img.size[0], 0))
        img.save(img_path)


def process_samples(exp_dir: Path, result_dir: Path):
    result_dir.mkdir(parents=True, exist_ok=True)
    samples = [s.name for s in exp_dir.iterdir() if s.is_dir()]
    
    for sample in samples:
        logging.info("Processing sample %s", sample.name)
        tomo_dir = exp_dir / sample
        tomo_names = [f.name for f in tomo_dir.glob("*") if f.suffix in tomogram_exts]
        for tomo_name in tqdm(tomo_names):
            with h5py.File(tomo_dir / tomo_name) as fh:
                data = fh["data"][()].astype(np.float32)
                features = fh["dino_features"][()].astype(np.float32)
                export_pca(data, features, tomo_name[:-4], result_dir / sample.name)
