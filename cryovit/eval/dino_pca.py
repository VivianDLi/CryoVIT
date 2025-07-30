import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from numpy.typing import NDArray
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from umap import UMAP

from cryovit.config import Sample


def fit_pca(features: NDArray) -> NDArray:
    features = torch.from_numpy(features)
    x = features.permute(1, 2, 3, 0).contiguous()

    x = x.view(-1, x.shape[-1]).numpy()
    pca = PCA(n_components=1024)
    x = pca.fit_transform(x)

    reducer = UMAP(n_components=3, verbose=True, n_jobs=16)
    reducer.fit(x)

    features = F.interpolate(features, scale_factor=2, mode="bicubic")
    shape = features.shape[1:]

    features = features.permute(1, 2, 3, 0).contiguous()
    features = features.view(-1, features.shape[-1]).numpy()

    features = pca.transform(features)
    features = reducer.transform(features)

    return features.reshape(*shape, 3)


def color_features(features: NDArray, alpha) -> Image:
    features = features - features.min(axis=(0, 1, 2))
    features = features / features.max(axis=(0, 1, 2))

    hsv = rgb_to_hsv(features)
    hsv[..., 1] = 0.9
    hsv[..., 2] = 0.75
    hsv[..., 0] = (alpha + hsv[..., 0]) % 1.0

    rgb = hsv_to_rgb(hsv)
    rgb = (255 * rgb).astype(np.uint8)

    rgb = np.repeat(rgb, 8, axis=1)
    rgb = np.repeat(rgb, 8, axis=2)
    return rgb


def save_features(f_img: Image, d_img: Image, idx: int, tomo_name: str, sample: Sample):
    img_dir = Path("cryovit/eval/images") / sample.name
    os.makedirs(img_dir, exist_ok=True)

    img = Image.new("RGB", (2 * f_img.size[0], f_img.size[1]))
    img.paste(d_img)
    img.paste(f_img, box=(d_img.size[0], 0))
    img.save(img_dir / f"{tomo_name}_{idx}.png")


def process_sample(sample: Sample, root: Path):
    tomo_root = root / "dino_features" / sample.name
    csv_path = root / "csv" / f"{sample.name}.csv"

    for j, row in tqdm(pd.read_csv(csv_path).iterrows()):
        with h5py.File(tomo_root / row["tomo_name"]) as fh:
            features = fh["dino_features"][()].astype(np.float32)
            features = fit_pca(features)
            features = color_features(features, alpha=0)
            indices = np.arange(fh["data"].shape[0], step=10)

            for i, idx in enumerate(indices):
                f_img = Image.fromarray(features[idx][::-1])
                d_img = Image.fromarray(fh["data"][idx][::-1])
                save_features(f_img, d_img, idx, row["tomo_name"], sample)

        if j == 15:
            break


if __name__ == "__main__":
    root = Path(sys.argv[1])

    samples = [sample for sample in Sample]

    for sample in samples[4:]:
        print("Processing sample", sample.name)
        process_sample(sample, root)
