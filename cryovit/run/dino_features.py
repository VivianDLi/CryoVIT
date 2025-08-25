"""Script for extracting DINOv2 features from tomograms."""

from pathlib import Path
from typing import Dict, Optional, Union

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
from cryovit.types import DinoFeaturesData, IntTomogramData, LabelData
from cryovit.visualization.dino_pca import export_pca

torch.set_float32_matmul_precision("high")  # ensures tensor cores are used
dino_model = ("facebookresearch/dinov2", "dinov2_vitg14_reg")  # the giant variant of DINOv2

@torch.inference_mode()
def _dino_features(
    data: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
) -> DinoFeaturesData:
    """Extract patch features from a tomogram using a DINOv2 model.

    Args:
        data (torch.Tensor): The input data tensors containing the tomogram's data.
        model (nn.Module): The pre-loaded DINOv2 model used for feature extraction.
        batch_size (int): The number of 2D slices of a tomograms processed in each batch.

    Returns:
        NDArray[np.float16]: A numpy array containing the extracted features in reduced precision.
    """
    data = data.cuda()
    w, h = np.array(data.shape[-2:]) // 14  # the number of patches extracted per slice
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
    data: Dict[str, Union[IntTomogramData, LabelData]],
    features: IntTomogramData,
    tomo_name: str,
    dst_dir: Path,
) -> None:
    """Save extracted features to a specified directory, and copy the source tomogram file.

    Args:
        data (Dict[str, NDArray]): A dictionary containing tomogram data and labels.
        features (NDArray[np.float16]): Extracted features to be saved.
        tomo_name (str): The name of the tomogram file.
        src_dir (Path): The source directory containing the original tomogram file.
        dst_dir (Path): The destination directory where the tomogram and features are saved.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dst_dir / tomo_name, "w") as fh:
        fh.create_group("labels")
        for key in data:
            if key != "data":
                fh["labels"].create_dataset(key, data=data[key], shape=data[key].shape, dtype=data[key].dtype, compression="gzip")
            else:
                fh.create_dataset("data", data=data[key], shape=data[key].shape, dtype=data[key].dtype, compression="gzip")
        fh.create_dataset("dino_features", data=features, shape=features.shape, dtype=features.dtype)

def _process_sample(src_dir: Path, dst_dir: Path, csv_dir: Path, model: torch.nn.Module, sample: str, datamodule: BaseDataModule, batch_size: int, image_dir: Optional[Path]):
    """Process all tomograms in a single sample by extracting and saving their DINOv2 features."""
    tomo_dir = src_dir / sample
    result_dir = dst_dir / sample
    csv_file = csv_dir / f"{sample}.csv"
    # If no .csv file, use all tomograms in the dataset
    if not csv_file.exists():
        records = list([f.name for f in tomo_dir.glob("*") if f.suffix in tomogram_exts])
    else:
        records = pd.read_csv(csv_file)["tomo_name"].to_list()

    dataset = instantiate(datamodule.dataset, data_root=tomo_dir)(records=records)
    dataloader = instantiate(datamodule.dataloader)(dataset=dataset)

    for i, x in track(
        enumerate(dataloader),
        description=f"[green]Computing features for {sample}",
        total=len(dataloader),
    ):
        features = _dino_features(x, model, batch_size)
        
        data = {}
        with h5py.File(tomo_dir / records[i]) as fh:
            for key in fh.keys():
                data[key] = fh[key][()]
            
        _save_data(data, features, records[i], result_dir)
        # Save PCA calculation of features
        if image_dir is not None:
            export_pca(data["data"], features.astype(np.float32), records[i][:-4], image_dir / sample)

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
    sample_names = [cfg.sample.name] if cfg.sample is not None else [s for s in samples if (src_dir / s).exists()]

    torch.hub.set_dir(cfg.dino_dir)
    model = torch.hub.load(*dino_model, verbose=False).cuda()
    model.eval()

    for sample_name in sample_names:
        _process_sample(
            src_dir,
            dst_dir,
            csv_dir,
            model,
            sample_name,
            cfg.datamodule,
            cfg.batch_size,
            image_dir if cfg.export_features else None
        )
