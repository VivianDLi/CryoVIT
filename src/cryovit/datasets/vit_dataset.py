"""Dataset class for loading tomograms for DINOv2 models."""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from cryovit.config import DINO_PATCH_SIZE

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VITDataset(Dataset):
    """Dataset class for Vision Transformer models, loading and processing tomograms."""

    def __init__(
        self, data_root: Path, use_sam: bool, records: list[str]
    ) -> None:
        """Initializes a dataset object to load tomograms, applying normalization and resizing for DINOv2 models.

        Args:
            root (Path): Root directory where tomogram files are stored.
            use_sam (bool): Whether to apply preprocessing for the SAM model or DINOv2 model.
            records (list[str]): A list of strings representing paths to tomogram files in the root directory.
        """

        self.root = (
            data_root if isinstance(data_root, Path) else Path(data_root)
        )
        self.use_sam = use_sam
        self.records = records
        self.transform = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self._printed_resize_warning = False

    def __len__(self) -> int:
        """Returns the number of tomograms in the dataset."""

        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a preprocessed tomogram tensor from the dataset by index.

        Args:
            idx (int): Index of the tomogram to retrieve.

        Returns:
            torch.Tensor: A tensor representing the normalized and resized tomogram.

        Raises:
            IndexError: If the index is out of the dataset's bounds.
        """

        if idx >= len(self):
            raise IndexError

        record = self.records[idx]
        data = self._load_tomogram(record)
        return (
            self._sam_transform(data)
            if self.use_sam
            else self._dino_transform(data)
        )

    def _load_tomogram(self, record: str) -> NDArray[np.float32]:
        """Loads a tomogram from disk, assuming it is stored as an .hdf file in a `data` key.

        Args:
            record (str): The file path to the tomogram relative to the root directory.

        Returns:
            NDArray[np.uint8]: The loaded tomogram as a numpy array.
        """

        tomo_path = self.root / record

        with h5py.File(tomo_path) as fh:
            data: np.ndarray = fh["data"][()]  # type: ignore

        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        return data

    def _dino_transform(self, data: NDArray[np.float32]) -> torch.Tensor:
        """Applies normalization and resizing transformations to the tomogram.

        Args:
            data (NDArray[np.float32]): The loaded tomogram data as a numpy array.

        Returns:
            torch.Tensor: The transformed data as a PyTorch tensor.
        """

        scale = (DINO_PATCH_SIZE / 16, DINO_PATCH_SIZE / 16)
        _, h, w = data.shape
        # Resize height and width to be multiples of 16
        H = int(np.ceil(h / 16) * 16)
        W = int(np.ceil(w / 16) * 16)
        if h != H or w != W:
            if not self._printed_resize_warning:
                logging.warning(
                    "Resizing tomogram from %s to %s", (h, w), (H, W)
                )
                self._printed_resize_warning = True
            data = np.pad(data, ((0, 0), (0, H - h), (0, W - w)), mode="edge")
            h, w = H, W
        assert (
            h % 16 == 0 and w % 16 == 0
        ), f"Invalid height: {h} or width: {w}"

        np_data = np.expand_dims(data, axis=1)  # D, C, H, W (i.e., B, C, H, W)
        np_data = np.repeat(np_data, 3, axis=1)

        torch_data = torch.from_numpy(
            np_data
        ).float()  # data expected to be float already, [0-1]
        return F.interpolate(torch_data, scale_factor=scale, mode="bicubic")

    def _sam_transform(self, data: NDArray[np.float32]) -> torch.Tensor:
        """Applies normalization and resizing transformations to the tomogram for SAM model.

        Args:
            data (NDArray[np.float32]): The loaded tomogram data as a numpy array.

        Returns:
            torch.Tensor: The transformed data as a PyTorch tensor.
        """

        _, h, w = data.shape
        np_data = np.expand_dims(data, axis=(0, 2))  # B, D, C, H, W
        np_data = np.repeat(np_data, 3, axis=2)

        torch_data = torch.from_numpy(
            np_data
        ).float()  # data expected to be float already, [0-1]
        return torch_data
