"""Dataset class for loading DINOv2 features and labels for CryoVIT user models."""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from cryovit.config import DINO_PATCH_SIZE
from cryovit.datasets.vit_dataset import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from cryovit.types import FileData, TomogramData
from cryovit.utils import load_data, load_labels


class FileDataset(Dataset):
    """A dataset class for handling and preprocessing tomographic data for CryoVIT models."""

    def __init__(
        self,
        files: list[FileData],
        input_key: str | None,
        label_key: str | None,
        train: bool = False,
        predict: bool = False,
        for_dino: bool = False,
    ) -> None:
        """Creates a new FileDataset object.

        Args:
            records (pd.DataFrame): A DataFrame containing records of tomograms.
            input_key (str): The key in the HDF5 file to access input features.
            label_key (str): The key in the HDF5 file to access labels.
            data_root (Path): The root directory where the tomograms are stored.
            train (bool): Flag to determine if the dataset is for training (enables transformations).
            aux_keys (list[str]): Additional keys for auxiliary data to load from the HDF5 files.
        """
        self.files = files
        self.input_key = input_key
        self.label_key = label_key
        self.train = train
        self.predict = predict
        self.for_dino = for_dino
        self.transform = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self._key_cache = {}
        self._printed_resize_warning = False

    def __len__(self) -> int:
        """Returns the total number of tomograms in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> TomogramData:  # type: ignore
        """Retrieves a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            record (dict[str, Any]): A dictionary containing the loaded data and labels.

        Raises:
            IndexError: If index is out of the range of the dataset.
        """
        if idx >= len(self):
            raise IndexError

        file_data = self.files[idx]
        data = self._load_tomogram(file_data)
        aux_data = {}

        if self.for_dino:
            dino_data = self._dino_transform(data["input"])
            return TomogramData(
                sample=file_data.sample,
                tomo_name=file_data.tomo_path.name,
                split_id=None,
                data=dino_data,
                label=torch.zeros(
                    data["input"].shape, dtype=torch.bool
                ),  # dummy label,
                aux_data={"data": data["input"].squeeze(0)},
            )  # type: ignore
        if self.train:
            self._random_crop(data)
        elif not self.train or self.predict:  # i.e., eval
            # Load the full tomogram as aux_data for visualization
            aux_data = {
                "data": (
                    load_data(file_data.tomo_path, key="data")[0].squeeze(0)
                    if self.input_key != "data"
                    else data["input"].squeeze(0)
                )
            }

        return TomogramData(
            sample=file_data.sample,
            tomo_name=file_data.tomo_path.name,
            split_id=None,
            data=data["input"],
            label=data["label"],
            aux_data=aux_data,
        )  # type: ignore

    def _load_tomogram(self, file_data: FileData) -> dict[str, Any]:
        """Loads a single tomogram based on the record information.

        Args:
            record (pd.Series): A series containing the sample and tomogram names.

        Returns:
            data (dict[str, Any]): A dictionary with input data, label, and any auxiliary data.
        """
        tomo_path = file_data.tomo_path
        label_path = file_data.label_path

        # Cache the label key used for each tomogram to avoid redundant reads
        if tomo_path in self._key_cache:
            data, _ = load_data(tomo_path, key=self._key_cache[tomo_path])
        else:
            data, key = load_data(tomo_path, key=self.input_key)
            self._key_cache[tomo_path] = key
        labels = (
            load_labels(
                label_path, label_keys=file_data.labels, key=self.label_key
            )
            if label_path is not None and file_data.labels is not None
            else None
        )
        assert data is not None, f"Failed to load data from {tomo_path}"

        data_dict = {
            "input": data,
            "label": (
                labels[self.label_key]
                if labels is not None and self.label_key is not None
                else None
            ),
        }
        return data_dict

    def _random_crop(self, data: dict[str, Any]) -> None:
        """Applies a random crop to the input data in the record dictionary.

        Args:
            record (dict[str, Any]): The record dictionary containing 'input' and 'label' data.
        """
        max_depth = 128
        side = 32 if self.input_key == "dino_features" else 512
        d, h, w = data["input"].shape[-3:]
        x, y, z = min(d, max_depth), side, side

        if (d, h, w) == (x, y, z):
            return  # nothing to be done

        delta_d = d - x + 1
        delta_h = h - y + 1
        delta_w = w - z + 1

        di = np.random.choice(delta_d) if delta_d > 0 else 0
        hi = np.random.choice(delta_h) if delta_h > 0 else 0
        wi = np.random.choice(delta_w) if delta_w > 0 else 0

        data["input"] = data["input"][
            ..., di : di + x, hi : hi + y, wi : wi + z
        ]

        if self.input_key == "dino_features":
            hi, wi, y, z = 16 * np.array([hi, wi, y, z])

        data["label"] = data["label"][di : di + x, hi : hi + y, wi : wi + z]

    def _dino_transform(self, data: NDArray[np.uint8]) -> torch.Tensor:
        """Applies normalization and resizing transformations to the tomogram.

        Args:
            data (NDArray[np.uint8]): The loaded tomogram data as a numpy array.

        Returns:
            torch.Tensor: The transformed data as a PyTorch tensor.
        """
        scale = (DINO_PATCH_SIZE / 16, DINO_PATCH_SIZE / 16)
        h, w = data.shape[-2:]
        # Resize height and width to be multiples of 16
        H = int(np.ceil(h / 16) * 16)
        W = int(np.ceil(w / 16) * 16)
        if h != H or w != W:
            if not self._printed_resize_warning:
                logging.warning(
                    "Resizing tomogram from %s to %s", (h, w), (H, W)
                )
                self._printed_resize_warning = True
            data = np.pad(
                data, ((0, 0), (0, 0), (0, H - h), (0, W - w)), mode="edge"
            )
            h, w = H, W
        assert (
            h % 16 == 0 and w % 16 == 0
        ), f"Invalid height: {h} or width: {w}"

        np_data = data.transpose((1, 0, 2, 3))  # D, C, H, W
        np_data = np.repeat(np_data, 3, axis=1)  # C, D, H, W

        torch_data = torch.from_numpy(
            np_data
        ).float()  # data expected to be float32, [0-1]
        torch_data: torch.Tensor = self.transform(torch_data)[
            :, [0], :, :
        ]  # D, C, H, W
        torch_data: torch.Tensor = F.interpolate(
            torch_data, scale_factor=scale, mode="bicubic"
        )
        return torch_data.to(
            torch.float16
        )  # Use half precision to save memory
