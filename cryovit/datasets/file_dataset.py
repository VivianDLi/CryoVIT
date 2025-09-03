"""Dataset class for loading DINOv2 features and labels for CryoVIT user models."""

from typing import Any
from typing import Dict
from typing import List

import numpy as np
from torch.utils.data import Dataset

from cryovit.types import FileData, TomogramData
from cryovit.utils import load_data, load_labels

class FileDataset(Dataset):
    """A dataset class for handling and preprocessing tomographic data for CryoVIT models."""

    def __init__(
        self,
        files: List[FileData],
        input_key: str,
        label_key: str,
        train: bool = False,
        predict: bool = False
    ) -> None:
        """Creates a new FileDataset object.

        Args:
            records (pd.DataFrame): A DataFrame containing records of tomograms.
            input_key (str): The key in the HDF5 file to access input features.
            label_key (str): The key in the HDF5 file to access labels.
            data_root (Path): The root directory where the tomograms are stored.
            train (bool): Flag to determine if the dataset is for training (enables transformations).
            aux_keys (List[str]): Additional keys for auxiliary data to load from the HDF5 files.
        """
        self.files = files
        self.input_key = input_key
        self.label_key = label_key
        self.train = train
        self.predict = predict

    def __len__(self) -> int:
        """Returns the total number of tomograms in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> TomogramData:
        """Retrieves a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            record (Dict[str, Any]): A dictionary containing the loaded data and labels.

        Raises:
            IndexError: If index is out of the range of the dataset.
        """
        if idx >= len(self):
            raise IndexError

        file_data = self.files[idx]
        data = self._load_tomogram(file_data)

        if self.train:
            self._random_crop(data)

        return TomogramData(
            sample=file_data.sample,
            tomo_name=file_data.tomo_path.name,
            split_id=None,
            data=data["input"],
            label=data["label"],
            aux_data=None
        )

    def _load_tomogram(self, file_data: FileData) -> Dict[str, Any]:
        """Loads a single tomogram based on the record information.

        Args:
            record (pd.Series): A series containing the sample and tomogram names.

        Returns:
            data (Dict[str, Any]): A dictionary with input data, label, and any auxiliary data.
        """
        tomo_path = file_data.tomo_path
        label_path = file_data.label_path

        data = load_data(tomo_path, key=self.input_key)
        labels = load_labels(label_path, keys=file_data.labels) if label_path is not None else None
        assert data is not None, f"Failed to load data from {tomo_path}"
        if labels is not None:
            assert self.label_key in labels, f"Label key {self.label_key} not found in labels from {label_path}"
        
        data_dict = {
            "input": data,
            "label": labels[self.label_key] if labels is not None else None
        }
        return data_dict

    def _random_crop(self, data: Dict[str, Any]) -> None:
        """Applies a random crop to the input data in the record dictionary.

        Args:
            record (Dict[str, Any]): The record dictionary containing 'input' and 'label' data.
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

        data["input"] = data["input"][..., di : di + x, hi : hi + y, wi : wi + z]

        if self.input_key == "dino_features":
            hi, wi, y, z = 16 * np.array([hi, wi, y, z])

        data["label"] = data["label"][di : di + x, hi : hi + y, wi : wi + z]
