"""Dataset class for loading DINOv2 features for CryoVIT models."""

from pathlib import Path
from typing import Any, Optional
from typing import Dict
from typing import List

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from cryovit.config import tomogram_exts

class TomoDataset(Dataset):
    """A dataset class for handling and preprocessing tomographic data for CryoVIT models."""

    def __init__(
        self,
        records: Optional[pd.DataFrame],
        input_key: str,
        label_key: str,
        split_key: str,
        data_root: Path,
        aux_keys: List[str] = [],
        train: bool = False,
        predict: bool = False
    ) -> None:
        """Creates a new TomoDataset object.

        Args:
            records (pd.DataFrame): A DataFrame containing records of tomograms.
            input_key (str): The key in the HDF5 file to access input features.
            label_key (str): The key in the HDF5 file to access labels.
            data_root (Path): The root directory where the tomograms are stored.
            train (bool): Flag to determine if the dataset is for training (enables transformations).
            aux_keys (List[str]): Additional keys for auxiliary data to load from the HDF5 files.
        """
        self.records = self._read_records(records)
        self.input_key = input_key
        self.label_key = label_key
        self.split_key = split_key
        self.aux_keys = aux_keys
        self.data_root = data_root
        self.train = train
        self.predict = predict

    def __len__(self) -> int:
        """Returns the total number of tomograms in the dataset."""
        return len(self.records)

    def __getitem__(self, idx) -> Dict[str, Any]:
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

        record = self.records.iloc[idx]
        data = self._load_tomogram(record)

        if self.train:
            self._random_crop(data)

        return data

    def _read_records(self, records: Optional[pd.DataFrame]) -> pd.DataFrame:
        if records is not None:
            return records
        
        # Should only run for prediction datasets with no split file
        records = {
            "sample": [],
            "tomo_name": [],
        }
        # Read samples and files from directory
        samples = [f.name for f in self.data_root.glob("*") if f.is_dir()]
        for sample in samples:
            tomo_dir = self.data_root / sample
            for tomo_name in [t.name for t in tomo_dir.glob("*") if t.suffix in tomogram_exts]:
                records["sample"].append(sample)
                records["tomo_name"].append(tomo_name)
                
        records_df = pd.DataFrame(records)
        return records_df

    def _load_tomogram(self, record: pd.Series) -> Dict[str, Any]:
        """Loads a single tomogram based on the record information.

        Args:
            record (pd.Series): A series containing the sample and tomogram names.

        Returns:
            data (Dict[str, Any]): A dictionary with input data, label, and any auxiliary data.
        """
        tomo_path = self.data_root / record["sample"] / record["tomo_name"]

        # sample, tomo_name, split_id, input, label

        data = {"sample": record["sample"], "tomo_name": record["tomo_name"]}
        if self.split_key in record.index:
            data["split_id"] = record[self.split_key]

        with h5py.File(tomo_path) as fh:
            data["input"] = fh[self.input_key][()]
            data["label"] = fh[self.label_key][()]
            data |= {key: fh[key][()] for key in self.aux_keys}

        return data

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
