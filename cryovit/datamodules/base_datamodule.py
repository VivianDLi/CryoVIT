"""Module defining base data loading functionality for CryoVIT experiments."""

from pathlib import Path
from typing import Callable, List
from abc import ABC, abstractmethod

import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.functional import pad

from cryovit.types import TomogramData, BatchedTomogramMetadata, BatchedTomogramData

class BaseDataModule(LightningDataModule, ABC):
    """Base module defining common functions for creating data loaders."""

    def __init__(
        self,
        split_file: Path,
        dataloader_fn: Callable,
        dataset_fn: Callable,
        **kwargs
    ) -> None:
        """Initializes the BaseDataModule with dataset parameters, a dataloader function, and a path to the split file.

        Args:
            split_file (Path): The path to the CSV file containing data splits.
            dataloader_fn (Callable): Function to create a DataLoader from a dataset.
            dataset_params (Callable): Function to create a Dataset from a dataframe of records.
        """
        super().__init__()
        self.dataset_fn = dataset_fn
        self.dataloader_fn = dataloader_fn
        self.split_file = split_file if isinstance(split_file, Path) else Path(split_file)

    @abstractmethod
    def train_df(self) -> pd.DataFrame:
        """Abstract method to generate train splits."""
        raise NotImplementedError

    @abstractmethod
    def val_df(self) -> pd.DataFrame:
        """Abstract method to generate validation splits."""
        raise NotImplementedError

    @abstractmethod
    def test_df(self) -> pd.DataFrame:
        """Abstract method to generate test splits."""
        raise NotImplementedError
    
    @abstractmethod
    def predict_df(self) -> pd.DataFrame:
        """Abstract method to generate predict splits."""
        raise NotImplementedError

    def _load_splits(self, predict: bool = False) -> None:
        if not self.split_file.exists() and not predict:
            raise RuntimeError(f"Split file {self.split_file} not found")
        elif not self.split_file.exists(): # Create "splits_file" for prediction in dataset
            self.record_df = None
        else:
            self.record_df = pd.read_csv(self.split_file)

    def train_dataloader(self) -> DataLoader:
        """Creates DataLoader for training data.

        Returns:
            DataLoader: A DataLoader instance for training data.
        """
        self._load_splits()
        records = self.train_df()
        if records.empty:
            raise ValueError("No training data found in the provided split file.")
        dataset = self.dataset_fn(records, train=True)
        return self.dataloader_fn(dataset, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self) -> DataLoader:
        """Creates DataLoader for validation data.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """
        self._load_splits()
        records = self.val_df()
        if records.empty:
            raise ValueError("No validation data found in the provided split file.")
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(dataset, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self) -> DataLoader:
        """Creates DataLoader for testing data.

        Returns:
            DataLoader: A DataLoader instance for testing data.
        """
        self._load_splits()
        records = self.test_df()
        if records.empty:
            raise ValueError("No testing data found in the provided split file.")
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(dataset, shuffle=False, collate_fn=collate_fn)

    def predict_dataloader(self) -> DataLoader:
        """Creates DataLoader for prediction data.

        Returns:
            DataLoader: A DataLoader instance for prediction data.
        """
        self._load_splits(predict=True)
        records = self.predict_df()
        if records.empty:
            raise ValueError("No prediction data found in the provided split file.")
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(dataset, shuffle=False, collate_fn=collate_fn)

def collate_fn(batch: List[TomogramData]) -> BatchedTomogramData:
    """Combine multiple tomograms into a single batch with metadata."""
    # Initialize metadata
    unique_samples = {} # use dictionaries as ordered sets
    unique_names = {}
    tomo_identifiers = torch.empty(len(batch), 2, dtype=torch.long)
    split_id = torch.empty(len(batch), dtype=torch.int)
    use_splits = True
    
    tomo_sizes = torch.empty(len(batch), dtype=torch.int)
    # Get tomogram sizes
    for tomo_idx, tomo_data in enumerate(batch):
        D = tomo_data.data.shape[-3]
        tomo_sizes[tomo_idx] = D
    
    # Initialize data arrays
    max_size = tomo_sizes.max().item()
    C, _, Hp, Wp = batch[0].data.size()
    H, W = batch[0].label.size()[-2:]
    tomo_batch = torch.empty(len(batch), C, max_size, Hp, Wp, dtype=torch.float)
    labels = torch.empty(len(batch), max_size, H, W, dtype=torch.float)
    aux_data = {
        key: [] for key in batch[0].aux_data
    }
    # Loop through batch tomograms
    for tomo_idx, tomo_data in enumerate(batch):
        # Get data
        data = tomo_data.data
        label = tomo_data.label
        for key, value in tomo_data.aux_data.items():
            aux_data[key].append(value)

        # Pad data
        D = data.size()[-3]
        if D != max_size:
            padding = max_size - D
            data = pad(data, (0, 0, 0, 0, 0, padding), mode="constant", value=0)
            label = pad(data, (0, 0, 0, 0, 0, padding), mode="constant", value=-1) # ignore padding
        tomo_batch[tomo_idx] = data
        labels[tomo_idx] = label
        
        # Get metadata
        sample = tomo_data.sample
        name = tomo_data.tomo_name
        split = tomo_data.split_id
        
        unique_samples[sample] = None
        s_id = list(unique_samples).index(sample)
        unique_names[name] = None
        n_id = list(unique_names).index(name)
        tomo_identifiers[tomo_idx][0] = s_id
        tomo_identifiers[tomo_idx][1] = n_id
        if split is not None and use_splits:
            split_id[tomo_idx] = split
        else:
            use_splits = False
            
    tomo_batch = tomo_batch.permute((0, 2, 1, 3, 4)) # (B, C, D, H, W) -> (B, D, C, H, W)
    metadata = BatchedTomogramMetadata(
        samples=list(unique_samples),
        tomo_names=list(unique_names),
        unique_id=tomo_identifiers,
        split_id=split_id if use_splits else None
    )
    return BatchedTomogramData(
        tomo_batch=tomo_batch,
        tomo_sizes=tomo_sizes,
        labels=labels,
        aux_data=aux_data,
        metadata=metadata
    )