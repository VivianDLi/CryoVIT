"""Module defining base data loading functionality for CryoVIT experiments."""

from pathlib import Path
from typing import Callable
from abc import ABC, abstractmethod

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cryovit.datasets import TomoDataset


class BaseDataModule(LightningDataModule, ABC):
    """Base module defining common functions for creating data loaders."""

    def __init__(
        self,
        split_file: Path,
        dataloader_fn: Callable,
        dataset_fn: Callable
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
        self.split_file = split_file

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
        dataset = self.dataset_fn(records, train=True)
        return self.dataloader_fn(dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Creates DataLoader for validation data.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """
        self._load_splits()
        records = self.val_df()
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Creates DataLoader for testing data.

        Returns:
            DataLoader: A DataLoader instance for testing data.
        """
        self._load_splits()
        records = self.test_df()
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Creates DataLoader for prediction data.

        Returns:
            DataLoader: A DataLoader instance for prediction data.
        """
        self._load_splits(predict=True)
        records = self.predict_df()
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(dataset, shuffle=False)
