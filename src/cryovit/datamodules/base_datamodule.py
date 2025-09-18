"""Module defining base data loading functionality for CryoVIT experiments."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cryovit.datamodules.utils import collate_fn


class BaseDataModule(LightningDataModule, ABC):
    """Base module defining common functions for creating data loaders for experiments."""

    def __init__(
        self,
        split_file: Path,
        dataset_fn: Callable,
        dataloader_fn: Callable,
        **kwargs,
    ) -> None:
        """Initializes the BaseDataModule with dataset parameters, a dataloader function, and a path to the split file.

        Args:
            split_file (Path): The path to the .csv file containing data splits.
            dataset_fn (Callable): Function to create a Dataset from a dataframe of records.
            dataloader_fn (Callable): Function to create a DataLoader from a dataset.
        """

        super().__init__()
        self.dataset_fn = dataset_fn
        self.dataloader_fn = dataloader_fn
        self.split_file = (
            split_file if isinstance(split_file, Path) else Path(split_file)
        )
        self.record_df = pd.read_csv(self.split_file)

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

    def train_dataloader(self) -> DataLoader:
        """Creates DataLoader for training data.

        Returns:
            DataLoader: A DataLoader instance for training data.
        """

        records = self.train_df()
        if records.empty:
            raise ValueError(
                "No training data found in the provided split file."
            )
        dataset = self.dataset_fn(records, train=True)
        return self.dataloader_fn(dataset, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self) -> DataLoader:
        """Creates DataLoader for validation data.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """

        records = self.val_df()
        if records.empty:
            raise ValueError(
                "No validation data found in the provided split file."
            )
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(
            dataset, shuffle=False, collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Creates DataLoader for testing data.

        Returns:
            DataLoader: A DataLoader instance for testing data.
        """

        records = self.test_df()
        if records.empty:
            raise ValueError(
                "No testing data found in the provided split file."
            )
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(
            dataset, shuffle=False, collate_fn=collate_fn
        )

    def predict_dataloader(self) -> DataLoader:
        """Creates DataLoader for prediction data.

        Returns:
            DataLoader: A DataLoader instance for prediction data.
        """

        records = self.predict_df()
        if records.empty:
            raise ValueError(
                "No prediction data found in the provided split file."
            )
        dataset = self.dataset_fn(records, train=False)
        return self.dataloader_fn(
            dataset, shuffle=False, collate_fn=collate_fn
        )
