"""Module defining data loading functionality for running CryoViT on user datasets."""

import logging
from collections.abc import Callable
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cryovit.datamodules.utils import collate_fn
from cryovit.types import FileData


class FileDataModule(LightningDataModule):
    """Module defining common functions for creating data loaders for file-based datasets."""

    def __init__(
        self,
        data_paths: list[Path],
        dataset_fn: Callable,
        dataloader_fn: Callable,
        val_paths: list[Path] | None = None,
        data_labels: list[Path] | None = None,
        val_labels: list[Path] | None = None,
        labels: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the BaseDataModule with dataset parameters, a dataloader function, and a path to the split file.

        Args:
            data_paths (list[Path]): A list of paths to the data files for training/testing/prediction.
            dataset_fn (Callable): Function to create a Dataset from a list of FileData objects.
            dataloader_fn (Callable): Function to create a DataLoader from a dataset.
            val_paths (Optional[list[Path]]): A list of paths to the data files for validation.
            data_labels (Optional[list[Path]]): A list of paths to the label files for training/testing/prediction. Should only be missing for inference.
            val_labels (Optional[list[Path]]): A list of paths to the label files for validation.
            labels (Optional[list[str]]): A list of label keys to load from the label files. Should only be missing if no labels are provided.
        """

        super().__init__()
        self.data_files = self._combine_files_and_labels(
            data_paths, data_labels, labels
        )
        self.val_files = (
            self._combine_files_and_labels(val_paths, val_labels, labels)
            if val_paths is not None
            else []
        )
        self.dataset_fn = dataset_fn
        self.dataloader_fn = dataloader_fn

    def _combine_files_and_labels(
        self,
        files: list[Path],
        labels: list[Path] | None,
        label_keys: list[str] | None,
    ) -> list[FileData]:
        """Combines data files and label files into a list of FileData objects. Replaces missing labels with None."""

        file_labels = [None] * len(files) if labels is None else labels
        if len(files) != len(file_labels):
            raise ValueError(
                "Number of data files must match number of label files."
            )
        combined = []
        for fp, lp in zip(files, file_labels, strict=True):
            if not fp.exists() or (lp is not None and not lp.exists()):
                logging.warning(
                    "File %s or label %s does not exist, skipping.", fp, lp
                )
                continue
            combined.append(
                FileData(
                    tomo_path=fp,
                    label_path=lp,
                    sample=fp.parent.name,
                    labels=label_keys,
                )
            )
        return combined

    def train_dataloader(self) -> DataLoader:
        """Creates DataLoader for training data.

        Returns:
            DataLoader: A DataLoader instance for training data.
        """

        if len(self.data_files) == 0:
            raise ValueError("No training data provided.")
        dataset = self.dataset_fn(self.data_files, train=True)
        return self.dataloader_fn(dataset, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self) -> DataLoader:
        """Creates DataLoader for validation data.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """

        if len(self.val_files) == 0:
            logging.warning(
                "No validation data provided, using training data."
            )
            val_files = self.data_files
        else:
            val_files = self.val_files
        dataset = self.dataset_fn(val_files, train=False)
        return self.dataloader_fn(
            dataset, shuffle=False, collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Creates DataLoader for testing data.

        Returns:
            DataLoader: A DataLoader instance for testing data.
        """

        if len(self.data_files) == 0:
            raise ValueError("No testing data provided.")
        dataset = self.dataset_fn(self.data_files, train=False)
        return self.dataloader_fn(
            dataset, shuffle=False, collate_fn=collate_fn
        )

    def predict_dataloader(self) -> DataLoader:
        """Creates DataLoader for prediction data.

        Returns:
            DataLoader: A DataLoader instance for prediction data.
        """

        if len(self.data_files) == 0:
            raise ValueError("No prediction data provided.")
        dataset = self.dataset_fn(self.data_files, train=False)
        return self.dataloader_fn(
            dataset, shuffle=False, collate_fn=collate_fn
        )
