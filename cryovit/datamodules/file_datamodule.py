"""Module defining data loading functionality for running CryoViT on user datasets."""

import logging
from collections.abc import Callable
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cryovit.config import tomogram_exts
from cryovit.datamodules.base_datamodule import collate_fn
from cryovit.types import FileData


class FileDataModule(LightningDataModule):
    """Module defining common functions for creating data loaders."""

    def __init__(
        self,
        data_path: Path,
        dataset_fn: Callable,
        dataloader_fn: Callable,
        val_path: Path | None = None,
        data_labels: Path | None = None,
        val_labels: Path | None = None,
        labels: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initializes the BaseDataModule with dataset parameters, a dataloader function, and a path to the split file.

        Args:
            split_file (Path): The path to the CSV file containing data splits.
            dataloader_fn (Callable): Function to create a DataLoader from a dataset.
            dataset_params (Callable): Function to create a Dataset from a dataframe of records.
        """
        super().__init__()
        self.data_files = self._load_files(data_path, data_labels, labels)
        self.val_files = (
            self._load_files(val_path, val_labels, labels)
            if val_path is not None
            else []
        )
        self.dataset_fn = dataset_fn
        self.dataloader_fn = dataloader_fn

    @staticmethod
    def _load_files(
        data_path: Path,
        data_labels: Path | None = None,
        labels: list[str] | None = None,
    ) -> list[FileData]:
        if data_path.is_dir():
            file_paths = sorted(
                [f for f in data_path.rglob("*") if f.suffix in tomogram_exts]
            )
        elif data_path.is_file() and data_path.suffix == ".txt":
            with open(data_path) as f:
                file_paths = [Path(line.strip()) for line in f if line.strip()]
        else:
            raise ValueError(
                "Data path must be a directory or a .txt file listing data files."
            )
        assert (
            len(file_paths) != 0
        ), f"No valid tomogram files found in {data_path}."
        if data_labels is None:
            label_paths = [None] * len(file_paths)
        else:
            if data_labels.is_dir():
                label_paths = sorted(
                    [
                        f
                        for f in data_labels.rglob("*")
                        if f.suffix in tomogram_exts
                    ]
                )
            elif data_labels.is_file() and data_labels.suffix == ".txt":
                with open(data_labels) as f:
                    label_paths = [
                        Path(line.strip()) for line in f if line.strip()
                    ]
            else:
                raise ValueError(
                    "Data labels path must be a directory or a .txt file listing label files."
                )
            assert (
                len(label_paths) != 0
            ), f"No valid label files found in {data_labels}."
            assert len(file_paths) == len(
                label_paths
            ), "Number of data files must match number of label files."
            file_paths = label_paths

        files = []
        for fp, lp in zip(file_paths, label_paths, strict=True):
            if not fp.exists() or (lp is not None and not lp.exists()):
                logging.warning(
                    "File %s or label %s does not exist, skipping.", fp, lp
                )
                continue
            files.append(
                FileData(
                    tomo_path=fp,
                    label_path=lp,
                    sample=fp.parent.name,
                    labels=labels,
                )
            )
        return files

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
            logging.info("No validation data provided, using training data.")
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
        dataset = self.dataset_fn(self.data_files, train=False, predict=True)
        return self.dataloader_fn(
            dataset, shuffle=False, collate_fn=collate_fn
        )
