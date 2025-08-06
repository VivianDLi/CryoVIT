import logging
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

class TestPredictionWriter(Callback):
    """Callback to write predictions to disk during model evaluation."""

    def __init__(self, results_dir: Path, label_key: str) -> None:
        """Creates a callback to save predictions on the test data.

        Args:
            results_dir (Path): directory in which the predictions should be saved.
        """
        self.results_dir = results_dir
        self.label_key = label_key

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handles the end of a test batch to save outputs.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The module being tested.
            outputs (STEP_OUTPUT | None): Outputs from the test batch.
            batch (Any): The batch of data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader.
        """
        output_file = self.results_dir / outputs["sample"] / outputs["tomo_name"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Binary classify predictions into segmentations
        data = outputs["data"].cpu().numpy()
        preds = outputs["preds"].cpu().numpy()
        labels = outputs["label"].cpu().numpy()
        segs = np.where(preds > 0.5, 1, 0).astype(np.uint8)
        
        with h5py.File(output_file, "a") as fh:
            if "data" not in fh:
                fh.create_dataset("data", data=data)
            pred_key = "predictions/" + self.label_key
            if pred_key not in fh:
                fh.create_dataset(pred_key, data=segs, compression="gzip")
            label_key = "labels/" + self.label_key
            if label_key not in fh:
                fh.create_dataset(label_key, data=labels, compression="gzip")
        
class CsvWriter(Callback):
    """Callback to save model performance metrics to a .csv."""
    
    def __init__(self, csv_result_path: Path) -> None:
        """Creates a callback to save performance metrics on the test data.
        
        Args:
            csv_result_path (Path): .csv file in which metrics should be saved.
        """
        self.csv_result_path = csv_result_path
        
    def _create_metric_df(self, sample: str, tomo_name: str, split_id: Optional[int] = None, **metrics) -> pd.DataFrame:
        result_dict = {"sample": sample, "tomo_name": tomo_name}
        for name, value in metrics:
            result_dict[name] = value
        if split_id is not None:
            result_dict["split_id"] = split_id
        
        return pd.DataFrame(result_dict)
        
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Handles the end of a test batch to save metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The module being tested.
            outputs (STEP_OUTPUT | None): Outputs from the test batch.
            batch (Any): The batch of data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader.
        """
        # Get metric results
        metric_names = [m for m in outputs["metrics"]]
        sample, tomo_name, split_id = outputs["sample"], outputs["tomo_name"], outputs["split_id"] if "split_id" in outputs else None
        # Create results .csv if it doesn't exist
        column_names = ["sample", "tomo_name"] + metric_names
        if split_id is not None:
            column_names += ["split_id"]
        if not self.csv_result_path.exists():
            results_df = pd.DataFrame(columns=column_names)
        else:
            results_df = pd.read_csv(self.csv_result_path)
        # Warn if row already exists and remove (i.e., replace)
        matching_rows = (results_df["tomo_name"] == tomo_name) & (results_df["sample"] == sample)
        if split_id is not None:
            matching_rows = matching_rows & (results_df["split_id"] == split_id)
        if len(matching_rows) > 0:
            logging.warning(f"Data with sample {sample}, name {tomo_name}, and split {split_id} already has an entry. Replacing...")
            results_df = results_df[~matching_rows]
        # Add metrics to df
        metrics_df = self._create_metric_df(sample, tomo_name, split_id, outputs["metrics"])
        results_df = pd.concat([results_df, metrics_df], ignore_index=True)
        results_df.to_csv(self.csv_result_path, mode="w", index=False)
            