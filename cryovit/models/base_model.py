"""Base Model class for 3D Tomogram Segmentation."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Literal
from typing import Dict
from typing import Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import Tensor
from torch import nn
from torch.optim import Optimizer

from cryovit.types import BatchedModelResult, BatchedTomogramData

class BaseModel(LightningModule, ABC):
    """Base model with configurable loss functions and metrics."""

    def __init__(
        self,
        input_key: str,
        lr: float,
        weight_decay: float,
        losses: Tuple[Dict],
        metrics: Tuple[Dict],
        name: str = "BaseModel",
        **kwargs
    ) -> None:
        """Initializes the BaseModel with specified learning rate, weight decay, loss functions, and metrics.

        Args:
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay factor for AdamW optimizer.
            losses (List[Dict]): List of loss function configs instances for training, validation, and testing.
            metrics (List[Dict]): List of metric function configs for training, validation, and testing.
        """
        super(BaseModel, self).__init__()
        self.name = name
        self.input_key = input_key
        self.lr = lr
        self.weight_decay = weight_decay

        self.configure_losses(losses)
        self.configure_metrics(metrics)
        
        self.save_hyperparameters()

    def configure_optimizers(self) -> Optimizer:
        """Configures the optimizer with the initialization parameters."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
    def configure_losses(self, losses: Dict[str, Callable]) -> None:
        self.loss_fns = losses
    
    def configure_metrics(self, metrics: Dict) -> None:
        self.metric_fns = nn.ModuleDict(
            {
                "TRAIN": nn.ModuleDict({m: deepcopy(m_fn) for m, m_fn in metrics.items()}),
                "VAL": nn.ModuleDict({m: deepcopy(m_fn) for m, m_fn in metrics.items()}),
                "TEST": nn.ModuleDict({m: deepcopy(m_fn) for m, m_fn in metrics.items()}),
            }
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Logs gradient norms just before the optimizer updates weights."""
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def _masked_predict(
        self, batch: BatchedTomogramData
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs prediction while applying a mask to the inputs and labels based on the label value."""
        y_true = batch.labels # (B, D, H, W)

        y_pred_full = self(batch) # (B, D, H, W)
        mask = (y_true > -1.0).detach()

        y_pred = torch.masked_select(y_pred_full, mask).view(-1, 1)
        y_true = torch.masked_select(y_true, mask).view(-1, 1)

        return y_pred, y_true, y_pred_full

    def compute_losses(self, y_pred: Tensor, y_true: Tensor) -> Dict[str, Tensor]:
        losses = {k: v(y_pred, y_true) for k, v in self.loss_fns.items()}
        losses["total"] = sum(losses.values())
        return losses

    def log_stats(self, losses: Dict[str, Tensor], prefix: Literal["train", "val", "test"]) -> None:
        """Logs computed loss and metric statistics for each training or validation step."""
        # Log losses
        loss_log_dict = {f"{prefix}/loss/{k}": v for k, v in losses.items()}
        on_step = prefix == "train"
        self.log_dict(loss_log_dict, prog_bar=True, on_epoch=not on_step, on_step=on_step, batch_size=1)
        
        # Log metrics
        metric_log_dict = {}
        for m, m_fn in self.metric_fns[prefix.upper()].items():
            metric_log_dict[f"{prefix}/metric/{m}"] = m_fn
        self.log_dict(metric_log_dict, prog_bar=True, on_epoch=True, on_step=False, batch_size=1)

    def _do_step(self, batch: BatchedTomogramData, batch_idx: int, prefix: Literal["train", "val", "test"]) -> float:
        """Processes a single batch of data, computes the loss and updates metrics."""
        y_pred, y_true, _ = self._masked_predict(batch)

        losses = self.compute_losses(y_pred, y_true)

        for _, m_fn in self.metric_fns[prefix.upper()].items():
            m_fn(y_pred, y_true)

        self.log_stats(losses, prefix)
        return losses["total"]

    def training_step(self, batch: BatchedTomogramData, batch_idx: int) -> float:
        """Processes one batch during training."""
        return self._do_step(batch, batch_idx, "train")

    def validation_step(self, batch: BatchedTomogramData, batch_idx: int) -> float:
        """Processes one batch during validation."""
        return self._do_step(batch, batch_idx, "val")

    def test_step(self, batch: BatchedTomogramData, batch_idx: int) -> BatchedModelResult:
        """Processes one batch during testing, captures predictions, and computes losses and metrics.

        Args:
            batch (BatchedTomogramData): The batch of data being processed.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, Any]: A dictionary containing test results and metrics for this batch.
        """
        assert batch.aux_data is not None and "data" in batch.aux_data, "Batch aux_data must contain 'data' key for testing."
        input_data = [torch.from_numpy(batch.aux_data["data"][i]) for i in range(batch.num_tomos)]
    
        y_pred, y_true, y_pred_full = self._masked_predict(batch)

        samples, tomo_names = batch.metadata.identifiers
        split_id = batch.metadata.split_id
        data = [t_data.cpu().numpy() for t_data in input_data]
        labels = [t_labels.cpu().numpy() for t_labels in batch.labels]
        preds = [t_preds.cpu().numpy() for t_preds in y_pred_full]
        if split_id is not None:
            split_id = [s_id.item() for s_id in split_id]

        losses = {k: v.item() for k, v in self.compute_losses(y_pred, y_true).items()}
        metrics = {}
        for m, m_fn in self.metric_fns["TEST"].items():
            score = m_fn(y_pred, y_true)
            metrics[m] = score.item()
            m_fn.reset()

        return BatchedModelResult(
            num_tomos=batch.num_tomos,
            samples=samples,
            tomo_names=tomo_names,
            split_id=split_id,
            data=data,
            label=labels,
            preds=preds,
            losses=losses,
            metrics=metrics
        )
    
    def predict_step(self, batch: BatchedTomogramData, batch_idx: int) -> BatchedModelResult:
        result = self.test_step(batch, batch_idx)
        
        # Normalize to [0-255]
        for n in range(result.num_tomos):
            data = result.data[n]
            pred = result.preds[n]
            result.data[n] = (255 * (data - data.min()) / (data.max() - data.min())).astype(np.uint8)
            result.preds[n] = (255 * (pred - pred.min()) / (pred.max() - pred.min())).astype(np.uint8)
            
        return result

    @abstractmethod
    def forward(self):
        """Should be implemented in subclass."""
        raise NotImplementedError("The forward method must be implemented by subclass.")