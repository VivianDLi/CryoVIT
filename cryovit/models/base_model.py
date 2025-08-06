"""Base Model class for 3D Tomogram Segmentation."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Literal
from typing import Dict
from typing import List
from typing import Tuple

from hydra.utils import instantiate
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Metric


class BaseModel(LightningModule, ABC):
    """Base model with configurable loss functions and metrics."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        losses: Tuple[Dict],
        metrics: Tuple[Dict],
    ) -> None:
        """Initializes the BaseModel with specified learning rate, weight decay, loss functions, and metrics.

        Args:
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay factor for AdamW optimizer.
            losses (List[Dict]): List of loss function configs instances for training, validation, and testing.
            metrics (List[Dict]): List of metric function configs for training, validation, and testing.
        """
        super(BaseModel, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        self.configure_losses(losses)
        self.configure_metrics(metrics)

    def configure_optimizers(self) -> Optimizer:
        """Configures the optimizer with the initialization parameters."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
    def configure_losses(self, losses: List[Dict]) -> Dict[str, Callable]:
        self.loss_fns = {
            l["name"]: instantiate(l) for l in losses
        }
    
    def configure_metrics(self, metrics: List[Dict]) -> Dict[str, Dict[str, Callable]]:
        self.metric_fns = nn.ModuleDict(
            {
                "TRAIN": nn.ModuleDict({m["name"]: instantiate(m) for m in metrics}),
                "VAL": nn.ModuleDict({m["name"]: instantiate(m) for m in metrics}),
                "TEST": nn.ModuleDict({m["name"]: instantiate(m) for m in metrics}),
            }
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Logs gradient norms just before the optimizer updates weights."""
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def _masked_predict(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs prediction while applying a mask to the inputs and labels based on the label value."""
        data = batch["input"]
        y_true = batch["label"]

        y_pred_full = self(data)
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

    def _do_step(self, batch: Dict[str, Tensor], batch_idx: int, prefix: Literal["train", "val", "test"]) -> float:
        """Processes a single batch of data, computes the loss and updates metrics."""
        y_pred, y_true, _ = self._masked_predict(batch)

        losses = self.compute_losses(y_pred, y_true)

        for metric_fn in self.metric_fns[prefix.upper()].values():
            metric_fn(y_pred, y_true)

        self.log_stats(losses, prefix)
        return losses["total"]

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> float:
        """Processes one batch during training."""
        return self._do_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> float:
        """Processes one batch during validation."""
        return self._do_step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        """Processes one batch during testing, captures predictions, and computes losses and metrics.

        Args:
            batch (Dict[str, Tensor]): The batch of data being processed.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, Any]: A dictionary containing test results and metrics for this batch.
        """
        y_pred, y_true, y_pred_full = self._masked_predict(batch)
        # Normalize to [0-1]
        y_pred_full = (y_pred_full - y_pred_full.min()) / (y_pred_full.max() - y_pred_full.min())

        results = {
            "sample": batch["sample"],
            "tomo_name": batch["tomo_name"],
            "data": batch["data"].cpu().numpy(),
            "label": batch["label"].cpu().numpy(),
            "preds": y_pred_full.cpu().numpy(),
        }
        if "split_id" in batch:
            results["split_id"] = batch["split_id"]
            
        results["losses"] = {k: v.item() for k, v in self.compute_losses(y_pred, y_true).items()}

        results["metrics"] = {}
        for m, m_fn in self.metric_fns["TEST"].items():
            score = m_fn(y_pred, y_true)
            results["metrics"][m] = score.item()
            m_fn.reset()

        return results
    
    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        _, _, y_pred_full = self._masked_predict(batch)
        # Normalize to [0-255]
        data = batch["data"]
        data = (255 * (data - data.min()) / (data.max() - data.min())).to(torch.uint8)
        y_pred_full = (255 * (y_pred_full - y_pred_full.min()) / (y_pred_full.max() - y_pred_full.min())).to(torch.uint8)
        
        results = {
            "sample": batch["sample"],
            "tomo_name": batch["tomo_name"],
            "data": data.cpu().numpy(),
            "preds": y_pred_full.cpu().numpy()
        }
        
        return results

    @abstractmethod
    def forward(self):
        """Should be implemented in subclass."""
        raise NotImplementedError("The forward method must be implemented by subclass.")