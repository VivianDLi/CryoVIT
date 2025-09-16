"""Base Model class for 3D Tomogram Segmentation."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import Tensor, nn
from torch.optim import Optimizer

from cryovit.types import BatchedModelResult, BatchedTomogramData


class BaseModel(LightningModule, ABC):
    """Base model with configurable loss functions and metrics."""

    def __init__(
        self,
        input_key: str,
        lr: float,
        weight_decay: float,
        losses: dict[str, Callable],
        metrics: dict[str, Callable],
        name: str = "BaseModel",
        custom_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """Initializes the BaseModel with specified learning rate, weight decay, loss functions, and metrics.

        Args:
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay factor for AdamW optimizer.
            losses (list[dict]): list of loss function configs instances for training, validation, and testing.
            metrics (list[dict]): list of metric function configs for training, validation, and testing.
        """
        super().__init__()
        self.name = name
        self.input_key = input_key
        self.lr = lr
        self.weight_decay = weight_decay

        if custom_kwargs is not None:
            for key, value in custom_kwargs.items():
                setattr(self, key, value)

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

    def configure_losses(self, losses: dict[str, Callable]) -> None:
        self.loss_fns = losses

    def configure_metrics(self, metrics: dict[str, Callable]) -> None:
        self.metric_fns = nn.ModuleDict(
            {
                "TRAIN": nn.ModuleDict({m: deepcopy(m_fn) for m, m_fn in metrics.items()}),  # type: ignore
                "VAL": nn.ModuleDict({m: deepcopy(m_fn) for m, m_fn in metrics.items()}),  # type: ignore
                "TEST": nn.ModuleDict({m: deepcopy(m_fn) for m, m_fn in metrics.items()}),  # type: ignore
            }
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Logs gradient norms just before the optimizer updates weights."""
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def _masked_predict(
        self, batch: BatchedTomogramData, use_mito_mask: bool = False  # type: ignore
    ) -> dict[str, Tensor]:
        """Performs prediction while applying a mask to the inputs and labels based on the label value."""
        y_true = batch.labels  # (B, D, H, W)

        y_pred_full = self(batch)  # (B, D, H, W)
        mask = (y_true > -1.0).detach()
        if use_mito_mask:
            assert (
                batch.aux_data is not None and "labels/mito" in batch.aux_data
            ), "Batch aux_data must contain 'labels/mito' key for mito masking."
            # assumes eval code with batch size of 1
            mito_mask = torch.from_numpy(batch.aux_data["labels/mito"][0]) > 0
            mito_mask = mito_mask.to(dtype=mask.dtype, device=mask.device)
            mask = mask & mito_mask  # Combine masks

        y_pred = torch.masked_select(y_pred_full, mask).view(-1, 1)
        y_true = torch.masked_select(y_true, mask).view(-1, 1)

        return {"preds": y_pred, "labels": y_true, "preds_full": y_pred_full}

    def compute_losses(
        self, y_pred: Tensor, y_true: Tensor
    ) -> dict[str, Tensor]:
        losses = {k: v(y_pred, y_true) for k, v in self.loss_fns.items()}
        losses["total"] = sum(losses.values())
        return losses

    def log_stats(
        self,
        losses: dict[str, Tensor],
        prefix: Literal["train", "val", "test"],
        batch_size: int,
    ) -> None:
        """Logs computed loss and metric statistics for each training or validation step."""
        # Log losses
        loss_log_dict = {f"{prefix}/loss/{k}": v for k, v in losses.items()}
        on_step = prefix == "train"
        self.log_dict(
            loss_log_dict,
            prog_bar=True,
            on_epoch=not on_step,
            on_step=on_step,
            batch_size=batch_size,
        )

        # Log metrics
        metric_log_dict = {}
        for m, m_fn in self.metric_fns[prefix.upper()].items():  # type: ignore
            metric_log_dict[f"{prefix}/metric/{m}"] = m_fn
        self.log_dict(
            metric_log_dict,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )

    def _do_step(self, batch: BatchedTomogramData, batch_idx: int, prefix: Literal["train", "val", "test"]) -> Tensor:  # type: ignore
        """Processes a single batch of data, computes the loss and updates metrics."""
        out = self._masked_predict(batch)
        y_pred, y_true = out["preds"], out["labels"]

        losses = self.compute_losses(y_pred, y_true)

        for _, m_fn in self.metric_fns[prefix.upper()].items():  # type: ignore
            m_fn(y_pred, y_true)

        self.log_stats(losses, prefix, batch.num_tomos)
        return losses["total"]

    def training_step(self, batch: BatchedTomogramData, batch_idx: int) -> Tensor:  # type: ignore
        """Processes one batch during training."""
        return self._do_step(batch, batch_idx, "train")  # type: ignore

    def validation_step(self, batch: BatchedTomogramData, batch_idx: int) -> Tensor:  # type: ignore
        """Processes one batch during validation."""
        return self._do_step(batch, batch_idx, "val")  # type: ignore

    def test_step(self, batch: BatchedTomogramData, batch_idx: int) -> BatchedModelResult:  # type: ignore
        """Processes one batch during testing, captures predictions, and computes losses and metrics.

        Args:
            batch (BatchedTomogramData): The batch of data being processed.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: A dictionary containing test results and metrics for this batch.
        """
        assert (
            batch.aux_data is not None and "data" in batch.aux_data
        ), "Batch aux_data must contain 'data' key for testing."
        input_data = batch.aux_data["data"]

        # whether to use mito labels to mask granule predictions in evaluation
        use_mito_mask = (
            "labels/mito" in batch.aux_data and batch.aux_data["labels/mito"]
            if batch.aux_data is not None
            else False
        )
        out_dict = self._masked_predict(batch, use_mito_mask=use_mito_mask)
        y_pred, y_true, y_pred_full = (
            out_dict["preds"],
            out_dict["labels"],
            out_dict["preds_full"],
        )
        aux_data = {}
        for key in out_dict:
            if key not in ["preds", "labels", "preds_full"] and key.endswith(
                "full"
            ):
                aux_data_full = out_dict[key]
                aux_data[key] = [
                    t_aux_data.cpu().numpy() for t_aux_data in aux_data_full
                ]

        samples, tomo_names = batch.metadata.identifiers
        split_id = batch.metadata.split_id
        labels = [t_labels.cpu().numpy() for t_labels in batch.labels]
        preds = [t_preds.cpu().numpy() for t_preds in y_pred_full]
        if split_id is not None:
            split_id = [s_id.item() for s_id in split_id]

        losses = {
            k: v.item() for k, v in self.compute_losses(y_pred, y_true).items()
        }
        metrics = {}
        for m, m_fn in self.metric_fns["TEST"].items():  # type: ignore
            score = m_fn(y_pred, y_true)
            metrics[m] = score.item()
            m_fn.reset()

        return BatchedModelResult(
            num_tomos=batch.num_tomos,
            samples=samples,
            tomo_names=tomo_names,
            split_id=split_id,
            data=input_data,
            label=labels,
            preds=preds,  # type: ignore
            losses=losses,
            metrics=metrics,
            aux_data=aux_data if aux_data else None,
        )

    def predict_step(self, batch: BatchedTomogramData, batch_idx: int) -> BatchedModelResult:  # type: ignore
        assert (
            batch.aux_data is not None and "data" in batch.aux_data
        ), "Batch aux_data must contain 'data' key for prediction."
        input_data = batch.aux_data["data"]
        preds = self(batch)
        labels = [t_labels.cpu().numpy() for t_labels in batch.labels]
        samples, tomo_names = batch.metadata.identifiers

        return BatchedModelResult(
            num_tomos=batch.num_tomos,
            samples=samples,
            tomo_names=tomo_names,
            split_id=None,
            data=input_data,
            label=labels,
            preds=[t_preds.cpu().numpy() for t_preds in preds],
            losses={},
            metrics={},
            aux_data=None,
        )

    @abstractmethod
    def forward(self):
        """Should be implemented in subclass."""
        raise NotImplementedError(
            "The forward method must be implemented by subclass."
        )
