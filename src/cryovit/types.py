"""Defines custom types and dataclasses for model inputs and outputs."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import tensorclass


#### Enum Definitions ####
class Sample(Enum):
    """Enum of all valid CryoET Samples."""

    BACHD = "BACHD"
    BACHD_Microtubules = "BACHD Microtubules"
    dN17_BACHD = "dN17 BACHD"
    Q109 = "Q109"
    Q109_Microtubules = "Q109 Microtubules"
    Q18 = "Q18"
    Q18_Microtubules = "Q18 Microtubules"
    Q20 = "Q20"
    Q53 = "Q53"
    Q53_KD = "Q53 PIAS1"
    Q66 = "Q66"
    Q66_GRFS1 = "Q66 GRFS1"
    Q66_KD = "Q66 PIAS1"
    WT = "Wild Type"
    WT_Microtubules = "Wild Type Microtubules"
    cancer = "Cancer"
    AD = "AD"
    AD_Abeta = "AD Abeta"
    Aged = "Aged"
    Young = "Young"
    RGC_CM = "RGC CM"
    RGC_control = "RGC Control"
    RGC_naPP = "RGC naPP"
    RGC_PP = "RGC PP"
    CZI_Algae = "Algae"
    CZI_Campy_C = "Campy C"
    CZI_Campy_CDel = "Campy C-Deletion"
    CZI_Campy_F = "Campy F"


class ModelType(Enum):
    """Enum of all valid model types."""

    CRYOVIT = "cryovit"
    UNET3D = "unet3d"
    SAM2 = "sam2"
    MEDSAM = "medsam"


#### Class Definitions ####


@dataclass
class FileData:
    """
    This class represents the file data for a single tomogram.

    Attributes:
        sample: A string representing the sample. None if not available.
        tomo_path: A path to the raw tomogram data.
        label_path: A path to the segmentation labels. None if not available.
        labels: A list of strings representing the label names. None if not available.
    """

    tomo_path: Path
    label_path: Path | None = None
    labels: list[str] | None = None
    sample: str | None = None


@tensorclass  # type: ignore
class TomogramData:
    """
    This class represents the data in single tomogram.

    Attributes:
        sample: A string representing the experiment sample.
        tomo_name: A string for the file_path of the raw tomogram data.
        split_id: An optional identifier for training/val/test splits.
        data: A [DxCxHxW] tensor containing the tomogram data.
        label: A [DxHxW] tensor containing the segmentation labels.
        aux_data: An optional dictionary containing additional data, such as raw data input for dino_features.
    """

    sample: str
    tomo_name: str
    split_id: int | None

    data: torch.FloatTensor
    label: torch.BoolTensor
    aux_data: dict[str, Any] | None = None


@tensorclass  # type: ignore
class BatchedTomogramMetadata:
    """
    This class represents metadata about a batch of tomograms.

    Attributes:
        samples: A list containing all possible samples of tomogram files in the batch.
        tomo_names: A list containing all possible names of tomogram files in the batch.
        unique_id: A [Bx2] tensor containing the corresponding index in samples and tomo_names for each tomogram. Index consists of (sample_id, tomo_name_id).
        split_id: An optional list containing the training/val/test split the tomogram is a part of.
    """

    samples: list[str]
    tomo_names: list[str]
    unique_id: torch.LongTensor
    split_id: list[torch.IntTensor] | None

    @property
    def identifiers(self) -> tuple[list[str], list[str]]:
        samples = [self.samples[i[0].item()] for i in self.unique_id]  # type: ignore
        names = [self.tomo_names[i[1].item()] for i in self.unique_id]  # type: ignore
        return samples, names


@tensorclass  # type: ignore
class BatchedTomogramData:
    """
    This class represents a batch of tomograms with associated annotations.

    Attributes:
        tomo_batch: A [[BxDxCxHxW] tensor containing the tomogram data for each slice in the batch, where D is a tomogram's depth, and B is the number of tomograms in the batch. The D dimension is padded to the max in the batch.
        tomo_sizes: A [B] tensor containing the size (D) of each tomogram in the batch.
        num_total_slices: An integer containing the total number of slices in the batch.
        labels: A [[BxDxHxW] tensor containing the binary labels for segmentation objects in the batch.
        aux_data: A dictionary containing additional data as a list of values, such as raw data input for dino_features.
        metadata: An instance of BatchedTomogramMetadata containing metadata about the batch and the tomograms inside.
    """

    tomo_batch: torch.Tensor
    tomo_sizes: torch.Tensor
    labels: torch.Tensor
    metadata: BatchedTomogramMetadata  # type: ignore
    min_slices: int
    aux_data: dict[str, list[Any]] | None = None

    def pin_memory(self, device=None):
        self.tomo_batch = self.tomo_batch.pin_memory(device=device)
        self.tomo_sizes = self.tomo_sizes.pin_memory(device=device)
        self.labels = self.labels.pin_memory(device=device)
        return self

    @property
    def num_tomos(self) -> int:
        """Returns the number of tomograms in the batch."""
        return self.tomo_batch.shape[0]

    @property
    def num_slices(self) -> int:
        """Returns the maximum number of slices in the batch."""
        return self.tomo_batch.shape[1]

    def index_to_flat_batch(self, idx: int) -> torch.Tensor:
        """Returns a [BxD] tensor containing the indices corresponding to a certain slice in a flat batch tensor."""
        if idx >= self.num_slices:
            raise IndexError(
                f"Slice index {idx} is out of bounds for the maximum number of slices {self.num_slices}."
            )
        batch_idxs = torch.argwhere(self.tomo_sizes > idx).long()
        batch_sizes = self.tomo_sizes[batch_idxs].flatten()
        batch_ll = torch.cumsum(batch_sizes, dim=0) - batch_sizes
        slice_idxs = batch_ll + idx
        return slice_idxs.long()

    @property
    def flat_tomo_batch(self) -> torch.Tensor:
        """Returns a [[BxD]xCxHxW] tensor from a [BxDxCxHxW] tensor (C is optional)."""
        return self.tomo_batch.reshape(-1, *self.tomo_batch.shape[2:])


@dataclass
class BatchedModelResult:
    """
    This class represents the model result from a batch of tomograms, organized per tomogram.

    Attributes:
        num_tomos: The number of tomograms in the batch.
        samples: The sample for each tomogram in the batch.
        tomo_names: The file name for each tomogram in the batch.
        split_id: The optional split id for each tomogram in the batch.
        datas: The raw tomogram data for each tomogram in the batch.
        labels: The true segmentation labels for each tomogram in the batch.
        preds: The model predictions for each tomogram in the batch.
        losses: A dictionary of losses for each tomogram in the batch.
        metrics: A dictionary of metrics for each tomogram in the batch.
        aux_data: An optional dictionary containing auxiliary data for each tomogram in the batch.
    """

    num_tomos: int
    samples: list[str]
    tomo_names: list[str]
    split_id: list[int] | None
    data: list[NDArray[np.float32]]
    label: list[NDArray[np.uint8]]
    preds: list[NDArray[np.float32]]
    losses: dict[str, float]
    metrics: dict[str, float]
    aux_data: dict[str, list[Any]] | None = None
