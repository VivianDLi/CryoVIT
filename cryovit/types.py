"""Defines custom types and dataclasses for model inputs and outputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, NewType, Tuple
import torch
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from tensordict import tensorclass

#### Type Definitions ####

# define dino features data type (C, H, W) float16
DinoFeaturesData = NewType("DinoFeaturesData", NDArray[np.float16])

# define tomogram data type (D, H, W) float32/uint8
FloatTomogramData = NewType("FloatTomogramData", NDArray[np.float32])
IntTomogramData = NewType("IntTomogramData", NDArray[np.uint8])

# define segmentation label data type (D, H, W) uint8
LabelData = NewType("LabelData", NDArray[np.uint8])

#### Class Definitions ####

@tensorclass
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
    label_path: Optional[Path] = None
    labels: Optional[List[str]] = None
    sample: Optional[str] = None

@tensorclass
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
    split_id: Optional[int]
    
    data: torch.FloatTensor
    label: torch.BoolTensor
    aux_data: Optional[Dict[str, Any]] = None

@tensorclass
class BatchedTomogramMetadata:
    """
    This class represents metadata about a batch of tomograms.

    Attributes:
        samples: A list containing all possible samples of tomogram files in the batch.
        tomo_names: A list containing all possible names of tomogram files in the batch.
        unique_id: A [Bx2] tensor containing the corresponding index in samples and tomo_names for each tomogram. Index consists of (sample_id, tomo_name_id).
        split_id: An optional list containing the training/val/test split the tomogram is a part of.
    """
    
    samples: List[str]
    tomo_names: List[str]
    unique_id: torch.LongTensor
    split_id: Optional[torch.IntTensor]
    
    @property
    def identifiers(self) -> Tuple[List[str], List[str]]:
        samples = [self.samples[i[0].item()] for i in self.unique_id]
        names = [self.tomo_names[i[1].item()] for i in self.unique_id]
        return samples, names


@tensorclass
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
    
    tomo_batch: torch.FloatTensor
    tomo_sizes: torch.IntTensor
    labels: torch.BoolTensor
    metadata: BatchedTomogramMetadata
    min_slices: int
    aux_data: Optional[Dict[str, List[Any]]] = None
    
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
    
    def index_to_flat_batch(self, idx: int) -> torch.LongTensor:
        """Returns a [BxD] tensor containing the indices corresponding to a certain slice in a flat batch tensor."""
        if idx >= self.num_slices:
            raise IndexError(f"Slice index {idx} is out of bounds for the maximum number of slices {self.max_slices}.")
        batch_idxs = torch.argwhere(self.tomo_sizes > idx).long()
        batch_sizes = self.tomo_sizes[batch_idxs].flatten()
        batch_ll = torch.cumsum(batch_sizes, dim=0) - batch_sizes
        slice_idxs = batch_ll + idx
        return slice_idxs
    
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
    samples: List[str]
    tomo_names: List[str]
    split_id: Optional[List[int]]
    data: List[FloatTomogramData]
    label: List[LabelData]
    preds: List[FloatTomogramData]
    losses: Dict[str, float]
    metrics: Dict[str, float]
    aux_data: Optional[Dict[str, List[Any]]] = None