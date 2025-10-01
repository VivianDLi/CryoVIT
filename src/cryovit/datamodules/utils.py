"""Utility functions for datamodules."""

import torch
from torch.nn.functional import pad

from cryovit.types import (
    BatchedTomogramData,
    BatchedTomogramMetadata,
    TomogramData,
)


def collate_fn(batch: list[TomogramData]) -> BatchedTomogramData:  # type: ignore
    """Combine multiple tomograms into a single batch with metadata."""

    # Initialize metadata
    unique_samples = {}  # use dictionaries as ordered sets
    unique_names = {}
    tomo_identifiers = torch.empty(len(batch), 2, dtype=torch.long)
    split_id = torch.empty(len(batch), dtype=torch.int)
    use_splits = True

    tomo_sizes = torch.empty(len(batch), dtype=torch.int)
    min_slices = float("inf")
    # Get tomogram sizes
    for tomo_idx, tomo_data in enumerate(batch):
        D = tomo_data.data.shape[-3]
        tomo_sizes[tomo_idx] = D
        min_slices = min(min_slices, D)

    # Initialize data arrays
    max_size = int(tomo_sizes.max().item())
    C, _, Hp, Wp = batch[0].data.size()
    H, W = batch[0].label.size()[-2:]
    tomo_batch = torch.empty(
        len(batch), C, max_size, Hp, Wp, dtype=torch.float
    )
    labels = torch.empty(len(batch), max_size, H, W, dtype=torch.float)
    aux_data = {key: [] for key in batch[0].aux_data}
    # Loop through batch tomograms
    for tomo_idx, tomo_data in enumerate(batch):
        # Get data
        data = tomo_data.data
        label = tomo_data.label
        for key, value in tomo_data.aux_data.items():
            aux_data[key].append(value)

        # Pad data
        D = data.size()[-3]
        if max_size != D:
            padding = max_size - D
            data = pad(
                data, (0, 0, 0, 0, 0, padding), mode="constant", value=0
            )
            label = pad(
                data, (0, 0, 0, 0, 0, padding), mode="constant", value=-1
            )  # ignore padding
        tomo_batch[tomo_idx] = data
        labels[tomo_idx] = label

        # Get metadata
        sample = tomo_data.sample
        name = tomo_data.tomo_name
        split = tomo_data.split_id

        unique_samples[sample] = None
        s_id = list(unique_samples).index(sample)
        unique_names[name] = None
        n_id = list(unique_names).index(name)
        tomo_identifiers[tomo_idx][0] = s_id
        tomo_identifiers[tomo_idx][1] = n_id
        if split is not None and use_splits:
            split_id[tomo_idx] = split
        else:
            use_splits = False

    tomo_batch = tomo_batch.permute(
        (0, 2, 1, 3, 4)
    )  # (B, C, D, H, W) -> (B, D, C, H, W)
    metadata = BatchedTomogramMetadata(
        samples=list(unique_samples),
        tomo_names=list(unique_names),
        unique_id=tomo_identifiers,
        split_id=split_id if use_splits else None,
    )  # type: ignore
    return BatchedTomogramData(
        tomo_batch=tomo_batch,
        tomo_sizes=tomo_sizes,
        labels=labels,
        aux_data=aux_data,
        metadata=metadata,
        min_slices=min_slices,
    )  # type: ignore
