"""Utility functions to process data and models in a format recognizable by CryoVIT."""

import logging
import pickle
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import mrcfile
import numpy as np
import tifffile as tf
import torch
from hydra.utils import instantiate

from cryovit.config import BaseModel, tomogram_exts
from cryovit.models.sam2 import create_sam_model_from_weights
from cryovit.types import ModelType

#### General File Utilities ####

# Recognized file extensions for data files
RECOGNIZED_FILE_EXTS = [
    ".h5",
    ".hdf",
    ".hdf5",
    ".mrc",
    ".mrcs",
    ".tiff",
    ".tif",
]


def id_generator(size: int = 6, chars=string.ascii_lowercase + string.digits):
    """Generates a random string of fixed size."""

    return "".join(random.choice(chars) for _ in range(size))


#### Data Loading Utilities ####


@dataclass
class FileMetadata:
    """Metadata information for a file.

    Attributes:
        drange: The dynamic range of the data.
        dshape: The shape of the data.
        dtype: The data type of the data.
        nunique: The number of unique values in the data.
    """

    drange: tuple[float, float]
    dshape: tuple[int, ...]
    dtype: np.dtype
    nunique: int = 0


def _read_hdf_keys(
    hdf_file: h5py.File | h5py.Group, data_key: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, FileMetadata]]:
    """Recursively read all keys in an HDF5 file or group, returning as a tuple of dictionaries of key: <data>/<metadata>. If a data_key is specified, only reads that key."""

    if len(hdf_file.keys()) == 0:
        return {}, {}
    data_results = {}
    metadata_results = {}
    if data_key is not None:
        try:
            data: np.ndarray = hdf_file[data_key][()]  # type: ignore
            drange = (float(np.min(data)), float(np.max(data)))
            dshape = data.shape
            dtype = data.dtype
            nunique = len(np.unique(data))
            data_results[data_key] = data
            metadata_results[data_key] = FileMetadata(
                drange=drange, dshape=dshape, dtype=dtype, nunique=nunique
            )
            return data_results, metadata_results
        except KeyError:
            logging.warning(
                "Key %s not found in file %s. Attempting to read all keys instead.",
                data_key,
                hdf_file,
            )
    for key in hdf_file:
        if isinstance(hdf_file[key], h5py.Group):
            group_data_results, group_metadata_results = _read_hdf_keys(hdf_file[key])  # type: ignore
            data_results.update(
                {f"{key}/{k}": v for k, v in group_data_results.items()}
            )
            metadata_results.update(
                {f"{key}/{k}": v for k, v in group_metadata_results.items()}
            )
        elif isinstance(hdf_file[key], h5py.Dataset):
            data: np.ndarray = hdf_file[key][()]  # type: ignore
            drange = (float(np.min(data)), float(np.max(data)))
            dshape = data.shape
            dtype = data.dtype
            nunique = len(np.unique(data))
            data_results[key] = data
            metadata_results[key] = FileMetadata(
                drange=drange, dshape=dshape, dtype=dtype, nunique=nunique
            )
        else:
            raise ValueError(
                f"Unknown HDF5 object type found for key {key} in file {hdf_file.name}: {type(hdf_file[key])}."
            )
    return data_results, metadata_results


def read_hdf(
    hdf_file: str | Path, key: str | None = None
) -> tuple[str, np.ndarray, FileMetadata]:
    """Read data from an HDF5 file. If a key is not specified, assumes the data with the most unique values is the data.

    Args:
        hdf_file: The path to the HDF5 file.
        key: The key to read from the HDF5 file. If None, assumes the data with the most unique values is the data. If not a valid key, will attempt to read all keys and use the one with the most unique values.

    Returns:
        A tuple of the key used, the data, and the metadata.
    """

    with h5py.File(hdf_file, "r") as f:
        data_dict, metadata_dict = _read_hdf_keys(f, data_key=key)
    if key is None:
        # Assume the data with the most unique values is the data
        data_key = max(metadata_dict.items(), key=lambda x: x[1].nunique)[0]
        logging.info(
            "No key specified for file %s. Assuming data is the key with the most unique values, and using key '%s' with %d unique values. If this is incorrect, please specify the `data_key` manually as a `/`-separated string.",
            hdf_file,
            data_key,
            metadata_dict[data_key].nunique,
        )
    else:
        data_key = key
    data = data_dict[data_key]
    metadata = metadata_dict[data_key]
    return data_key, data, metadata


def read_mrc(mrc_file: str | Path) -> tuple[np.ndarray, FileMetadata]:
    """Read data from an MRC file.

    Args:
        mrc_file: The path to the MRC file.

    Returns:
        A tuple of the data and the metadata.
    """

    data: np.ndarray = mrcfile.read(mrc_file)
    drange = (float(np.min(data)), float(np.max(data)))
    dshape = data.shape
    dtype = data.dtype
    nunique = len(np.unique(data))
    return data, FileMetadata(
        drange=drange, dshape=dshape, dtype=dtype, nunique=nunique
    )


def read_tiff(tiff_file: str | Path) -> tuple[np.ndarray, FileMetadata]:
    """Read data from a TIFF file.

    Args:
        tiff_file: The path to the TIFF file.

    Returns:
        A tuple of the data and the metadata.
    """

    data: np.ndarray = tf.imread(tiff_file)
    drange = (float(np.min(data)), float(np.max(data)))
    dshape = data.shape
    dtype = data.dtype
    nunique = len(np.unique(data))
    return data, FileMetadata(
        drange=drange, dshape=dshape, dtype=dtype, nunique=nunique
    )


def load_data(
    file_path: str | Path, key: str | None = None
) -> tuple[np.ndarray, str]:
    """Load data or labels from a given file path. Supports .h5, .hdf5, .mrc, .mrcs formats.

    Args:
        file_path: The path to the data file.
        key: An optional key to specify which dataset to load from an HDF5 file.

    Raises:
        ValueError: If the file format is unsupported.
        FileNotFoundError: If the specified file does not exist.

    Returns:
        A tuple of the data and the key used (empty string if not applicable).
    """

    file_path = Path(file_path)
    found_key = ""
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if file_path.suffix in [".h5", ".hdf", ".hdf5"]:
        found_key, data, metadata = read_hdf(file_path, key=key)
    elif file_path.suffix in [".mrc", ".mrcs"]:
        data, metadata = read_mrc(file_path)
    elif file_path.suffix in [".tiff", ".tif"]:
        data, metadata = read_tiff(file_path)
    else:
        raise ValueError(
            f"Unsupported file format for file {file_path}. Supported formats are .h5, .hdf, .hdf5, .mrc, .mrcs, .tiff, .tif, and image folders."
        )

    # Assumed float data already normalized or are DINO features
    if metadata.dtype in [np.uint8, np.int8, np.uint16, np.int16]:
        # Normalize to [0, 1] and return as float32
        data = data.astype(np.float32) / 255.0

    if len(data.shape) == 3:
        data = data[np.newaxis, ...]  # add channel dimension
    return data, found_key


def _match_label_keys_to_data(
    data: np.ndarray, label_keys: list[str], metadata: FileMetadata
) -> dict[str, np.ndarray]:
    """Match label keys to data based on the number of unique values in the data and the provided label keys. Assumes label_keys are in ascending-value order."""

    labels = {}
    nunique = (
        metadata.nunique if metadata.drange[0] >= 0 else metadata.nunique - 1
    )  # ignore negative values for nunique count
    if nunique == len(label_keys):
        label_values = sorted(np.unique(data).tolist())
        for i, key in zip(label_values, label_keys, strict=True):
            label = np.where((data != i) & (data != -1), 0, data)
            labels[key] = np.where(label == i, 1, label).astype(np.int8)
    elif nunique == len(label_keys) + 1 and 0 in np.unique(data):
        logging.debug(
            "Assuming 0 is the background class in label data and hasn't been specified in label_keys."
        )
        label_values = sorted([x for x in np.unique(data).tolist() if x > 0])
        for i, key in zip(label_values, label_keys, strict=True):
            label = np.where((data != i) & (data != -1), 0, data)
            labels[key] = np.where(label == i, 1, label).astype(np.int8)
    else:
        raise ValueError(
            f"Number of unique values in label data ({metadata.nunique}) does not match number of provided label keys ({len(label_keys)})."
        )
    return labels


def load_labels(
    file_path: str | Path, label_keys: list[str], key: str | None
) -> dict[str, np.ndarray]:
    """Load labels from a given file path, given a list of label names in ascending-value order. Supports .h5, .hdf5, .mrc, .mrcs, .tiff, and .tif formats.

    Args:
        file_path: The path to the label file.
        label_keys: A list of label names in ascending-value order (e.g., ['mito', 'cristae'] for 0=background, 1=mito, 2=cristae).
        key: An optional key to specify which dataset to load from an HDF5 file.

    Raises:
        ValueError: If the number of unique values in the label data does not match the number of provided label keys.
        ValueError: If the file format is unsupported.
        ValueError: If the specified key is not found in the label data.
        FileNotFoundError: If the specified file does not exist.

    Returns:
        A dictionary of label name to int8 label array.
    """

    assert (
        key is None or key in label_keys
    ), f"Label key {key} must be one of the specified label keys {label_keys} or None."
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    labels = {}
    if file_path.suffix in [".h5", ".hdf", ".hdf5"]:
        _, data, metadata = read_hdf(file_path, key=key)
        if len(label_keys) > 1:
            labels_dict = _match_label_keys_to_data(data, label_keys, metadata)
            labels.update(labels_dict)
        else:
            labels[key] = data.astype(np.int8)
    elif file_path.suffix in [".mrc", ".mrcs"]:
        data, metadata = read_mrc(file_path)
        labels.update(_match_label_keys_to_data(data, label_keys, metadata))
    elif file_path.suffix in [".tiff", ".tif"]:
        data, metadata = read_tiff(file_path)
        labels.update(_match_label_keys_to_data(data, label_keys, metadata))
    else:
        raise ValueError(
            f"Unsupported file format for file {file_path}. Supported formats are .h5, .hdf, .hdf5, .mrc, .mrcs, .tiff, .tif, and image folders."
        )
    return labels


def load_files_from_path(path: Path) -> list[Path]:
    """Load files from a given directory or a .txt file listing file paths.

    Args:
        path (Path): The path to the directory or .txt file.

    Raises:
        ValueError: If the path is not a directory or a .txt file.

    Returns:
        list[Path]: A list of file paths.
    """

    if path.is_dir():
        file_paths = sorted(
            [f for f in path.rglob("*") if f.suffix in tomogram_exts]
        )
    elif path.is_file() and path.suffix == ".txt":
        with open(path) as f:
            file_paths = [Path(line.strip()) for line in f if line.strip()]
    else:
        raise ValueError(
            "Data path must be a directory or a .txt file listing data files."
        )
    assert len(file_paths) > 0, f"No valid tomogram files found in {path}."
    return file_paths


#### Creation Utilities ####


@dataclass
class SavedModel:
    """A class to represent a pre-trained model.

    Attributes:
        name: The name of the model.
        model_type: The type of the model, e.g., 'CryoVIT', '3D U-Net'.
        label_key: The label key used for training the model.
        model_cfg: The config dictionary to instantiate the model.
        weights: The saved weights of the model.
    """

    name: str
    model_type: ModelType
    label_key: str
    model_cfg: BaseModel
    weights: dict[str, Any]


def save_model(
    model_name: str,
    label_key: str,
    model: torch.nn.Module,
    model_cfg: BaseModel,
    save_path: str | Path,
) -> None:
    """Save a model to a given path.

    Args:
        model_name: The name of the model.
        label_key: The label key used for training the model.
        model: The model to save.
        model_cfg: The config dictionary to instantiate the model.
        save_path: The path to save the model to.
    """

    weights = model.state_dict()
    model_type = model_cfg.name.lower()
    saved_model = SavedModel(
        name=model_name,
        model_type=ModelType(model_type),
        label_key=label_key,
        model_cfg=model_cfg,
        weights=weights,
    )
    with open(save_path, "wb") as f:
        pickle.dump(saved_model, f)


def save_model_from_weights(
    model_name: str,
    label_key: str,
    model_type: ModelType,
    weights_path: str | Path,
    save_path: str | Path,
    **kwargs,
) -> None:
    """Save a model to a given path from a weights file.

    Args:
        model_name: The name of the model.
        label_key: The label key used for training the model.
        model_type: The type of the model.
        weights_path: The path to the weights file.
        save_path: The path to save the model to.
        **kwargs: Additional keyword arguments to pass to the model config. To access nested config parameters, use double underscores (e.g., `a.b -> a__b`).

    Raises:
        FileNotFoundError: If the weights file does not exist.
    """
    from hydra import compose, initialize

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights file {weights_path} does not exist.")
    weights = torch.load(weights_path)
    # load model from Hydra config
    with initialize(
        version_base="1.2",
        config_path="configs",
    ):
        cfg = compose(
            config_name="infer_model",
            overrides=[f"model={model_type.value}"]
            + [f"model.{k}={v}" for k, v in kwargs.items()],
        )
    model_dir = Path(__file__).parent / "foundation_models"
    if cfg.model._target_ == "cryovit.models.sam2.SAM2":
        # Load SAM2 pre-trained models
        model = create_sam_model_from_weights(cfg.model, model_dir / "SAM2")
    else:
        model = instantiate(cfg.model)
    model.load_state_dict(weights)

    save_model(model_name, label_key, model, cfg.model, save_path)


def load_model(
    model_path: str | Path, load_model: bool = True
) -> tuple[torch.nn.Module | None, ModelType, str, str]:
    """Load a model from a given path.

    Args:
        model_path: The path to the model file.
        load_model: Whether to load the model weights. If False, only returns the model type, name, and label key.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    Returns:
        A tuple of the model (or None if load_model is False), the model type, the model name, and the label key.
    """

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    with open(model_path, "rb") as f:
        saved_model = pickle.load(f)
    if load_model:
        model_dir = Path(__file__).parent / "foundation_models"
        if saved_model.model_cfg._target_ == "cryovit.models.sam2.SAM2":
            # Load SAM2 pre-trained models
            model = create_sam_model_from_weights(
                saved_model.model_cfg, model_dir / "SAM2"
            )
        else:
            model = instantiate(saved_model.model_cfg)
        model.load_state_dict(saved_model.weights)
    else:
        model = None
    return (
        model,
        saved_model.model_type,
        saved_model.name,
        saved_model.label_key,
    )
