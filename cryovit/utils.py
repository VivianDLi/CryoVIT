"""Utility functions to process data and models in a format recognizable by CryoVIT."""

import logging
import string
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pickle

import h5py
import mrcfile
from hydra.utils import instantiate
import numpy as np
from torch import Tensor
import torch

#### General File Utilities ####

def id_generator(size: int = 6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#### Data Loading Utilities ####

@dataclass
class FileMetadata:
    drange: Tuple[float, float]
    dshape: Tuple[int, ...]
    dtype: np.dtype
    nunique: int = 0

def read_hdf_keys(hdf_file: Union[h5py.File, h5py.Group]) -> Tuple[Dict[str, np.ndarray], Dict[str, FileMetadata]]:
    if len(hdf_file.keys()) == 0:
        return {}, {}
    data_results = {}
    metadata_results = {}
    for key in hdf_file.keys():
        if isinstance(hdf_file[key], h5py.Group):
            group_data_results, group_metadata_results = read_hdf_keys(hdf_file[key])
            data_results.update({f"{key}/{k}": v for k, v in group_data_results.items()})
            metadata_results.update({f"{key}/{k}": v for k, v in group_metadata_results.items()})
        elif isinstance(hdf_file[key], h5py.Dataset):
            data = hdf_file[key][()]
            drange = (float(np.min(data)), float(np.max(data)))
            dshape = data.shape
            dtype = data.dtype
            nunique = len(np.unique(data))
            data_results[key] = data
            metadata_results[key] = FileMetadata(drange=drange, dshape=dshape, dtype=dtype, nunique=nunique)
        else:
            raise ValueError(f"Unknown HDF5 object type found for key {key} in file {hdf_file.name}: {type(hdf_file[key])}.")
    return data_results, metadata_results

def read_hdf(hdf_file: Union[str, Path]) -> Dict[str, FileMetadata]:
    with h5py.File(hdf_file, 'r') as f:
        return read_hdf_keys(f)

def read_mrc(mrc_file: Union[str, Path]) -> Tuple[np.array, FileMetadata]:
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        data = mrc.data
        drange = (float(np.min(data)), float(np.max(data)))
        dshape = data.shape
        dtype = data.dtype
        nunique = len(np.unique(data))
        return data, FileMetadata(drange=drange, dshape=dshape, dtype=dtype, nunique=nunique)

def read_tiff(tiff_file: Union[str, Path]) -> Tuple[np.ndarray, FileMetadata]:
    raise NotImplementedError("TIFF reading not yet implemented.")

def read_folder(folder_path: Union[str, Path]) -> Tuple[np.ndarray, FileMetadata]:
    raise NotImplementedError("Folder reading not yet implemented.")

def load_data(file_path: Union[str, Path], key: Optional[str] = None) -> np.ndarray:
    """Load data or labels from a given file path. Supports .h5, .hdf5, .mrc, .mrcs formats."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if file_path.suffix in ['.h5', '.hdf5']:
        data_dict, metadata_dict = read_hdf(file_path)
        if key is None:   
            # Assume the data with the most unique values is the data
            data_key = max(metadata_dict.items(), key=lambda x: x[1].nunique)[0]
            logging.warning(f"No key specified for file {file_path}. Assuming data is the key with the most unique values, and using key '{data_key}' with {metadata_dict[data_key].nunique} unique values. If this is incorrect, please specify the `data_key` manually as a `/`-separated string.")
        else:
            data_key = key
        data = data_dict[data_key]
        metadata = metadata_dict[data_key]
    elif file_path.suffix in ['.mrc', '.mrcs']:
        data, metadata = read_mrc(file_path)
    elif file_path.suffix in ['.tiff', '.tif']:
        data, metadata = read_tiff(file_path)
    elif file_path.is_dir():
        data, metadata = read_folder(file_path)
    else:
        raise ValueError(f"Unsupported file format for file {file_path}. Supported formats are .h5, .hdf5, .mrc, .mrcs, .tiff, .tif, and image folders.")

    if metadata.dtype in [np.float32, np.float16]:
        # Normalize to [0, 1] and return
        data = (data - metadata.drange[0]) / (metadata.drange[1] - metadata.drange[0])
        return data.astype(np.float32)
    elif metadata.dtype in [np.uint8, np.int8, np.uint16, np.int16]:
        # Normalize to [0, 1] and return as float32
        data = data.astype(np.float32)
        data = (data - metadata.drange[0]) / (metadata.drange[1] - metadata.drange[0])
        return data
    else:
        raise ValueError(f"Unsupported data type {metadata.dtype} in file {file_path}. Supported types are float32, float16 for data and uint8, int8, uint16, int16 for labels.")
    
def _match_label_keys_to_data(data: np.ndarray, label_keys: List[str], metadata: FileMetadata) -> Dict[str, np.ndarray]:
    """Match label keys to data based on unique values in the data."""
    labels = {}
    if metadata.nunique == len(label_keys):
        label_values = sorted(np.unique(data).tolist())
        for i, key in zip(label_values, label_keys):
            labels[key] = (data == i).astype(np.bool)
    elif metadata.nunique == len(label_keys) + 1 and 0 in np.unique(data):
        logging.warning(f"Assuming 0 is the background class in label data and hasn't been specified in label_keys.")
        label_values = sorted([x for x in np.unique(data).tolist() if x != 0])
        for i, key in zip(label_values, label_keys):
            labels[key] = (data == i).astype(np.bool)
    else:
        raise ValueError(f"Number of unique values in label data ({metadata.nunique}) does not match number of provided label keys ({len(label_keys)}).")
    return labels
    
def load_labels(file_path: Union[str, Path], label_keys: List[str]) -> Dict[str, np.ndarray]:
    """Load labels from a given file path, given a list of label names in ascending-value order. Supports .h5, .hdf5, .mrc, .mrcs, .tiff, .tif, and image folder formats."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    labels = {}
    if file_path.suffix in ['.h5', '.hdf5']:
        data_dict, metadata_dict = read_hdf(file_path)
        if len(metadata_dict) > 1:
            assert all([key in metadata_dict for key in label_keys]), f"Not all specified label keys {label_keys} found in file {file_path}. Available keys are {list(metadata_dict.keys())}."
            for key in label_keys:
                data = data_dict[key]
                labels[key] = (data > 0).astype(np.bool)
        else:
            data, metadata = list(data_dict.values())[0], list(metadata_dict.values())[0]
            labels = _match_label_keys_to_data(data, label_keys, metadata)
    elif file_path.suffix in ['.mrc', '.mrcs']:
        data, metadata = read_mrc(file_path)
        labels = _match_label_keys_to_data(data, label_keys, metadata)
    elif file_path.suffix in ['.tiff', '.tif']:
        data, metadata = read_tiff(file_path)
        labels = _match_label_keys_to_data(data, label_keys, metadata)
    elif file_path.is_dir():
        data, metadata = read_folder(file_path)
        labels = _match_label_keys_to_data(data, label_keys, metadata)
    else:
        raise ValueError(f"Unsupported file format for file {file_path}. Supported formats are .h5, .hdf5, .mrc, .mrcs, .tiff, .tif, and image folders.")
    
#### DINO Feature Utilities ####

def extract_dino_features(images: List[Tensor]) -> List[Tensor]:
    """Extract DINO features from a list of images."""
    dino_model = None
    features = []
    for img in images:
        feat = dino_model(img)
        features.append(feat)
    return features

#### Creation Utilities ####

@dataclass
class SavedModel:
    name: str
    model_type: str
    label_key: str
    model_cfg: Dict[str, Any]
    weights: Dict[str, Any]

def save_model(model_name: str, label_key: str, model: torch.nn.Module, model_cfg: Dict, save_path: Union[str, Path]) -> None:
    """Save a model to a given path."""
    weights = model.state_dict()
    model_type = model_cfg.name.lower()
    saved_model = SavedModel(name=model_name, model_type=model_type, label_key=label_key, model_cfg=model_cfg, weights=weights)
    with open(save_path, "wb") as f:
        pickle.dump(saved_model, f)

def load_model(model_path: Union[str, Path]) -> Tuple[torch.nn.Module, str, str, str]:
    """Load a model from a given path. Returns the model, model type, model name, and label key."""
    with open(model_path, "rb") as f:
        saved_model = pickle.load(f)
    model = instantiate(saved_model.model_cfg)
    model.load_state_dict(saved_model.weights)
    return model, saved_model.model_type, saved_model.name, saved_model.label_key