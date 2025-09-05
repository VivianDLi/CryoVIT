"""Extract visualizations of DINO features with PCA."""

from pathlib import Path

import h5py
import numpy as np


def _process_file(
    file_name: str,
    label_dict: dict[str, Path],
    result_dir: Path,
    threshold: float = 0.5,
) -> None:
    import cv2
    import seaborn as sns

    colors = sns.color_palette("deep")[:4]
    hue_palette = {
        "mito": colors[0],
        "cristae": colors[1],
        "microtubule": colors[2],
        "granule": colors[3],
    }

    label_data = {}
    sample = "unknown"
    for label, f_path in label_dict.items():
        sample = f_path.parent.name
        with h5py.File(f_path, "r") as fh:
            if "data" not in label_data:
                label_data["data"] = fh["data"][()].astype(np.float32)  # type: ignore # [0, 1], [D, H, W]
            label_data[label] = fh["predictions"][label][()].astype(np.float32)  # type: ignore # [0, 1], [D, H, W]
    # Combine all labels into a single segmentation map
    combined_rgb_seg = np.zeros(
        (*label_data["data"].shape, 3), dtype=np.float32
    )  # [D, H, W, 3]
    for label, seg in label_data.items():
        if label in hue_palette:
            color = np.array(hue_palette.get(label, (1.0, 1.0, 1.0))).reshape(
                (1, 1, 1, 3)
            )  # default to white if label not in palette
            rgb_seg = np.stack([seg, seg, seg], axis=-1)  # to RGB
            rgb_seg = rgb_seg * color  # colorize
            combined_rgb_seg += rgb_seg
        else:
            print(f"Couldn't find color for label {label}")
    combined_rgb_seg = np.clip(combined_rgb_seg, 0, 1)
    label_data["data"] = np.clip(label_data["data"], 0, 1)
    final_data = np.stack(
        [label_data["data"], label_data["data"], label_data["data"]], axis=-1
    )  # to RGB
    # Add in data and normalize
    final_rgb = np.where(
        combined_rgb_seg > threshold, combined_rgb_seg, final_data
    )
    # Combine and convert to cv2 format
    final_combined = np.concatenate(
        [final_data, final_rgb], axis=2
    )  # [D, H, 2*W, 3]
    final_combined = (final_combined * 255).astype(np.uint8)
    frames = []
    for frame in final_combined:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # to BGR for cv2
        frames.append(frame)
    # Save as Video
    result_path = result_dir / sample / (file_name + ".mp4")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    video_path = str(result_path.with_suffix(".mp4"))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    fps = 30
    width, height = final_combined.shape[2], final_combined.shape[1]
    video_writer = cv2.VideoWriter(
        str(video_path), fourcc, fps, (width, height)
    )
    for img_frame in frames:
        video_writer.write(img_frame)
    video_writer.release()
    print("Saved video to", video_path)


def _process_files(
    file_dict: dict[str, dict[str, Path]], result_dir: Path
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for f_name, labels in file_dict.items():
        print(f"Processing file {f_name}")
        _process_file(f_name, labels, result_dir)


def process_experiment(
    exp_dir: Path,
    result_dir: Path,
    exp_template: str,
    labels: list[str] | None,
) -> None:
    import seaborn as sns

    colors = sns.color_palette("deep")[:4]
    hue_palette = {
        "mito": colors[0],
        "cristae": colors[1],
        "microtubule": colors[2],
        "granule": colors[3],
    }

    result_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        exp_names = [
            d.name
            for d in exp_dir.iterdir()
            if d.is_dir()
            and d.name.startswith(exp_template)
            and d.name.split("_")[-1] in hue_palette
        ]
        labels = [name.split("_")[-1] for name in exp_names]
    else:
        exp_names = [
            d.name
            for d in exp_dir.iterdir()
            if d.is_dir()
            and d.name.startswith(exp_template)
            and d.name.split("_")[-1] in labels
        ]

    files = {}
    for label, exp_name in zip(labels, exp_names, strict=True):
        file_dir = exp_dir / exp_name
        tomo_files = list(file_dir.glob("**/*.hdf"))
        print(
            f"Found {len(tomo_files)} .hdf files for label {label} in experiment directory {exp_name}"
        )
        files[label] = tomo_files
    all_files = {}
    for label, fs in files.items():
        for f in fs:
            if f.stem not in all_files:
                all_files[f.stem] = {label: f.resolve()}
            else:
                all_files[f.stem][label] = f.resolve()
    # Now, all_files maps each file name to a dictionary of labels and corresponding file paths for that label
    _process_files(
        all_files,
        result_dir / f"{exp_template}_{'_'.join(labels)}_segmentations",
    )
