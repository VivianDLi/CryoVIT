import argparse
from pathlib import Path

from cryovit._logging_config import setup_logging
from cryovit.visualization import (
    process_experiment,
    process_fractional_experiment,
    process_multi_experiment,
    process_multi_label_experiment,
    process_samples,
    process_single_experiment,
    process_sparse_experiment,
)

setup_logging("INFO")
import logging  # noqa: E402

model_names = {
    "cryovit": "CryoViT",
    "unet3d": "3D U-Net",
    "sam2": "SAM2",
}

experiment_names = {
    "dino_pca": {},
    "segmentations": {
        s_group: {
            m_key: f"single_{s_group.lower()}_{m_key}" for m_key in model_names
        }
        for s_group in ["HD"]
    },
    "single": {
        s_group: {
            f"single_{s_group.lower()}_{m_key}_mito": [m_value, s_group]
            for m_key, m_value in model_names.items()
        }
        for s_group in ["AD", "HD", "RGC", "Algae"]
    },
    "multi": {
        s_group: {
            m_value: {
                f"{s_group[0]}_to_{s_group[1]}_{m_key}_mito": [
                    m_value,
                    "forward",
                ],
                f"{s_group[1]}_to_{s_group[0]}_{m_key}_mito": [
                    m_value,
                    "backward",
                ],
            }
            for m_key, m_value in model_names.items()
        }
        for s_group in [
            ("hd", "healthy"),
            ("old", "young"),
            ("neuron", "fibro_cancer"),
        ]
    },
    "multi_label": {
        f"fractional_{m_key}_{s_group}": [m_value, s_group]
        for m_key, m_value in model_names.items()
        for s_group in [
            "mito",
            "cristae",
            "microtubule",
            "granule",
            "bacteria",
            "mito_membrane",
        ]
    },
    "fractional": {
        s_group: {
            f"fractional_{m_key}_{s_group}": [m_value]
            for m_key, m_value in model_names.items()
        }
        for s_group in [
            "mito",
            "cristae",
            "microtubule",
            "granule",
            "bacteria",
        ]
    },
    "sparse": {
        s_group: {
            f"fractional_cryovit_mito_{s_label.lower()}": [
                f"CryoViT with {s_label} Labels"
            ]
            for s_label in ["Sparse", "Dense"]
        }
        for s_group in ["single", "fractional"]
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main function to visualize the results of certain CryoViT experiments."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Directory of experiment results",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--exp_type",
        type=str,
        required=True,
        choices=experiment_names.keys(),
        help="Type of experiment to visualize",
    )
    parser.add_argument(
        "--exp_group",
        type=str,
        default=None,
        required=False,
        help="Experiment group to visualize (e.g., 'hd', 'ad', 'rgc'). All options if not specified.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="Specific labels to visualize for `segmentation` visualization (e.g., `mito`, `cristae`). All options if not specified.",
    )

    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    result_dir = Path(args.result_dir)

    # Sanity checking
    assert (
        exp_dir.exists() and exp_dir.is_dir()
    ), "Experiment directory does not exist or is not a directory."
    if args.exp_group is not None and args.exp_type != "dino_pca":
        assert (
            args.exp_group in experiment_names[args.exp_type]
        ), f"Experiment group {args.exp_group} not found in experiment type {args.exp_type}. Available groups: {list(experiment_names[args.exp_type].keys())}"

    exp_group = (
        [args.exp_group]
        if args.exp_group
        else list(experiment_names[args.exp_type].keys())
    )

    exp_names = {}
    for group in exp_group:
        if group in experiment_names[args.exp_type]:
            exp_names[group] = experiment_names[args.exp_type][group]

    match args.exp_type:
        case "dino_pca":
            process_samples(exp_dir, result_dir, sample=args.exp_group)
        case "segmentations":
            for group in exp_group:
                for model in exp_names[group]:
                    process_experiment(
                        exp_dir,
                        result_dir,
                        exp_template=exp_names[group][model],
                        labels=args.labels,
                    )
        case "single":
            process_single_experiment(
                args.exp_type, "all", exp_names, exp_dir, result_dir
            )
        case "multi":
            for group, model_and_names in exp_names.items():
                combined_names = {}
                for names in model_and_names.values():
                    combined_names.update(names)
                process_multi_experiment(
                    args.exp_type, group, combined_names, exp_dir, result_dir
                )
        case "multi_label":
            process_multi_label_experiment(
                args.exp_type, exp_names, exp_dir, result_dir
            )
        case "fractional":
            for label, names in exp_names.items():
                process_fractional_experiment(
                    args.exp_type,
                    label,
                    names,
                    exp_dir,
                    result_dir,
                    plot_points=True,
                )
                process_fractional_experiment(
                    args.exp_type,
                    label,
                    names,
                    exp_dir,
                    result_dir,
                    plot_points=False,
                )
        case "sparse":
            for group, names in exp_names.items():
                process_sparse_experiment(
                    args.exp_type, group, names, exp_dir, result_dir
                )
        case _:
            logging.error("Unknown experiment type: %s", args.exp_type)
