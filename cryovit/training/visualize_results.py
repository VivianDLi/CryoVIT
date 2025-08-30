import argparse
from pathlib import Path

from cryovit.visualization import process_single_experiment, process_multi_experiment, process_fractional_experiment, process_samples, process_experiment

model_names = {
    "cryovit": "CryoViT",
    "unet3d": "3D U-Net",
    # "sam2": "SAM2",
    # "medsam": "MedSAM"
}

experiment_names = {
    "dino_pca": {},
    "segmentations": {s_group: {m_key: f"single_{s_group}_{m_key}" for m_key in model_names} for s_group in ["ad", "hd"]},
    "mitochondria": {s_group: {f"single_{s_group}_{m_key}_mito": [m_value] for m_key, m_value in model_names.items()} for s_group in ["ad", "hd", "rgc", "algae"]},
    "cristae": {s_group: {f"single_{s_group}_{m_key}_cristae": [m_value] for m_key, m_value in model_names.items()} for s_group in ["ad", "hd"]},
    "microtubules": {s_group: {f"single_{s_group}_{m_key}_microtubule": [m_value] for m_key, m_value in model_names.items()} for s_group in ["ad", "hd"]},
    "granules": {s_group: {f"single_{s_group}_{m_key}_granule": [m_value] for m_key, m_value in model_names.items()} for s_group in ["hd"]},
    "bacteria": {s_group: {f"single_{s_group}_{m_key}_bacteria": [m_value] for m_key, m_value in model_names.items()} for s_group in ["campy"]},
    "multi": {s_group: {m_value: {f"{s_group[0]}_to_{s_group[1]}_{m_key}_mito": [m_value, "forward"], f"{s_group[1]}_to_{s_group[0]}_{m_key}_mito": [m_value, "backward"]} for m_key, m_value in model_names.items()} for s_group in [("hd", "healthy"), ("old", "young"), ("neuron", "fibro_cancer")]},
    "fractional": {"hd": {f"fractional_{m_key}_mito": [m_value] for m_key, m_value in model_names.items()}},
    "sparse": {
        "single": {f"single_sample_cryovit_mito": ["CryoViT"], f"single_sample_cryovit_mito_ai": ["CryoViT"]},
        "fractional": {f"fractional_sample_cryovit_mito": ["Sparse"], f"fractional_sample_cryovit_mito_ai": ["Dense"]}
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the results of certain CryoViT experiments.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Directory of experiment results")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--exp_type", type=str, required=True, choices=experiment_names.keys(), help="Type of experiment to visualize")
    parser.add_argument("--exp_group", type=str, default=None, required=False, help="Experiment group to visualize (e.g., 'hd', 'ad', 'rgc'). All options if not specified.")
    parser.add_argument("--labels", type=str, nargs='+', default=None, required=False, help="Specific labels to visualize for `segmentation` visualization (e.g., `mito`, `cristae`). All options if not specified.")

    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    result_dir = Path(args.result_dir)

    # Sanity checking
    assert exp_dir.exists() and exp_dir.is_dir(), "Experiment directory does not exist or is not a directory."
    if args.exp_group is not None:
        assert args.exp_group in experiment_names[args.exp_type]
    exp_group = [args.exp_group] if args.exp_group else list(experiment_names[args.exp_type].keys())
    
    exp_names = {}
    for group in exp_group:
        exp_names[group] = experiment_names[args.exp_type][group]

    if args.exp_type == "dino_pca":
        process_samples(exp_dir, result_dir)
    elif args.exp_type == "segmentations":
        for group in exp_group:
            for model in exp_names[group]:
                process_experiment(exp_dir, result_dir, exp_template=exp_names[group][model], labels=args.labels)
    elif args.exp_type == "multi":
        for group, model_and_names in exp_names.items():
            combined_names = {}
            for model, names in model_and_names.items():
                combined_names.update(names)
            process_multi_experiment(args.exp_type, group, combined_names, exp_dir, result_dir)
    elif args.exp_type == "fractional":
        for group, names in exp_names.items():
            process_fractional_experiment(args.exp_type, group, names, exp_dir, result_dir)
    elif args.exp_type == "sparse":
        for sample_type, names in exp_names.items():
            if sample_type == "single":
                process_single_experiment(args.exp_type, "hd", names, exp_dir, result_dir)
            elif sample_type == "fractional":
                process_fractional_experiment(args.exp_type, "hd", names, exp_dir, result_dir)
            else:
                print(f"Unknown sample type: {sample_type}")
                continue
    else:
        for group, names in exp_names.items():
            process_single_experiment(args.exp_type, group, names, exp_dir, result_dir)
