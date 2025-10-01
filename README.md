# CryoVIT: Efficient Segmentation of Cryo-electron Tomograms with Vision Foundation Models

## Installation Instructions

CryoVIT uses Mamba to manage python packages and dependencies and can be downloaded [here](https://github.com/conda-forge/miniforge). You should also be able to use Conda instead of Mamba but setting up the environment may take an unreasonably long time.

1. Clone the CryoVIT github repository: `git clone https://github.com/sanketx/CryoVIT.git`
2. Setup the mamba environment: `mamba env create -f CryoVIT/env.yml`
3. Activate the mamba environment: `mamba activate cryovit_env`

## Usage Instructions

For more details, check out the [documentation](https://cryovit.readthedocs.io/en/latest/).

## Replicating Experiments

All experiments are managed using Hydra, with their configurations stored in `src/cryovit/configs`.

Before running experiments, make sure to change the relevant entries in `src/cryovit/configs/paths/default.yaml`.

You can run a specific experiment using the scripts in the `cryovit.training` module.

For example:

```console
$ mamba activate cryovit_env
$ python -m cryovit.training.dino_features
$ python -m cryovit.training.train_model +experiment=<experiment_config_name>
$ python -m cryovit.training.eval_model +experiment=<experiment_config_name>
```

These commands assume your data is setup in the following format:

```
dataset/
├── sample_1/
│   ├── tomogram1.hdf
│   ├── tomogram2.hdf
│   └── ...
├── sample_2/
│   ├── tomogram1.hdf
│   ├── tomogram2.hdf
│   └── ...
├── sample_3/
│   └── ...
├── sample_4/
│   └── ...
│   ...
```

where each tomogram has a `data` key, with labels saved in `labels/<label_name>` keys.

All figures in the paper were produced using the `cryovit.training.visualize_results` command.