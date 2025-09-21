#!/bin/bash

exp_name=$1
model=$2

# Setup environment
env_dir=/sdf/home/v/vdl21/projects/libs/

export CONDA_PREFIX="${env_dir}/cryovit_env"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

# Handle optional model and label_key arguments
if [ "$#" == 4 ]; then
    sample=$3

    # Setup W&B API key
    if [ -z "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY=$4
    fi

    $env_dir/cryovit_env/bin/python -m \
        cryovit.training.train_model \
        +experiments=$exp_name \
        model=$model \
        datamodule.sample=$sample
else
    # Setup W&B API key
    if [ -z "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY=$3
    fi

    $env_dir/cryovit_env/bin/python -m \
        cryovit.training.train_model \
        +experiments=$exp_name \
        model=$model
fi