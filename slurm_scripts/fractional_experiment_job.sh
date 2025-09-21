#!/bin/bash

exp_name=$1
sample=$2
split_id=$3
model=$4
label_key=$5

# Setup W&B API key
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$6
fi

# Setup environment
env_dir=/sdf/home/v/vdl21/projects/libs/

export CONDA_PREFIX="${env_dir}/cryovit_env"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.train_model \
    +experiments=$exp_name \
    model=$model \
    datamodule.test_sample=$sample \
    datamodule.split_id=$split_id \
    label_key=$label_key