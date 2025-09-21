#!/bin/bash

sample=$1
split_id=$2
model=$3
label_key=$4

# Setup W&B API key
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$5
fi

env_dir=/sdf/home/v/vdl21/projects/libs/

export CONDA_PREFIX="${env_dir}/cryovit_env"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.train_model \
    model=$model \
    name="fractional_${model}_${label_key}" \
    label_key=$label_key \
    datamodule=fractional_loo \
    datamodule.sample=$sample \
    datamodule.split_id=$split_id