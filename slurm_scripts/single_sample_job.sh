#!/bin/bash

sample=$1
split_id=$2
sample_group=$3
model=$4
label_key=$5

env_dir=/sdf/home/v/vdl21/projects/libs/

export CONDA_PREFIX="${env_dir}/cryovit_env"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

# Setup W&B API key
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$6
fi

export CONDA_PREFIX="${env_dir}/cryovit_env"

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.train_model \
    model=$model \
    name="single_${sample_group}_${model}_${label_key}" \
    label_key=$label_key \
    datamodule=single \
    datamodule.sample=$sample \
    datamodule.split_id=$split_id

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.eval_model \
    model=$model \
    name="single_${sample_group}_${model}_${label_key}" \
    label_key=$label_key \
    datamodule=single \
    datamodule.sample=$sample \
    datamodule.split_id=$split_id