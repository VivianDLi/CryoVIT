#!/bin/bash

exp_name=$1
split_id=$2
model=$3
label_key=$4

# Setup W&B API key
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$5
fi

# Setup environment
env_dir=/sdf/home/v/vdl21/projects/libs/

export CONDA_PREFIX="${env_dir}/cryovit_env"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.train_model \
    +experiments=$exp_name \
    model=$model \
    datamodule.split_id=$split_id \
    label_key=$label_key

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.eval_model \
    +experiments=$exp_name \
    model=$model \
    datamodule.split_id=$split_id \
    label_key=$label_key