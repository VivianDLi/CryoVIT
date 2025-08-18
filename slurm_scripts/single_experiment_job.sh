#!/bin/bash

exp_name=$1

# Setup environment
env_dir=/tmp/$USER/"$(uuidgen)"
mkdir -p $env_dir
tar -xf ~/projects/libs/cryovit_env.tar -C $env_dir

# Handle optional model and label_key arguments
if [ "$#" == 3 ]; then
    model=$2
    sample=$3

    # Setup W&B API key
    if [ -z "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY=$4
    fi

    $env_dir/cryovit_env/bin/python -m \
        cryovit.train_model \
        +experiments=$exp_name \
        model=$model \
        datamodule.sample=$sample
else
    # Setup W&B API key
    if [ -z "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY=$2
    fi

    $env_dir/cryovit_env/bin/python -m \
        cryovit.train_model \
        +experiments=$exp_name
fi

rm -rf $env_dir