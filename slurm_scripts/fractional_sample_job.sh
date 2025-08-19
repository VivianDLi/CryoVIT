#!/bin/bash

sample=$1
split_id=$2
sample_group=$3
model=$4
label_key=$5

env_dir=/sdf/home/v/vdl21/projects/libs/

$env_dir/cryovit_env/bin/python -m \
    cryovit.train_model \
    model=$model \
    name="fractional_${sample_group}_${model}_${label_key}" \
    label_key=$label_key \
    datamodule=fractional_loo \
    datamodule.sample=$sample \
    datamodule.split_id=$split_id