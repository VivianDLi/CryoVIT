#!/bin/bash

sample=$1
split_id=$2
sample_group=$3
model=$4
label_key=$5

env_dir=/tmp/$USER/"$(uuidgen)"
mkdir -p $env_dir
tar -xf ~/projects/libs/cryovit_env.tar -C $env_dir

$env_dir/cryovit_env/bin/python -m \
    cryovit.train_model \
    model=$model \
    name="single_${sample_group}_${model}_${label_key}" \
    label_key=$label_key \
    datamodule=single \
    datamodule.sample=$sample \
    datamodule.split_id=$split_id

rm -rf $env_dir