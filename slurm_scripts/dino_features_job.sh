#!/bin/bash

sample=$1

env_dir=/tmp/$USER/"$(uuidgen)"
mkdir -p $env_dir
tar -xf ~/projects/libs/cryovit_env.tar -C $env_dir

$env_dir/cryovit_env/bin/python -m \
    cryovit.dino_features \
    sample=$sample \
    paths.tomo_name=processed_with_annot \
    paths.feature_name=tomograms

rm -rf $env_dir