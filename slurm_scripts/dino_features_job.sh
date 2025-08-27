#!/bin/bash

sample=$1

env_dir=/sdf/home/v/vdl21/projects/libs/

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.dino_features \
    sample=$sample \
    paths.tomo_name=processed_with_annot \
    paths.feature_name=tomograms