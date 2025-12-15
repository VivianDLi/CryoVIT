#!/bin/bash

sample=$1

env_dir=/sdf/home/v/vdl21/projects/libs/

$env_dir/cryovit_env/bin/python -m \
    cryovit.training.sam_features \
    sample=$sample \
    paths.feature_name=processed_with_annot