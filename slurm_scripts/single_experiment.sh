#!/bin/bash

# Check if 1 or 3 arguments are provided
if [ "$#" -ne 2 ] && [ "$#" -ne 4 ]; then
    echo "Usage: $0 experiment_name (see configs/experiments) model (optional; cryovit, unet3d, sam2, or medsam) sample (optional) wandb_api_key"
    exit 1
fi

if [ "$#" -eq 4 ]; then
    exp_cmd="$(dirname "$0")/single_experiment_job.sh $1 $2 $3 $4"
    job_name="single_experiment_${1}_${2}_${3}"
else
    exp_cmd="$(dirname "$0")/single_experiment_job.sh $1 $2"
    job_name="single_experiment_${1}"
fi
out_dir="$(dirname "$0")/outputs"

sbatch \
    --partition="ampere" \
    --account="cryoem:C073" \
    --job-name="$job_name" \
    --output="${out_dir}/${job_name}.out" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem-per-cpu=6gb \
    --gres=gpu:a100:1 \
    --time=04:00:00 \
    --wrap="$exp_cmd"