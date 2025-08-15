#!/bin/bash

# Check if 0 or 1 arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 sample (or null for all samples)"
    exit 1
fi

exp_cmd="$(dirname "$0")/dino_features_job.sh $1"
job_name="dino_features_${1}"
out_dir="$(dirname "$0")/outputs"

sbatch \
    --partition="cryoem" \
    --job-name="$job_name" \
    --output="${out_dir}/${job_name}.out" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem-per-cpu=6gb \
    --gres=gpu:v100:1 \
    --time=02:00:00 \
    --wrap="$exp_cmd"