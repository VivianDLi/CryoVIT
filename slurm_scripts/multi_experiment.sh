#!/bin/bash

# Check if four arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 model (cryovit, unet3d, sam2, or medsam) label_key (mito, microtubule, cristae, granule, bacteria) wandb_api_key"
    exit 1
fi

case $2 in
    "mito")
        exp_name="multi_mito"
        ;;
    "microtubule")
        exp_name="multi_microtubule"
        ;;
    "cristae")
        exp_name="multi_cristae"
        ;;
    "granule")
        exp_name="multi_granule"
        ;;
    "bacteria")
        exp_name="multi_bacteria"
        ;;
    *)
        echo "Invalid label_key. Choose from: mito, microtubule, cristae, granule, bacteria."
        exit 1
        ;;
esac

max_jobs=1024  # Maximum concurrent jobs
total_jobs=$(( ${#samples[@]} * 10 ))
current_job=0

for split_id in {0..9}; do
    # Check the number of running jobs
    while [ $(squeue -u $USER --noheader | wc -l) -ge $max_jobs ]; do
        sleep 10  # Wait for 10 seconds before checking again
    done

    exp_cmd="$(dirname "$0")/multi_experiment_job.sh $exp_name $split_id $1 $2 $3"
    job_name="multi_experiment_${1}_${2}_${split_id}"
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
        --time=12:00:00 \
        --wrap="$exp_cmd"

    ((current_job++))
    echo Job $current_job / $total_jobs: \
        split_id=$split_id
done