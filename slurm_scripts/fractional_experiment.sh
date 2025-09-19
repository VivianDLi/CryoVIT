#!/bin/bash

# Check if four arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 model (cryovit, unet3d, sam2, or medsam) label_key (mito, microtubule, cristae, granule, bacteria) wandb_api_key"
    exit 1
fi

case $2 in
    "mito")
        samples=(
            "BACHD"
            "dN17_BACHD"
            "Q109"
            "Q18"
            "Q20"
            "Q53"
            "Q53_KD"
            "Q66"
            "Q66_GRFS1"
            "Q66_KD"
            "WT"
            "RGC_CM"
            "RGC_control"
            "RGC_naPP"
            "RGC_PP"
            "AD"
            "AD_Abeta"
            "Aged"
            "Young"
            "CZI_Algae"
        )
        exp_name="fractional_mito"
        ;;
    "microtubule")
        samples=(
            "Q109_Microtubules"
            "Q18_Microtubules"
            "BACHD_Microtubules"
            "WT_Microtubules"
            "AD"
            "AD_Abeta"
            "Aged"
            "Young"
        )
        exp_name="fractional_microtubule"
        ;;
    "cristae")
        samples=(
            "Q18"
            "Q53"
            "AD"
            "AD_Abeta"
            "Aged"
            "Young"
        )
        exp_name="fractional_cristae"
        ;;
    "granule")
        samples=(
            "BACHD"
            "Q109"
            "Q18"
            "Q20"
            "Q53"
            "Q66"
            "WT"
        )
        exp_name="fractional_granule"
        ;;
    "bacteria")
        samples=(
            "CZI_Campy_C"
            "CZI_Campy_CDel"
            "CZI_Campy_F"
        )
        exp_name="fractional_bacteria"
        ;;
    *)
        echo "Invalid label_key. Choose from: mito, microtubule, cristae, granule, bacteria."
        exit 1
        ;;
esac

max_jobs=1024  # Maximum concurrent jobs
total_jobs=$(( ${#samples[@]} * 10 ))
current_job=0

for sample in "${samples[@]}"; do
    for split_id in {1..10}; do
        # Check the number of running jobs
        while [ $(squeue -u $USER --noheader | wc -l) -ge $max_jobs ]; do
            sleep 10  # Wait for 10 seconds before checking again
        done

        exp_cmd="$(dirname "$0")/fractional_experiment_job.sh $exp_name $sample $split_id $1 $2 $3"
        job_name="fractional_experiment_${1}_${2}_${sample}_${split_id}"
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
            --time=6:00:00 \
            --wrap="$exp_cmd"

        ((current_job++))
        echo Job $current_job / $total_jobs: \
            sample=$sample, \
            split_id=$split_id
    done
done