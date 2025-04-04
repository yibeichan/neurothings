#!/bin/bash

#SBATCH --job-name=GLMsingle_localizer_face_v3  
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/GLMsingle_localizer_face_v3_%A_%a.out
#SBATCH --error=../logs/GLMsingle_localizer_face_v3_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=12G
#SBATCH --array=0
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

sub_index=$SLURM_ARRAY_TASK_ID

TASK_ID=${sub_ids[$sub_index]}

echo "Processing: $TASK_ID"

# Run first script
echo "Activating environment for first script..."
conda activate glmsingle
if ! python hcptrt_GLMsingle_localizer_face_v3.py "${TASK_ID}"; then
    echo "First script failed! Exiting..."
    conda deactivate
    exit 1
fi
conda deactivate

# # Only reach here if both scripts succeeded
# echo "Both scripts succeeded. Starting data transfer..."
# dest_dir="/orcd/data/satra/001/users/yibei/hcptrt/output/GLMsingle_v3/results/${TASK_ID}/"
# mkdir -p "$dest_dir"
# rsync -avz /orcd/scratch/bcs/001/yibei/hcptrt/output/GLMsingle_v3/results/${TASK_ID}/ "$dest_dir"

# echo "All operations completed successfully!"
