#!/bin/bash
#SBATCH --job-name=GLMsingle_localizer_emotion  
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/GLMsingle_localizer_emotion_%A_%a.out
#SBATCH --error=../logs/GLMsingle_localizer_emotion_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=12G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/.bashrc

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

sub_index=$SLURM_ARRAY_TASK_ID

TASK_ID=${sub_ids[$sub_index]}

echo "Processing: $TASK_ID"

# Run first script
echo "Activating environment for first script..."
mamba activate glmsingle
if ! python hcptrt_GLMsingle_localizer_emotion.py "${TASK_ID}"; then
    echo "First script failed! Exiting..."
    mamba deactivate
    exit 1
fi
mamba deactivate

# # Only reach here if both scripts succeeded
# echo "Both scripts succeeded. Starting data transfer..."
# dest_dir="/orcd/data/satra/001/users/yibei/hcptrt/output/GLMsingle/results_v2/${TASK_ID}/"
# mkdir -p "$dest_dir"
# rsync -avz --ignore-existing /orcd/scratch/bcs/001/yibei/hcptrt/output/GLMsingle/results_v2/${TASK_ID}/ "$dest_dir"

# echo "All operations completed successfully!"
