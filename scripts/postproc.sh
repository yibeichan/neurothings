#!/bin/bash

#SBATCH --job-name=postproc_friends
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/postproc_friends_%j.out
#SBATCH --error=../logs/postproc_friends_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source ~/.bashrc  # or source /etc/profile.d/conda.sh
micromamba activate friends-states

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $TASK_ID"

python 00_postproc.py "${TASK_ID}" "friends" --global_zscore
