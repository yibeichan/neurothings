#!/bin/bash

#SBATCH --job-name=face_localizer
#SBATCH --partition=normal
#SBATCH --output=../logs/face_localizer_%j.out
#SBATCH --error=../logs/face_localizer_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate neuro

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $TASK_ID"

python hcptrt_face_localizer.py "${TASK_ID}" 