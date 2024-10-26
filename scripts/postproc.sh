#!/bin/bash

#SBATCH --job-name=postproc_friends
#SBATCH --partition=normal
#SBATCH --output=../logs/postproc_friends_%j.out
#SBATCH --error=../logs/postproc_friends_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate neuro

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $TASK_ID"

python postproc.py "${TASK_ID}" "friends"

if [ $? -ne 0 ]; then
    echo "Error: Python script failed for ${TASK_ID}"
    exit 1
fi

# Add check for rsync success
dest_dir="/nese/mit/group/sig/projects/cneuromod/postproc_fmriprep/friends"
rsync -avL /om2/scratch/tmp/yibei/friends/output/friends_postproc/ "${dest_dir}"
if [ $? -ne 0 ]; then
    echo "Error: rsync failed"
    exit 1
fi