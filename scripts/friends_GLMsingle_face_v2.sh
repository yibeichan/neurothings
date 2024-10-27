#!/bin/bash

#SBATCH --job-name=friends_GLMsingle_face_v2
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/friends_GLMsingle_face_v2_%A_%a.out
#SBATCH --error=../logs/friends_GLMsingle_face_v2_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=12G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate glmsingle

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $TASK_ID"

python friends_GLMsingle_face_v2.py "${TASK_ID}"

if [ $? -eq 0 ]; then
  echo "Python job completed successfully for $TASK_ID."

  # Create the destination directory if it doesn't exist
  dest_dir="/orcd/data/satra/001/users/yibei/friends/output/GLMsingle/results/${TASK_ID}/"
  mkdir -p "$dest_dir"

  # Rsync the output folder to the remote server
  rsync -avz /orcd/scratch/bcs/001/yibei/friends/output/GLMsingle/results/${TASK_ID}/ "$dest_dir"

else
  echo "Python job failed for $TASK_ID, not running rsync."
fi
