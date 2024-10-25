#!/bin/bash

#SBATCH --job-name=GLMsingle_face
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/GLMsingle_face_%A_%a.out
#SBATCH --error=../logs/GLMsingle_face_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=4G
#SBATCH --array=30-59
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate glmsingle

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
hemi_ids=("lh" "rh")
roi_ids=("ofa" "ffa" "sts" "v1" "tl") 

sub_index=$((SLURM_ARRAY_TASK_ID / 10))
hemi_index=$(((SLURM_ARRAY_TASK_ID % 10) / 5))
roi_index=$((SLURM_ARRAY_TASK_ID % 5))

TASK_ID=${sub_ids[$sub_index]}
hemi_id=${hemi_ids[$hemi_index]}
roi_id=${roi_ids[$roi_index]}

echo "Processing: $TASK_ID, $hemi_id, $roi_id"

python GLMsingle_face.py "${TASK_ID}" "${hemi_id}" "${roi_id}"

if [ $? -eq 0 ]; then
  echo "Python job completed successfully for $TASK_ID, $hemi_id, $roi_id."

  # Create the destination directory if it doesn't exist
  dest_dir="/orcd/data/satra/001/users/yibei/friends/output/GLMsingle/results/${TASK_ID}/${hemi_id}_${roi_id}/"
  mkdir -p "$dest_dir"

  # Rsync the output folder to the remote server
  rsync -avz /orcd/scratch/bcs/001/yibei/friends/output/GLMsingle/results/${TASK_ID}/${hemi_id}_${roi_id}/ "$dest_dir"

else
  echo "Python job failed for $TASK_ID, $hemi_id, $roi_id, not running rsync."
fi
