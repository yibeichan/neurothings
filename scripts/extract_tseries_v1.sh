#!/bin/bash

#SBATCH --job-name=extract_ts_v1
#SBATCH --partition=normal
#SBATCH --output=../logs/extract_ts_v1_%j.out
#SBATCH --error=../logs/extract_ts_v1_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate friends-cnm

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "Processing: $TASK_ID"

python extract_tseries_v1.py "${TASK_ID}"

if [ $? -eq 0 ]; then
  echo "Python job completed successfully for $TASK_ID."

  # Rsync the output folder to a remote server
  rsync -avz /om2/scratch/tmp/yibei/friends/output/time_series_v1/${TASK_ID}/ /nese/mit/group/sig/projects/cneuromod/char_rep_friends/output/GLMsingle/time_series_v1/${TASK_ID}/

else
  echo "Python job failed for $TASK_ID, not running rsync."
fi