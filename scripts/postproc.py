import os
import glob
import nibabel as nib
import pandas as pd
import numpy as np
from nilearn import signal
from joblib import Parallel, delayed
from dotenv import load_dotenv
from argparse import ArgumentParser

def load_confounds(task_file):
    """Load and extract relevant confound regressors from fMRIPrep confounds file."""
    task = os.path.basename(task_file).split('_task-')[1].split('_')[0]
    print(f"Processing task: {task}")
    confound_files = glob.glob(os.path.join(
        os.path.dirname(task_file), f"*_task-{task}_desc-confounds_timeseries.tsv"))
    if not confound_files:
        raise FileNotFoundError(f"No confounds file found for task: {task}")
    
    confounds_df = pd.read_csv(confound_files[0], sep='\t')
    compcor_cols = confounds_df.filter(regex='a_comp_cor_').columns[:5]
    motion_cols = confounds_df.filter(regex='motion_|trans_|rot_').columns
    cosine_cols = confounds_df.filter(regex='cosine').columns
    
    if len(compcor_cols) + len(motion_cols) + len(cosine_cols) == 0:
        raise ValueError("No matching confound columns found in confounds file")
    
    return confounds_df[list(compcor_cols) + list(motion_cols) + list(cosine_cols)]

def process_file(task_file, smoothing_mm, lh_surf, rh_surf, save_dir, global_zscore=False):
    cleaned_file_path = None
    try:
        confounds_df = load_confounds(task_file)
        task_img = nib.load(task_file)
        func_data = task_img.dataobj[:]
        confounds_df = confounds_df.fillna(0)

        if global_zscore:
            # Detrend and regress out confounds only
            cleaned_data = signal.clean(
                func_data,
                detrend=True,
                confounds=confounds_df,
                standardize=False,
                t_r=1.49
            )
            # Apply global z-score across all values
            global_mean = np.mean(cleaned_data)
            global_std = np.std(cleaned_data)
            cleaned_data = (cleaned_data - global_mean) / global_std
            suffix = '_cleaned_globalz.dtseries.nii'
        else:
            # Standard voxel-wise z-scoring
            cleaned_data = signal.clean(
                func_data,
                detrend=True,
                confounds=confounds_df,
                standardize='zscore_sample',
                t_r=1.49
            )
            suffix = '_cleaned.dtseries.nii'

        print(f"Cleaned {cleaned_data.shape}")
        task_cln = nib.Cifti2Image(cleaned_data, task_img.header)
        cleaned_file_path = os.path.join(save_dir, os.path.basename(task_file).replace('.dtseries.nii', suffix))
        nib.save(task_cln, cleaned_file_path)

    finally:
        print(cleaned_file_path)

def process_subject(subject_files, smoothing_mm, lh_surf, rh_surf, save_dir, global_zscore=False):
    if not subject_files:
        print("No subject files found.")
        return
    Parallel(n_jobs=-1)(
        delayed(process_file)(task_file, smoothing_mm, lh_surf, rh_surf, save_dir, global_zscore)
        for task_file in subject_files
    )

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(description="Postprocess fMRIPrep data")
    parser.add_argument("sub_id", help="Subject ID (e.g., sub-001)", type=str)
    parser.add_argument("task", help="Task name (e.g., friends)", type=str)
    parser.add_argument("--smoothing_mm", type=float, default=2.15, help="Smoothing kernel size in mm")
    parser.add_argument("--global_zscore", action="store_true", help="Apply global z-scoring instead of voxel-wise")

    args = parser.parse_args()
    sub_id = args.sub_id
    task = args.task
    smoothing_mm = args.smoothing_mm
    global_zscore = args.global_zscore

    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.getenv("DATA_DIR")

    if not base_dir or not scratch_dir or not data_dir:
        print("BASE_DIR or SCRATCH_DIR environment variables not set")
        exit(1)

    output_dir = os.path.join(scratch_dir, "output")
    save_subdir = f"{task}_cleaned_globalz" if global_zscore else f"{task}_cleaned"
    save_dir = os.path.join(data_dir, "cneuromod_postproc", save_subdir, sub_id)
    os.makedirs(save_dir, exist_ok=True)

    subject_files = glob.glob(os.path.join(
        data_dir, "cneuromod.processed", "fmriprep",
        f"{task}/{sub_id}/ses-*/func/*fsLR_den-91k_bold.dtseries.nii"))

    lh_surf = os.path.join(scratch_dir, "data", "subj_fsLR", f"{sub_id}_L.midthickness.32k_fs_LR.surf.gii")
    rh_surf = os.path.join(scratch_dir, "data", "subj_fsLR", f"{sub_id}_R.midthickness.32k_fs_LR.surf.gii")

    process_subject(subject_files, smoothing_mm, lh_surf, rh_surf, save_dir, global_zscore)
