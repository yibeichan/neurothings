import os
import glob
import nibabel as nib
import pandas as pd
from nilearn import signal
import subprocess
import tempfile
from joblib import Parallel, delayed
from dotenv import load_dotenv
from argparse import ArgumentParser
import numpy as np

def load_confounds(task_file):
    """Load and extract relevant confound regressors from fMRIPrep confounds file.
    
    Args:
        task_file (str): Path to the task file
        
    Returns:
        pd.DataFrame: DataFrame containing selected confound regressors
        
    Raises:
        FileNotFoundError: If confounds file cannot be found
        ValueError: If required confound columns are missing
    """
    # Extract task name from file path
    task = os.path.basename(task_file).split('_')[2].split('-')[1]
    print(f"Processing task: {task}")
    # Find confounds file
    confound_files = glob.glob(os.path.join(os.path.dirname(task_file), 
                                          f"*_task-{task}_desc-confounds_timeseries.tsv"))
    if not confound_files:
        raise FileNotFoundError(f"No confounds file found for task: {task}")
    
    # Load confounds data
    confounds_df = pd.read_csv(confound_files[0], sep='\t')
    
    # Select relevant columns
    compcor_cols = confounds_df.filter(regex='a_comp_cor_').columns
    wm_csf_cols = confounds_df.filter(regex='csf_|white_matter_').columns
    motion_cols = confounds_df.filter(regex='motion_|trans_|rot_').columns
    cosine_cols = confounds_df.filter(regex='cosine').columns
    
    # Verify we found some columns
    if len(compcor_cols) + len(wm_csf_cols) + len(motion_cols) + len(cosine_cols) == 0:
        raise ValueError("No matching confound columns found in confounds file")
    
    # Return selected columns
    return confounds_df[list(compcor_cols) + list(wm_csf_cols) + list(motion_cols) + list(cosine_cols)]

def process_file(task_file, smoothing_mm, lh_surf, rh_surf, save_dir):
    cleaned_file_path = None
    try:
        # Single call to load_confounds
        confounds_df = load_confounds(task_file)
        task_img = nib.load(task_file)
        confounds = confounds_df.to_numpy()
        
        # Before the signal.clean() call
        if np.any(np.isnan(confounds)) or np.any(np.isinf(confounds)):
            print("Warning: Found NaN or inf values in confounds")
            # Either remove those timepoints or replace with zeros/means
            confounds = np.nan_to_num(confounds)  # replaces NaN/inf with finite numbers
        
        clean_func_signal = signal.clean(task_img.dataobj[:], 
                                       detrend=True, 
                                       confounds=confounds, 
                                       standardize='zscore_sample', 
                                       t_r=1.49)
        print(f"Cleaned {task_file}")
        
        task_cln = nib.Cifti2Image(clean_func_signal, task_img.header)
        with tempfile.NamedTemporaryFile(suffix='.dtseries.nii', delete=False) as tmpfile:
            cleaned_file_path = tmpfile.name
            nib.save(task_cln, cleaned_file_path)
        
        smooth_output_file = os.path.join(save_dir, 
                                        os.path.basename(task_file).replace('.dtseries.nii', 
                                                                         '_cleaned_smoothed.dtseries.nii'))
        smoothing_command = f"wb_command -cifti-smoothing {cleaned_file_path} {smoothing_mm} {smoothing_mm} COLUMN {smooth_output_file} -left-surface {lh_surf} -right-surface {rh_surf}"
        
        subprocess.run(smoothing_command, shell=True, check=True)
        print(f"Smoothed {task_file}")
    finally:
        if cleaned_file_path and os.path.exists(cleaned_file_path):
            os.remove(cleaned_file_path)

def process_subject(subject_files, smoothing_mm, lh_surf, rh_surf, save_dir):
    if not subject_files:
        print("No subject files found.")
        return
    Parallel(n_jobs=-1)(
        delayed(process_file)(task_file, smoothing_mm, lh_surf, rh_surf, save_dir) for task_file in subject_files)

if __name__ == "__main__":
    
    load_dotenv()
    
    parser = ArgumentParser(description="Postprocess fmriprep data")
    parser.add_argument("sub_id", help="Subject ID (e.g., sub-001)", type=str)
    parser.add_argument("task", help="Task name (e.g., hcptrt)", type=str)
    parser.add_argument("--smoothing_mm", type=float, default=2.15, help="Smoothing kernel size in mm")
    args = parser.parse_args()
    
    sub_id = args.sub_id
    task = args.task
    smoothing_mm = args.smoothing_mm
    
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    nese_dir = os.getenv("NESE_DIR")
    
    if not base_dir or not scratch_dir:
        print("BASE_DIR or SCRATCH_DIR environment variables not set")
        exit(1)
    
    data_dir = os.path.join(scratch_dir, "data")
    output_dir = os.path.join(scratch_dir, "output")
    save_dir = os.path.join(output_dir, f"{task}_postproc", f"{sub_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    subject_files = glob.glob(os.path.join(nese_dir, f"{task}.fmriprep/{sub_id}/ses-*/func/*fsLR_den-91k_bold.dtseries.nii"))
    lh_surf = os.path.join(data_dir, f"subj_fsLR/{sub_id}_L.midthickness.32k_fs_LR.surf.gii")
    rh_surf = os.path.join(data_dir, f"subj_fsLR/{sub_id}_R.midthickness.32k_fs_LR.surf.gii")
    
    process_subject(subject_files, smoothing_mm, lh_surf, rh_surf, save_dir)
