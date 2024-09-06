import os
import glob
import nibabel as nib
from nilearn import signal
from nilearn.interfaces.fmriprep import load_confounds_strategy
import subprocess
import tempfile
from joblib import Parallel, delayed
from dotenv import load_dotenv
from argparse import ArgumentParser

def process_file(task_file, smoothing_mm, lh_surf, rh_surf, save_dir):
    try:
        confounds_df, _ = load_confounds_strategy(task_file, denoise_strategy="compcor")
        task_img = nib.load(task_file)
        clean_func_signal = signal.clean(task_img.dataobj[:], detrend=True, confounds=confounds_df, standardize='zscore_sample', t_r=1.49)
        print(f"Cleaned {task_file}")
        
        task_cln = nib.Cifti2Image(clean_func_signal, task_img.header)
        with tempfile.NamedTemporaryFile(suffix='.dtseries.nii', delete=False) as tmpfile:
            cleaned_file_path = tmpfile.name
        
        nib.save(task_cln, cleaned_file_path)
        smooth_output_file = os.path.join(save_dir, os.path.basename(task_file).replace('.dtseries.nii', '_cleaned_smoothed.dtseries.nii'))
        smoothing_command = f"wb_command -cifti-smoothing {cleaned_file_path} {smoothing_mm} {smoothing_mm} COLUMN {smooth_output_file} -left-surface {lh_surf} -right-surface {rh_surf}"
        
        subprocess.run(smoothing_command, shell=True, check=True)
        print(f"Smoothed {task_file}")
    finally:
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
    parser.add_argument("--smoothing_mm", type=int, default=2, help="Smoothing kernel size in mm")
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