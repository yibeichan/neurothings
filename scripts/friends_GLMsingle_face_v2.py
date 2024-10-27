import os
import glob
import time
import warnings
from os.path import join
from argparse import ArgumentParser
import json
import cProfile
import pstats
import io
import shutil
import logging
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

import glmsingle
from glmsingle.glmsingle import GLM_single

# Suppress warnings
warnings.filterwarnings('ignore')

# Add type hints for better code documentation
def get_ses_ids(sub_dir: str) -> list[str]:
    # Only include directories that match the session pattern (ses-XXX)
    ses_ids = [os.path.basename(d) for d in glob.glob(join(sub_dir, "ses-*")) if os.path.isdir(d)]
    print(f"Found {len(ses_ids)} sessions for {sub_dir}")
    return sorted(ses_ids)

def get_brain_files(sub_dir, ses_id):
    brain_files = sorted(glob.glob(join(sub_dir, ses_id, "func", "*-91k_bold.dtseries.nii")))
    # lets print all files in join(sub_dir, ses_id, "func")
    if not brain_files:
        raise FileNotFoundError(f"No brain files found for {ses_id}")
    return brain_files

def rename_y_files(files):
    """Rename misplaced stimuli. The stimulus order varies within each subject. Generates new filenames as the output."""
    replacements = {
        "ses-002_task-s01e06": "ses-002_task-s01e01",
        "ses-003_task-s01e06": "ses-003_task-s01e01",
        "ses-004_task-s01e06": "ses-004_task-s01e01"
    }
    
    # Create a new list for the updated filenames
    updated_files = []
    
    for file in files:
        updated_file = file
        for old, new in replacements.items():
            if old in file:
                updated_file = file.replace(old, new)
                break
        updated_files.append(updated_file)
        
    return updated_files

def load_design_matrix(dm_file):
    try:
        dm_data = np.load(dm_file, allow_pickle=True)
        return dm_data["dm_dict"].item()
    except Exception as e:
        raise ValueError(f"Error loading design matrix from {dm_file}: {str(e)}")

# Consider adding validation for the mask file existence
def process_brain_data(sub_id, brain_file):
    try:
        brain_data = nib.load(brain_file).get_fdata()
        mask_file = f"/orcd/scratch/bcs/001/yibei/hcptrt/output/GLMsingle/facemask/{sub_id}_wm_full_mask.npy"
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
        mask_data = np.load(mask_file)
        return brain_data[:, mask_data == 1].T
    except Exception as e:
        raise ValueError(f"Error processing brain data from {brain_file}: {str(e)}")

def retry_rmtree(path, max_retries=5, delay=1):
    for i in range(max_retries):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            if i == max_retries - 1:  # last attempt
                raise
            time.sleep(delay)

# Consider adding more logging for debugging
def process_session(sub_id, scratch_dir, ses_id, data_dict, n_jobs):
    logging.info(f"Processing session {ses_id} for subject {sub_id}")
    output_dir = join(scratch_dir, "output", "GLMsingle", "results", sub_id, ses_id)
    figure_dir = join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True) 
    os.makedirs(figure_dir, exist_ok=True)
    
    brain_files = get_brain_files(join(dataset_dir, "fmriprep", "friends", f"{sub_id}"), ses_id)
    design_matrix_files = rename_y_files(brain_files)
    
    brain_data = []
    design_matrices = []
    eids = []
    
    for b_file, dm_file in zip(brain_files, design_matrix_files):
        eid = os.path.basename(dm_file).split("_")[2].split("-")[1]
        logging.info(f"Processing file pair: {b_file}, {dm_file}")
        processed_data = process_brain_data(sub_id, b_file)
        # If this is the only run for the session (len(brain_files) == 1)
        if len(brain_files) == 1:
            # Split the data in half
            mid_point = processed_data.shape[1] // 2
            brain_data.append(processed_data[:, :mid_point])
            brain_data.append(processed_data[:, mid_point:])
            
            # Split the design matrix in half
            dm = data_dict[f"friends_{eid}"]
            design_matrices.append(dm[:mid_point, :])
            design_matrices.append(dm[mid_point:, :])
            
            # Add the same eid twice since we split one run into two
            eids.extend([eid, eid])
        else:
            # Multiple runs - proceed as before
            brain_data.append(processed_data)
            design_matrices.append(data_dict[f"friends_{eid}"])
            eids.append(eid)
    
    # Save eids to a JSON file
    eids_file = join(output_dir, f"{ses_id}_eids.json")
    with open(eids_file, 'w') as f:
        json.dump(eids, f)
    
    print(f"Saved eids for session {ses_id} to {eids_file}")

    # Continue with GLMsingle processing...
    stimdur = 2
    tr = 1.49
    
    start_time = time.time()

    opt = {
        'wantlibrary': 1,
        'wantglmdenoise': 1,
        'wantfracridge': 1,
        'wantfileoutputs': [0,0,0,0],
        'wantmemoryoutputs': [1,1,1,1],
        'n_jobs': n_jobs
    }
    glmsingle_obj = GLM_single(opt)
    print(f'running GLMsingle...')
    try:
        results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, outputdir=output_dir, figuredir=figure_dir)
    except OSError:
        print(f"Encountered file lock, attempting to remove directory: {output_dir}")
        retry_rmtree(output_dir)
        results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, outputdir=output_dir, figuredir=figure_dir)
    # save results to a file
    results_file = join(output_dir, "results.npz")
    np.savez(results_file, **results)
    print(f"Saved results for session {ses_id} to {results_file}")
    
    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

def get_last_session_position(data_dir, sub_id, ses_ids):
    # Path to GLMsingle results directory
    results_dir = join(data_dir, "output", "GLMsingle", "results", sub_id)
    
    if not os.path.exists(results_dir):
        logging.info("No results directory found, starting from first session")
        return 0
        
    # Get all completed session directories
    completed_sessions = [d for d in os.listdir(results_dir) if d.startswith('ses-')]
    if not completed_sessions:
        logging.info("Results directory empty, starting from first session")
        return 0
    
    # Ensure both lists are properly sorted
    completed_sessions.sort()  # Sort completed sessions
    last_session = completed_sessions[-1]  # Get the last completed session
    
    # Find position in sorted ses_ids
    try:
        position = ses_ids.index(last_session)
        return position
    except ValueError:
        logging.warning(f"Last completed session {last_session} not found in expected session list. Starting from beginning.")
        return 0

def main(sub_id, dataset_dir, scratch_dir, n_jobs):
    dm_file = join(scratch_dir, "output", "GLMsingle", "design_matrix_face.npz")
    data_dict = load_design_matrix(dm_file)
    ses_ids = get_ses_ids(join(dataset_dir, "fmriprep", "friends", f"{sub_id}"))
    print(f"Found {len(ses_ids)} sessions for {sub_id}")
    print(ses_ids)
    
    start_idx = get_last_session_position(scratch_dir, sub_id, ses_ids)
    if start_idx >= len(ses_ids):
        logging.info("All sessions have been processed")
        return
    elif start_idx > 0:
        logging.info(f"Resuming from session {ses_ids[start_idx]} (position {start_idx + 1} of {len(ses_ids)})")
    else:
        logging.info(f"Starting from first session (total sessions: {len(ses_ids)})")
    
    
    for ses_id in tqdm(ses_ids[start_idx:], desc="Processing sessions"):
        process_session(sub_id, scratch_dir, ses_id, data_dict, n_jobs)

if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser(description="Fit GLMsingle to the data.")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("--n_jobs", help="Number of jobs", type=int, default=32)
    args = parser.parse_args()

    scratch_dir = os.path.join(os.getenv("SCRATCH_DIR"), "friends")
    dataset_dir = os.getenv("DATASET_DIR")
    # Start CPU profiling
    pr = cProfile.Profile()
    pr.enable()

    # Record start time
    start_time = time.time()

    # Run the main function
    main(args.sub_id, dataset_dir, scratch_dir, args.n_jobs)

    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Stop CPU profiling
    pr.disable()

    # Print execution time
    print(f"Total execution time: {duration:.2f} seconds")

    # Print CPU profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
