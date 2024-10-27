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
import multiprocessing
import logging
from datetime import datetime

import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

import glmsingle
from glmsingle.glmsingle import GLM_single

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_logging(sub_id, hemi, roi, data_dir):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = join(data_dir, "output", "GLMsingle", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = join(log_dir, f"GLMsingle_{sub_id}_{hemi}_{roi}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    logging.info(f"Starting GLMsingle processing for sub-{sub_id}, {hemi}, {roi}")

def get_ses_ids(sub_dir, hemi, roi):
    # List all .nii files in the sub_dir
    files = glob.glob(join(sub_dir, f"*{hemi}_{roi}_tseries.nii"))
    print(files)
    # Extract unique session IDs from the filenames
    ses_ids = set()
    for file in files:
        # Split the filename and extract the session part
        part = os.path.basename(file).split("_")[1]
        ses_ids.add(part)
    
    # Convert to list and sort to ensure consistent order
    ses_ids = sorted(list(ses_ids))    
    if not ses_ids:
        raise ValueError(f"No session IDs found in {sub_dir}")
    
    return ses_ids

def get_brain_files(data_dir, sub_id, ses_id, hemi, roi):
    brain_files = glob.glob(join(data_dir, "output", "time_series_v1", sub_id, f"*{ses_id}*{hemi}_{roi}_tseries.nii"))
    if not brain_files:
        raise FileNotFoundError(f"No brain files found for {sub_id}, {ses_id}, {hemi}, {roi}")
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

def process_brain_data(brain_file):
    try:
        brain_data = nib.load(brain_file).get_fdata().T # Transpose the data to (n_vertices, n_timepoints)
        # Filter vertices with any non-zero values across all time points
        non_zero_vertices = np.any(brain_data > 0, axis=1)
        return brain_data[non_zero_vertices, :]
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

def process_session(sub_id, hemi, roi, data_dir, ses_id, data_dict):
    """Process a single session"""
    # Setup output directories
    output_dir = join(data_dir, "output", "GLMsingle", "results", sub_id, f"{hemi}_{roi}", ses_id)
    figure_dir = join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True) 
    os.makedirs(figure_dir, exist_ok=True)
    
    logging.info(f"Processing session {ses_id}")
    
    try:
        brain_files = get_brain_files(data_dir, sub_id, ses_id, hemi, roi)
        design_matrix_files = rename_y_files(brain_files)
        
        brain_data = []
        design_matrices = []
        eids = []
        
        for b_file, dm_file in zip(brain_files, design_matrix_files):
            eid = os.path.basename(dm_file).split("_")[2].split("-")[1]
            processed_data = process_brain_data(b_file)
            logging.info(f"Processed data shape for {eid}: {processed_data.shape}")
            
            if len(brain_files) == 1:
                mid_point = processed_data.shape[1] // 2
                brain_data.append(processed_data[:, :mid_point])
                brain_data.append(processed_data[:, mid_point:])
                
                dm = data_dict[f"friends_{eid}"]
                design_matrices.append(dm[:mid_point, :])
                design_matrices.append(dm[mid_point:, :])
                
                eids.extend([eid, eid])
                logging.info(f"Split single run into two parts for {eid}")
            else:
                brain_data.append(processed_data)
                design_matrices.append(data_dict[f"friends_{eid}"])
                eids.append(eid)
        
        # Save eids to a JSON file
        eids_file = join(output_dir, f"{ses_id}_eids.json")
        with open(eids_file, 'w') as f:
            json.dump(eids, f)
        
        logging.info(f"Saved eids for session {ses_id}")

        # GLMsingle processing
        stimdur = 2
        tr = 1.49
        
        opt = {
            'wantlibrary': 1,
            'wantglmdenoise': 1,
            'wantfracridge': 1,
            'wantfileoutputs': [0,0,0,0],
            'wantmemoryoutputs': [0,0,0,1],
            'n_jobs': 12
        }
        
        glmsingle_obj = GLM_single(opt)
        logging.info('Running GLMsingle...')
        
        try:
            results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, 
                                      outputdir=output_dir, figuredir=figure_dir)
        except OSError:
            logging.warning(f"Encountered file lock, attempting to remove directory: {output_dir}")
            retry_rmtree(output_dir)
            results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, 
                                      outputdir=output_dir, figuredir=figure_dir)
        
        # Save results
        results_file = join(output_dir, f"{hemi}_{roi}_results.npz")
        np.savez(results_file, **results)
        logging.info(f"Saved results for session {ses_id}")
        
    except Exception as e:
        logging.error(f"Error processing session {ses_id}: {str(e)}", exc_info=True)
        raise

def get_last_session_position(data_dir, sub_id, hemi, roi, ses_ids):
    # Path to GLMsingle results directory
    results_dir = join(data_dir, "output", "GLMsingle", "results", sub_id, f"{hemi}_{roi}")
    
    if not os.path.exists(results_dir):
        return 0
        
    # Get the highest session number from the directories
    completed_sessions = [d for d in os.listdir(results_dir) if d.startswith('ses-')]
    if not completed_sessions:
        return 0
    
    last_session = max(completed_sessions)
    # Find where this session is in ses_ids
    try:
        return ses_ids.index(last_session)
    except ValueError:
        return 0

def main(sub_id, hemi, roi, data_dir):
    # Setup logging
    setup_logging(sub_id, hemi, roi, data_dir)
    
    try:
        dm_file = join(data_dir, "output", "GLMsingle", "design_matrix_face.npz")
        data_dict = load_design_matrix(dm_file)
        ses_ids = get_ses_ids(join(data_dir, "output", "time_series_v1", sub_id), hemi, roi)    
        
        start_idx = get_last_session_position(data_dir, sub_id, hemi, roi, ses_ids)
        if start_idx > 0:
            logging.info(f"Resuming from session {ses_ids[start_idx]}")
        else:
            logging.info("Starting from the beginning")
        
        for ses_id in tqdm(ses_ids[start_idx:], desc="Processing sessions"):
            process_session(sub_id, hemi, roi, data_dir, ses_id, data_dict)
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser(description="Fit GLMsingle to the data.")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("hemi", help="Hemisphere", type=str)
    parser.add_argument("roi", help="Region of interest", type=str)
    args = parser.parse_args()
    
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = join(scratch_dir, "friends")
    
    # Start CPU profiling
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.time()

    try:
        main(args.sub_id, args.hemi, args.roi, data_dir)
    finally:
        # Always execute these cleanup/logging steps
        end_time = time.time()
        duration = end_time - start_time
        
        pr.disable()
        logging.info(f"Total execution time: {duration:.2f} seconds")
        
        # Print CPU profiling results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        logging.info("CPU Profiling Results:\n" + s.getvalue())
