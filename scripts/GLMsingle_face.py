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
from memory_profiler import profile

import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

import glmsingle
from glmsingle.glmsingle import GLM_single

# Suppress warnings
warnings.filterwarnings('ignore')

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
    

@profile
def main(sub_id, hemi, roi, data_dir, nese_dir):
    dm_file = join(nese_dir, "char_rep_friends", "output", "GLMsingle", "design_matrix_face.npz")
    data_dict = load_design_matrix(dm_file)
    ses_ids = get_ses_ids(join(data_dir, "output", "time_series_v1", sub_id), hemi, roi)    
    for ses_id in tqdm(ses_ids, desc="Processing sessions"):  
        output_dir = join(nese_dir, "char_rep_friends", "output", "GLMsingle", "results", sub_id, ses_id)
        os.makedirs(output_dir, exist_ok=True) 
        
        brain_files = get_brain_files(data_dir, sub_id, ses_id, hemi, roi)
        design_matrix_files = rename_y_files(brain_files)
        
        brain_data = []
        design_matrices = []
        eids = []
        
        for b_file, dm_file in zip(brain_files, design_matrix_files):
            eid = os.path.basename(dm_file).split("_")[2].split("-")[1]
            processed_data = process_brain_data(b_file)
            print(f"Processed data shape: {processed_data.shape}")
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
            'n_jobs': 12
        }
        glmsingle_obj = GLM_single(opt)
        print(f'running GLMsingle...')
        results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, outputdir=output_dir)
        # save results to a file
        results_file = join(output_dir, f"{hemi}_{roi}_results.npz")
        np.savez(results_file, **results)
        print(f"Saved results for session {ses_id} to {results_file}")
        
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser(description="Fit GLMsingle to the data.")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("hemi", help="Hemisphere", type=str)
    parser.add_argument("roi", help="Region of interest", type=str)
    args = parser.parse_args()
    sub_id = args.sub_id
    hemi = args.hemi
    roi = args.roi

    data_dir = os.getenv("SCRATCH_DIR")
    nese_dir = os.getenv("NESE_DIR")

    # Start CPU profiling
    pr = cProfile.Profile()
    pr.enable()

    # Record start time
    start_time = time.time()

    # Run the main function
    main(sub_id, hemi, roi, data_dir, nese_dir)

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
