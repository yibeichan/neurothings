import os
import glob
import pandas as pd
import numpy as np
import json
import time
from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv

import nibabel as nib
import glmsingle
from glmsingle.glmsingle import GLM_single

def get_binary_df(df: pd.DataFrame, n_tr: int, sorted_unique_stim_types: List[str]) -> pd.DataFrame:
    """
    Create a binary DataFrame representing stimulus occurrences.

    Args:
        df (pd.DataFrame): Input DataFrame with onset and stimulus information.
        n_tr (int): Number of time points.
        sorted_unique_stim_types (List[str]): Sorted list of unique stimulus types.

    Returns:
        pd.DataFrame: Binary DataFrame with TRs as rows and stimulus types as columns.
    """
    binary_df = pd.DataFrame(0, index=np.arange(n_tr), columns=sorted_unique_stim_types)
    
    for row in df.itertuples():
        onset = int(row.onset_TR)
        stim_type = row.stim_type
        if pd.notna(stim_type):
            binary_df.at[onset, stim_type] = 1
    return binary_df

def load_event_and_json_files(event_files: List[str], json_files: List[str], brain_files: List[str], tr: float) -> List[np.ndarray]:
    """
    Load event and JSON files and create design matrices.

    Args:
        event_files (List[str]): List of event file paths.
        json_files (List[str]): List of JSON file paths.
        brain_files (List[str]): List of brain file paths.
        tr (float): Repetition time.

    Returns:
        List[np.ndarray]: List of design matrices.
    """
    design_matrices = []
    for i, (event_file, json_file) in enumerate(zip(event_files, json_files)):
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)
            try:
                n_tr = len(json_data["time"]["samples"]["AcquisitionNumber"])
            except:
                logging.warning(f"AcquisitionNumber not found in {json_file}, using brain file shape instead")
                n_tr = nib.load(brain_files[i]).shape[0]
            
            df = pd.read_csv(event_file, sep="\t")
            df = df.dropna(subset=["stim_type"])
            df["onset_TR"] = df["onset"] / tr
            sorted_unique_stim_types = sorted(df["stim_type"].unique())
            binary_df = get_binary_df(df, n_tr, sorted_unique_stim_types)
            design_matrices.append(binary_df.to_numpy())
        except (IOError, json.JSONDecodeError, KeyError, pd.errors.EmptyDataError) as e:
            logging.error(f"Error processing files {event_file} and {json_file}: {str(e)}")
    return design_matrices

def main(sub_id: str, task: str, tr: float, n_jobs: int, event_files: List[str], json_files: List[str], brain_files: List[str], output_dir: str):
    # Window parameters
    window_size = 2
    step_size = 1
    
    # Process data in sliding windows
    for window_start in range(0, len(event_files), step_size):
        window_end = window_start + window_size
        if window_end > len(event_files):
            break
            
        print(f"Processing window {window_start//step_size + 1}: runs {window_start+1}-{window_end}")
        
        # Get current window of files
        current_event_files = event_files[window_start:window_end]
        current_json_files = json_files[window_start:window_end]
        current_brain_files = brain_files[window_start:window_end]
        
        session_ids = [os.path.basename(f).split("_")[1] for f in current_event_files]
        session_mapping = {session_id: i+1 for i, session_id in enumerate(sorted(set(session_ids)))}
        session_array = np.array([session_mapping[sid] for sid in session_ids])
        print(f"Session array: {session_array}")

        design_matrices = load_event_and_json_files(current_event_files, current_json_files, current_brain_files, tr)
        
        brain_data = []
        for brain_file in current_brain_files:
            try:
                img = nib.load(brain_file)
                brain_data.append(img.get_fdata().T)
            except Exception as e:
                logging.error(f"Error loading brain file {brain_file}: {str(e)}")

        assert len(design_matrices) == len(brain_data) == len(session_array), f"Number of design matrices, brain data, and session array do not match: {len(design_matrices)}, {len(brain_data)}, {len(session_array)}"

        
        # Update file names to include window information
        window_suffix = f"runs_{window_start+1}to{window_end}"
        design_matrices_file = os.path.join(output_dir, f"{sub_id}_{task}_design_matrices_{window_suffix}.npy")
        np.save(design_matrices_file, design_matrices)
        print(f"Saved design matrices to {design_matrices_file}")

        stimdur = 2
        start_time = time.time()

        results_dir = os.path.join(output_dir, f"results/{sub_id}/window_{window_start//step_size + 1}")
        os.makedirs(results_dir, exist_ok=True)
        figure_dir = os.path.join(results_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)

        opt = {
            'wantlibrary': 1,
            'wantglmdenoise': 1,
            'wantfracridge': 1,
            'wantfileoutputs': [0,0,0,0],
            'wantmemoryoutputs': [1,1,1,1],
            "sessionindicator": session_array,
            'n_jobs': n_jobs
        }
        glmsingle_obj = GLM_single(opt)
        print(f'running GLMsingle...')
        try:
            results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, outputdir=results_dir, figuredir=figure_dir)
        except OSError as e:
            logging.error(f"Encountered error during GLMsingle fitting: {str(e)}")
            raise

        
        results_file = os.path.join(results_dir, f"results_{window_suffix}.npz")
        np.savez(results_file, **results)
        print(f"Saved results for {window_suffix} to {results_file}")
        
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    dataset_dir = os.getenv("DATASET_DIR")

    if not all([base_dir, scratch_dir, dataset_dir]):
        raise ValueError("Missing required environment variables")

    data_dir = Path(dataset_dir) / "fmriprep" / "hcptrt"

    parser = ArgumentParser()
    parser.add_argument("sub_id", type=str)  # This remains required as a positional argument
    parser.add_argument("--task", type=str, default="wm")  # Optional with default
    parser.add_argument("--tr", type=float, default=1.49)  # Optional with default
    parser.add_argument("--n_jobs", type=int, default=32)  # Optional with default
    args = parser.parse_args()

    event_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_events.tsv"))
    json_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_bold.json"))
    brain_files = sorted(data_dir.glob(f"{args.sub_id}/*/func/*{args.task}*tseries.nii"))

    output_dir = os.path.join(scratch_dir, f"hcptrt/output/GLMsingle_v4")
    os.makedirs(output_dir, exist_ok=True)

    if not all([event_files, json_files, brain_files]):
        raise FileNotFoundError("Required files not found")

    logging.info(f"Found {len(event_files)} event files, {len(json_files)} JSON files, and {len(brain_files)} brain files")

    main(args.sub_id, args.task, args.tr, args.n_jobs, event_files, json_files, brain_files, output_dir)
