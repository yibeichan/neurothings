import os
import glob
import pandas as pd
import numpy as np
import json
import time
from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import List, Dict, Any

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
        stim = row.stim_type
        if pd.notna(stim):
            binary_df.at[onset, stim] = 1
    return binary_df

def load_event_and_json_files(event_files: List[str], json_files: List[str], tr: float, sorted_unique_stim_types: List[str]) -> List[np.ndarray]:
    """
    Load event and JSON files and create design matrices.

    Args:
        event_files (List[str]): List of event file paths.
        json_files (List[str]): List of JSON file paths.
        tr (float): Repetition time.
        sorted_unique_stim_types (List[str]): Sorted list of unique stimulus types.

    Returns:
        List[np.ndarray]: List of design matrices.
    """
    design_matrices = []
    for event_file, json_file in zip(event_files, json_files):
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)
            n_tr = len(json_data["time"]["samples"]["AcquisitionNumber"])
            
            df = pd.read_csv(event_file, sep="\t")
            df["onset_TR"] = df["onset"] / tr
            binary_df = get_binary_df(df, n_tr, sorted_unique_stim_types)
            design_matrices.append(binary_df.to_numpy())
        except (IOError, json.JSONDecodeError, KeyError, pd.errors.EmptyDataError) as e:
            logging.error(f"Error processing files {event_file} and {json_file}: {str(e)}")
    return design_matrices

def main(sub_id: str, task: str, tr: float, n_jobs: int, event_files: List[str], json_files: List[str], brain_files: List[str]):
    session_ids = [os.path.basename(f).split("_")[1] for f in event_files]
    session_mapping = {session_id: i+1 for i, session_id in enumerate(sorted(set(session_ids)))}
    session_array = np.array([session_mapping[sid] for sid in session_ids])

    all_unique_stim_types = set()
    for event_file in event_files:
        df = pd.read_csv(event_file, sep="\t")
        all_unique_stim_types.update(df["stim_type"].dropna().unique())

    # Sort the unique stim_types to ensure consistent order
    sorted_unique_stim_types = sorted(all_unique_stim_types)

    # Use the sorted_unique_stim_types in the loop
    design_matrices = load_event_and_json_files(event_files, json_files, tr, sorted_unique_stim_types)
    
    brain_data = []
    for brain_file in brain_files:
        try:
            img = nib.load(brain_file)
            brain_data.append(img.get_fdata().T)
        except Exception as e:
            logging.error(f"Error loading brain file {brain_file}: {str(e)}")

    output_dir = f"/orcd/scratch/bcs/001/yibei/hcptrt/output/GLMsingle/results/{sub_id}/{task}"
    os.makedirs(output_dir, exist_ok=True)
    figure_dir = f"/orcd/scratch/bcs/001/yibei/hcptrt/output/GLMsingle/figures/{sub_id}/{task}/figures"
    os.makedirs(figure_dir, exist_ok=True)

    stimdur = 2
    start_time = time.time()

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
        results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, outputdir=output_dir, figuredir=figure_dir)
    except OSError as e:
        logging.error(f"Encountered error during GLMsingle fitting: {str(e)}")
        raise

    # save results to a file
    results_file = os.path.join(output_dir, f"results.npz")
    np.savez(results_file, **results)
    print(f"Saved results for session to {results_file}")
    
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
    parser.add_argument("--sub_id", type=str, required=True)
    parser.add_argument("--task", type=str, default="wm")
    parser.add_argument("--tr", type=float, default=1.49)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()

    event_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_events.tsv"))
    json_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_bold.json"))
    brain_files = sorted(data_dir.glob(f"{args.sub_id}/*/func/*{args.task}*tseries.nii"))

    if not all([event_files, json_files, brain_files]):
        raise FileNotFoundError("Required files not found")

    logging.info(f"Found {len(event_files)} event files, {len(json_files)} JSON files, and {len(brain_files)} brain files")

    main(args.sub_id, args.task, args.tr, args.n_jobs, event_files, json_files, brain_files)
