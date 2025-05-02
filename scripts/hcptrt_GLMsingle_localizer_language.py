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

def create_language_binary_matrix(df: pd.DataFrame, n_tr: int, tr: float, predictor_columns: List[str]) -> np.ndarray:
    """
    Reconstructs trials for the language task and creates a binary design matrix.

    Args:
        df (pd.DataFrame): Input DataFrame with raw event data for one run.
        n_tr (int): Number of time points (TRs) for the run.
        tr (float): Repetition time.
        predictor_columns (List[str]): The predefined list of predictor names for the columns.

    Returns:
        np.ndarray: Binary design matrix (n_tr x n_predictors).
    """
    reconstructed_events = []

    # Process story blocks
    story_trials = df[df['trial_type'] == 'presentation_story'].groupby('ntrial')
    for ntrial, group in story_trials:
        presentation_story_row = group.iloc[0] # Should only be one per ntrial
        story_onset = presentation_story_row['onset']
        for i in range(8):
            reconstructed_events.append({
                'predictor': f'story_{i+1}',
                'onset': story_onset + i * 3.0,
                'duration': 3.0
            })

    # Process math trials
    for _, row in df.iterrows():
        if row['trial_type'] == 'presentation_math':
            reconstructed_events.append({
                'predictor': 'presentation_math_1',
                'onset': row['onset'],
                'duration': 3.0
            })
            reconstructed_events.append({
                'predictor': 'presentation_math_2',
                'onset': row['onset'] + 3.0, # Start second part after 3s
                'duration': 3.0
            })
        elif row['trial_type'] == 'question_math':
            reconstructed_events.append({
                'predictor': 'question_math',
                'onset': row['onset'],
                'duration': 3.0
            })
        elif row['trial_type'] == 'response_math':
            reconstructed_events.append({
                'predictor': 'response_math',
                'onset': row['onset'],
                'duration': 3.0
            })

    # Create binary matrix
    binary_matrix = pd.DataFrame(0, index=np.arange(n_tr), columns=predictor_columns)
    reconstructed_df = pd.DataFrame(reconstructed_events)

    for _, event in reconstructed_df.iterrows():
        predictor = event['predictor']
        if predictor in binary_matrix.columns:
            onset = event['onset']
            duration = event['duration']
            start_tr = int(np.floor(onset / tr))
            end_tr = int(np.floor((onset + duration) / tr)) # End TR is the start of the *next* TR after the event ends

            # Ensure indices are within bounds
            start_tr = max(0, start_tr)
            end_tr = min(n_tr -1 , end_tr) # Corrected end_tr bound

            if start_tr <= end_tr: # Ensure start is not after end
                 # Set TRs from start_tr up to and including end_tr to 1
                 # However, GLMsingle might expect events aligned to TR start. Let's stick to floor for now.
                 # Setting 1 for TRs that *overlap* with the event duration.
                 # A TR overlaps if event_start < tr_end and event_end > tr_start
                 # Simplified: mark TRs from floor(onset/tr) to floor((onset+duration)/tr)
                 # Let's refine the indexing based on common practice or GLMsingle examples if needed.
                 # For now, mark the range derived from floor division.
                 binary_matrix.loc[start_tr:end_tr, predictor] = 1
        else:
            logging.warning(f"Predictor '{predictor}' from reconstructed events not in defined columns. Skipping.")


    return binary_matrix.to_numpy()

def load_event_and_json_files(event_files: List[str], json_files: List[str], brain_files: List[str], tr: float, language_predictors: List[str]) -> List[np.ndarray]:
    """
    Load event and JSON files and create reconstructed language design matrices.

    Args:
        event_files (List[str]): List of event file paths.
        json_files (List[str]): List of JSON file paths.
        brain_files (List[str]): List of brain file paths.
        tr (float): Repetition time.
        language_predictors (List[str]): Predefined list of language task predictors.

    Returns:
        List[np.ndarray]: List of design matrices for the language task.
    """
    design_matrices = []
    for i, (event_file, json_file) in enumerate(zip(event_files, json_files)):
        try:
            # Determine n_tr
            try:
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                n_tr = len(json_data["time"]["samples"]["AcquisitionNumber"])
            except Exception: # Catch broad exceptions for JSON loading/parsing
                logging.warning(f"Could not read n_tr from {json_file}, trying brain file {brain_files[i]}")
                try:
                    # Assuming 4D NIfTI (X, Y, Z, T)
                    n_tr = nib.load(brain_files[i]).shape[3]
                except Exception as e_nib:
                    logging.error(f"Could not determine n_tr for run {i+1} from JSON or NIfTI: {e_nib}")
                    continue # Skip this run

            # Load raw event data
            df = pd.read_csv(event_file, sep="\t")
            # Basic cleaning/validation
            if 'onset' not in df.columns or 'duration' not in df.columns or 'trial_type' not in df.columns:
                 logging.error(f"Missing required columns in {event_file}. Skipping run {i+1}.")
                 continue
            df = df.dropna(subset=["onset", "duration", "trial_type"]) # Need these for reconstruction
            if df.empty:
                logging.warning(f"No valid trials found in {event_file} after dropping NaNs. Skipping run {i+1}.")
                continue

            # Create the binary matrix using the specific language reconstruction logic
            binary_matrix = create_language_binary_matrix(df, n_tr, tr, language_predictors)
            design_matrices.append(binary_matrix)

        except (IOError, pd.errors.EmptyDataError, FileNotFoundError) as e:
            logging.error(f"Error processing file pair {event_file}, {json_file}: {str(e)}. Skipping run {i+1}.")
            continue
        except Exception as e_gen:
             logging.error(f"An unexpected error occurred processing run {i+1} ({event_file}): {str(e_gen)}. Skipping run {i+1}.")
             continue

    return design_matrices

def main(sub_id: str, task: str, tr: float, n_jobs: int, event_files: List[str], json_files: List[str], brain_files: List[str], output_dir: str):
    session_ids = [os.path.basename(f).split("_")[1] for f in event_files]
    session_mapping = {session_id: i+1 for i, session_id in enumerate(sorted(set(session_ids)))}
    session_array = np.array([session_mapping[sid] for sid in session_ids])
    print(f"Session array: {session_array}")

    # Define the fixed predictors for the language task
    language_predictors = [
        f"story_{i+1}" for i in range(8)
    ] + [
        "presentation_math_1", "presentation_math_2",
        "question_math", "response_math"
    ]
    logging.info(f"Using fixed predictors for language task: {language_predictors}")

    # Load design matrices using the specific language processing function
    design_matrices = load_event_and_json_files(event_files, json_files, brain_files, tr, language_predictors)

    # --- Brain Data Loading (Assuming similar logic as before, check if reshape needed) ---
    brain_data = []
    successfully_loaded_indices = [] # Keep track of which runs loaded successfully
    original_indices = list(range(len(brain_files)))

    # Adjust design matrix loading to track successful loads
    # We need to modify load_event_and_json_files return value or track skips
    # Let's assume for now load_event_and_json_files returns a list potentially shorter than input files if skips happened
    # And we need a way to align brain_data and session_array
    # A simpler approach might be to have load_event_and_json_files return None for skipped runs

    # --- Revised Data Loading Loop --- #
    design_matrices = []
    brain_data = []
    valid_session_indices = []

    for i, (event_file, json_file, brain_file) in enumerate(zip(event_files, json_files, brain_files)):
        logging.info(f"Processing run {i+1}/{len(event_files)}...")
        design_matrix = None
        brain_run_data = None
        n_tr = -1 # Initialize n_tr

        # 1. Determine n_tr (essential for both matrix and brain data shape check)
        try:
            with open(json_file, "r") as f: json_data = json.load(f)
            n_tr = len(json_data["time"]["samples"]["AcquisitionNumber"])
        except Exception:
            logging.warning(f"N_tr from JSON failed for {json_file}, trying NIfTI...")
            try:
                img_header = nib.load(brain_file).header
                n_tr = img_header.get_data_shape()[3]
                logging.info(f"N_tr from NIfTI header: {n_tr}")
            except Exception as e_nib:
                logging.error(f"Cannot determine n_tr for run {i+1} ({brain_file}): {e_nib}. Skipping run.")
                continue # Skip this entire run
        if n_tr <= 0:
             logging.error(f"Invalid n_tr ({n_tr}) obtained for run {i+1}. Skipping run.")
             continue

        # 2. Load/Create Design Matrix
        try:
            df = pd.read_csv(event_file, sep="\t")
            if 'onset' not in df.columns or 'duration' not in df.columns or 'trial_type' not in df.columns:
                 raise ValueError(f"Missing required columns in {event_file}")
            df = df.dropna(subset=["onset", "duration", "trial_type"])
            if not df.empty:
                design_matrix = create_language_binary_matrix(df, n_tr, tr, language_predictors)
            else:
                 logging.warning(f"No valid trials in {event_file}, creating empty design matrix.")
                 # Create an all-zero matrix if no trials but run is valid
                 design_matrix = np.zeros((n_tr, len(language_predictors)))

        except Exception as e_design:
            logging.error(f"Error creating design matrix for run {i+1} ({event_file}): {e_design}. Skipping run.")
            continue # Skip this run

        # 3. Load Brain Data
        try:
            img = nib.load(brain_file)
            brain_data.append(img.get_fdata().T)
        except Exception as e:
            logging.error(f"Error loading brain file {brain_file}: {str(e)}")

        # If all parts succeeded for this run, add them to lists
        design_matrices.append(design_matrix)
        brain_data.append(brain_run_data)
        valid_session_indices.append(i) # Keep track of the original index

    # --- End Revised Data Loading Loop --- #

    if not design_matrices or not brain_data:
         logging.error("No valid runs could be processed. Exiting.")
         return

    # Filter session_array based on successfully loaded runs
    final_session_array = session_array[valid_session_indices]
    if len(design_matrices) != len(brain_data) or len(design_matrices) != len(final_session_array):
        # This should ideally not happen if logic above is correct, but check anyway
        logging.error(f"FATAL: Mismatch after filtering runs: {len(design_matrices)} designs, {len(brain_data)} brains, {len(final_session_array)} sessions")
        return
    print(f"Successfully processed {len(design_matrices)} runs.")
    print(f"Final Session array: {final_session_array}")


    # save unique predictors to a file
    predictor_file_path = os.path.join(output_dir, f"{sub_id}_{task}_unique_predictors.txt")
    with open(predictor_file_path, "w") as f:
        for predictor in language_predictors:
            f.write(f"{predictor}\n")
    logging.info(f"Saved fixed language predictors to {predictor_file_path}")

    stimdur = 3 # This is fixed for the reconstructed events
    start_time = time.time()

    results_dir = os.path.join(output_dir, f"results/{sub_id}/{task}")
    os.makedirs(results_dir, exist_ok=True)
    figure_dir = os.path.join(results_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    opt = {
        'wantlibrary': 1,
        'wantglmdenoise': 1,
        'wantfracridge': 1,
        'wantfileoutputs': [0,0,0,0],
        'wantmemoryoutputs': [1,1,1,1],
        "sessionindicator": final_session_array, # Use the filtered session array
        'n_jobs': n_jobs
    }
    glmsingle_obj = GLM_single(opt)
    print(f'running GLMsingle...')
    try:
        # Pass the filtered lists to fit
        results = glmsingle_obj.fit(design_matrices, brain_data, stimdur, tr, outputdir=results_dir, figuredir=figure_dir)
    except OSError as e:
        logging.error(f"Encountered error during GLMsingle fitting: {str(e)}")
        raise
    except Exception as e_fit: # Catch other potential errors during fit
        logging.error(f"An unexpected error occurred during GLMsingle fitting: {e_fit}")
        # Optionally log shapes for debugging
        logging.error(f"Input shapes: Design matrices ({len(design_matrices)} runs, shape of first: {design_matrices[0].shape if design_matrices else 'N/A'}), "
                      f"Brain data ({len(brain_data)} runs, shape of first: {brain_data[0].shape if brain_data else 'N/A'}), "
                      f"Session array ({final_session_array.shape if final_session_array is not None else 'N/A'}), stimdur={stimdur}, tr={tr}")
        raise


    # save results to a file
    results_file = os.path.join(results_dir, f"results.npz")
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
    parser.add_argument("sub_id", type=str)  # This remains required as a positional argument
    parser.add_argument("--task", type=str, default="language")  # Optional with default
    parser.add_argument("--tr", type=float, default=1.49)  # Optional with default
    parser.add_argument("--n_jobs", type=int, default=32)  # Optional with default
    args = parser.parse_args()

    event_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_events.tsv"))
    json_files = sorted(data_dir.glob(f"sourcedata/hcptrt/{args.sub_id}/*/func/*{args.task}*_bold.json"))
    brain_files = sorted(data_dir.glob(f"{args.sub_id}/*/func/*{args.task}*tseries.nii"))

    output_dir = os.path.join(scratch_dir, f"hcptrt/output/GLMsingle")
    os.makedirs(output_dir, exist_ok=True)

    if not all([event_files, json_files, brain_files]):
        raise FileNotFoundError("Required files not found")

    logging.info(f"Found {len(event_files)} event files, {len(json_files)} JSON files, and {len(brain_files)} brain files")

    main(args.sub_id, args.task, args.tr, args.n_jobs, event_files, json_files, brain_files, output_dir)
