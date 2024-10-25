import os
from dotenv import load_dotenv
from scipy import stats
import numpy as np
from argparse import ArgumentParser
from statsmodels.stats.multitest import fdrcorrection
import logging
from datetime import datetime

# At the top of the file, after imports
def setup_logging(base_dir, sub_id, task):
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(base_dir, 'logs', 'GLMsingle_localizer')
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{sub_id}_{task}_{timestamp}.log')

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    logging.info(f"Logging to: {log_file}")

def get_stim_results(results, unique_stims, stim_type_order):
    """Extract beta values for each stimulus type."""
    stim_arrays = {}
    for stim in unique_stims:
        stim_indices = np.where(np.array(stim_type_order) == stim)[0]
        stim_arrays[stim] = results[:, stim_indices]
        logging.info(f"Extracted {len(stim_indices)} trials for stimulus type: {stim}")
    return stim_arrays

def main(sub_id, task, glm_results, designinfo, stim_files, output_dir):
    logging.info(f"Processing subject {sub_id}, task {task}")
    
    # Load and validate data
    typed = glm_results['typed'].item()
    data = np.squeeze(typed["betasmd"])  # Shape: (91282, 1260)
    logging.info(f"Loaded beta data with shape: {data.shape}")
    
    # Create stimulus mapping
    stim_type = [stim_file.split("/")[-1].split("_")[0] for stim_file in stim_files]
    stim_mapping = {i: stim_type[i] for i in range(len(stim_type))}
    stim_type_order = [stim_mapping[sid] for sid in designinfo['stimorder']]
    unique_stims = np.unique(stim_type_order)
    logging.info(f"Found stimulus types: {unique_stims}")

    # Extract stimulus-specific data
    stim_arrays = get_stim_results(data, unique_stims, stim_type_order)
    
    # Create signal mask
    noisepool = np.squeeze(typed["noisepool"])
    signal_mask = ~noisepool.astype(bool)
    n_signal_vertices = np.sum(signal_mask)
    logging.info(f"Signal vertices: {n_signal_vertices} out of {data.shape[0]}")

    # Extract signal data for each stimulus
    signal_arrays = {}
    for stim, beta_array in stim_arrays.items():
        signal_arrays[stim] = beta_array[signal_mask, :]
        logging.info(f"Signal array shape for {stim}: {signal_arrays[stim].shape}")
    
    # Prepare data for statistical comparison
    group2 = signal_arrays[unique_stims[1]]
    group3 = signal_arrays[unique_stims[2]]
    logging.info(f"Comparing {unique_stims[1]} vs {unique_stims[2]}")

    # Statistical testing
    t_stats = np.zeros(n_signal_vertices)
    p_values = np.zeros(n_signal_vertices)

    for signal in range(n_signal_vertices):
        t_stat, p_val = stats.ttest_rel(group2[signal, :], group3[signal, :])
        t_stats[signal] = t_stat
        p_values[signal] = p_val
    
    # Multiple comparison correction
    significant_mask = fdrcorrection(p_values, alpha=0.05)[0]
    n_significant = np.sum(significant_mask)
    logging.info(f"Found {n_significant} significant vertices after FDR correction")

    # Find vertices where group2 > group3
    positive_significant = (t_stats > 0) & significant_mask
    n_positive = np.sum(positive_significant)
    logging.info(f"Found {n_positive} vertices with {unique_stims[1]} > {unique_stims[2]}")

    # Map back to full brain space
    full_mask = np.zeros(data.shape[0], dtype=bool)
    signal_indices = np.where(signal_mask)[0]
    significant_vertices_full = signal_indices[positive_significant]
    full_mask[significant_vertices_full] = True
    
    # Validate results
    assert np.sum(full_mask) == n_positive, "Mismatch in number of significant vertices"
    
    # Save results
    output_file = os.path.join(output_dir, f"{sub_id}_{task}_full_mask.npy")
    np.save(output_file, full_mask)
    logging.info(f"Saved mask to: {output_file}")

    stats_dict = {
        'total_vertices': data.shape[0],
        'signal_vertices': n_signal_vertices,
        'significant_vertices': n_significant,
        'positive_significant': n_positive,
        't_stats': t_stats,
        'p_values': p_values
    }
    stats_file = os.path.join(output_dir, f"{sub_id}_{task}_stats.npz")
    np.savez(stats_file, **stats_dict)
    logging.info(f"Saved statistics to: {stats_file}")

if __name__ == "__main__":
    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    
    parser = ArgumentParser(description="post-processing GLMsingle results")
    parser.add_argument("sub_id", help="Subject ID", type=str)
    parser.add_argument("--task", help="Task", type=str, default="wm")
    args = parser.parse_args()
    
    # Set up logging before any other operations
    setup_logging(base_dir, args.sub_id, args.task)
    
    # Continue with the rest of the script...
    result_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/results/{args.sub_id}/{args.task}"
    figure_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/results/{args.sub_id}/{args.task}/figures"
    glm_results = np.load(os.path.join(result_dir, "results.npz"), allow_pickle=True)   
    designinfo = np.load(os.path.join(result_dir, "DESIGNINFO.npy"), allow_pickle=True).item()
    stim_files = open(os.path.join(scratch_dir, f"hcptrt/output/GLMsingle/", f"{args.sub_id}_{args.task}_unique_stim_files.txt"), "r").readlines()
    output_dir = f"{scratch_dir}/hcptrt/output/GLMsingle/facemask"

    # Validate input paths
    assert os.path.exists(result_dir), f"Results directory not found: {result_dir}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    main(args.sub_id, args.task, glm_results, designinfo, stim_files, output_dir)
