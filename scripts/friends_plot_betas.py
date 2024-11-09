import os
import glob
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from scipy.stats import zscore

# Plot dimensions
FIGURE_SIZE = (15, 10)
DISTRIBUTION_FIGURE_SIZE = (12, 6)
CUMULATIVE_FIGURE_SIZE = (12, 6)

# Plot styling
HIST_BINS = 50
PLOT_ALPHA = 0.6
SIGNAL_COLOR = 'blue'
NOISE_COLOR = 'red'

# Data configuration
MAIN_CHARACTERS = ["chandler", "joey", "monica", "phoebe", "rachel", "ross"]
SAVE_PLOTS = True

def get_results(ses_id: str, result_dir: str) -> tuple[dict, dict]:
    result_file = os.path.join(result_dir, ses_id, "results.npz")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Results file not found: {result_file}")
    
    glm_results = np.load(result_file, allow_pickle=True)   
    designinfo = np.load(os.path.join(result_dir, ses_id, "DESIGNINFO.npy"), allow_pickle=True).item()
    typed = glm_results['typed'].item()

    return designinfo, typed

def get_betas(char: str, reversed_cn_dict: dict, mean_typed_stim_arrays: dict) -> np.ndarray:
    char_id = reversed_cn_dict[char]
    char_betas = mean_typed_stim_arrays[char_id]
    return zscore(char_betas)

def get_betas_for_main_char(ses_id: str, main_char: list[str], result_dir: str, cn_dict: dict) -> tuple[dict, np.ndarray]:     
    designinfo, typed = get_results(ses_id, result_dir)
    stim_type_order = designinfo['stimorder']
    unique_stims = np.unique(stim_type_order)

    typed_stim_arrays = get_stim_results(typed, unique_stims, stim_type_order)
    mean_typed_stim_arrays = {stim: np.mean(typed_stim_arrays[stim], axis=1) 
                            for stim in unique_stims}

    reversed_cn_dict = {v: i for i, v in enumerate(cn_dict)}
    noisepool = np.squeeze(typed["noisepool"]).astype(bool)
    
    # Instead of creating DataFrames here, return dictionaries of betas
    betas = {}
    for char in main_char:
        betas_char = get_betas(char, reversed_cn_dict, mean_typed_stim_arrays)
        betas[char] = betas_char
    return betas, noisepool

def get_stim_results(
    results: dict, 
    unique_stims: np.ndarray, 
    stim_type_order: list
) -> dict:
    data = np.squeeze(results["betasmd"])  # Shape: (91282, 1260)

    # Get unique stimulus types and create a dictionary to store results    
    stim_arrays = {}

    # For each unique stimulus type
    for stim in unique_stims:
        # Get indices where this stimulus type occurs
        stim_indices = np.where(np.array(stim_type_order) == stim)[0]
        # Select those columns from the data and store in dictionary
        stim_arrays[stim] = data[:, stim_indices]

    # Print shapes to verify
    for stim in unique_stims:
        print(f"Stimulus {stim}: {stim_arrays[stim].shape}")
    return stim_arrays

def plot_noise_pool(noisepool: np.ndarray, ses_id: str, sub_id: str, figure_dir: str) -> None:
    counts = np.bincount(noisepool.astype(int))

    plt.figure(figsize=FIGURE_SIZE)
    plt.bar(['Signal (0)', 'Noise (1)'], counts)
    plt.title(f'Distribution of Signal vs Noise Voxels {ses_id}')
    plt.ylabel('Count')
    plt.text(0, counts[0]/2, f'{counts[0]} voxels\n({counts[0]/len(noisepool):.1%})', 
             ha='center', va='center')
    plt.text(1, counts[1]/2, f'{counts[1]} voxels\n({counts[1]/len(noisepool):.1%})', 
             ha='center', va='center')
    plt.savefig(os.path.join(figure_dir, f"{sub_id}_{ses_id}_noise_pool.png"))
    plt.close()

def plot_r2(r2_data: np.ndarray, noisepool: np.ndarray, ses_id: str, sub_id: str, figure_dir: str) -> None:

    # Create separate arrays for signal and noise voxels
    signal_r2 = r2_data[~noisepool]  # Signal voxels (where noisepool is False)
    noise_r2 = r2_data[noisepool]   # Noise voxels (where noisepool is True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(signal_r2, bins=50, alpha=0.8, label='Signal', color='blue')
    plt.hist(noise_r2, bins=50, alpha=0.5, label='Noise', color='red')

    # Add labels and title
    plt.xlabel('R²')
    plt.ylabel('Count')
    plt.title(f'Distribution of R² Values for Signal and Noise Voxels {ses_id}')
    plt.legend()

    # Optional: Add summary statistics as text
    plt.text(0.02, 0.95, f'Signal mean: {np.mean(signal_r2):.3f}\nNoise mean: {np.mean(noise_r2):.3f}', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig(os.path.join(figure_dir, f"{sub_id}_{ses_id}_r2.png"))
    plt.close()

def plot_betas(
    mean_typed_stim_arrays: dict, 
    main_char: list, 
    reversed_cn_dict: dict, 
    noisepool: np.ndarray, 
    ses_id: str, 
    sub_id: str, 
    figure_dir: str
) -> None:
    if not main_char:
        raise ValueError("main_char list cannot be empty")
    if not os.path.isdir(figure_dir):
        raise ValueError(f"Invalid figure directory: {figure_dir}")

    # Add debug logging
    logging.info(f"Available stimulus IDs: {sorted(mean_typed_stim_arrays.keys())}")
    logging.info(f"Character mapping: {reversed_cn_dict}")

    # Create figure with subplots in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Beta Distributions by Character (Signal vs Noise) {ses_id}', fontsize=14)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot each character's distribution
    for idx, char in enumerate(main_char):
        
        ax = axes_flat[idx]
        char_id = reversed_cn_dict[char]
        
        # Add error handling for missing characters
        if char_id not in mean_typed_stim_arrays:
            logging.warning(f"No data found for character '{char}' (ID: {char_id}) in session {ses_id}")
            ax.text(0.5, 0.5, f'No data\nfor {char}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Get betas and split into signal/noise
        char_betas = mean_typed_stim_arrays[char_id]
        # z-score
        char_betas = zscore(char_betas)
        signal_betas = char_betas[~noisepool]
        noise_betas = char_betas[noisepool]
        
        # Plot histograms
        ax.hist(signal_betas, bins=HIST_BINS, alpha=PLOT_ALPHA, label='Signal', color=SIGNAL_COLOR)
        ax.hist(noise_betas, bins=HIST_BINS, alpha=PLOT_ALPHA, label='Noise', color=NOISE_COLOR)
        
        # Add labels and stats
        ax.set_title(char.capitalize())
        ax.set_xlabel('Beta Values')
        ax.set_ylabel('Count')
        
        # Add mean values as text
        stats_text = f'Signal μ: {np.mean(signal_betas):.3f}\nNoise μ: {np.mean(noise_betas):.3f}'
        ax.text(0.03, 0.97, stats_text, 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"{sub_id}_{ses_id}_betas.png"))
    plt.close()

def main(ses_id: str, sub_id: str, cn_dict: dict, result_dir: str, figure_dir: str) -> None:
    """Process GLM results and create visualization plots.
    
    Args:
        ses_id: Session identifier
        sub_id: Subject identifier
        result_dir: Directory containing GLM results
        figure_dir: Directory for saving figures
    """
    try:
        # Load and process data
        designinfo, typed = get_results(ses_id, result_dir)
        stim_type_order = designinfo['stimorder']
        unique_stims = np.unique(stim_type_order)
        
        typed_stim_arrays = get_stim_results(typed, unique_stims, stim_type_order)
        mean_typed_stim_arrays = {stim: np.mean(typed_stim_arrays[stim], axis=1) 
                                for stim in unique_stims}
        
        reversed_cn_dict = {v: i for i, v in enumerate(cn_dict)}
        print(f"Reversed character mapping: {reversed_cn_dict}")
        noisepool = np.squeeze(typed["noisepool"]).astype(bool)
        
        # Generate plots
        plot_noise_pool(noisepool, ses_id, sub_id, figure_dir)
        plot_betas(mean_typed_stim_arrays, MAIN_CHARACTERS, reversed_cn_dict, 
                  noisepool, ses_id, sub_id, figure_dir)
        plot_r2(np.squeeze(typed["R2"]), noisepool, ses_id, sub_id, figure_dir)
        
    except Exception as e:
        logging.error(f"Error processing session {ses_id}: {str(e)}")
        raise

def setup_directories(sub_id: str):
    """Setup required directories and return paths."""
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        raise ValueError("SCRATCH_DIR environment variable not set")
    
    output_dir = os.path.join(scratch_dir, "friends", "output")
    result_dir = os.path.join(output_dir, "GLMsingle", "results", sub_id)
    return output_dir, result_dir

def get_session_ids(result_dir: str) -> list[str]:
    """Get list of session IDs from result directory.
    
    Args:
        result_dir: Path to the results directory
        
    Returns:
        list[str]: List of session IDs
        
    Raises:
        FileNotFoundError: If result_dir doesn't exist
        ValueError: If no valid session directories are found
    """
    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"Results directory not found: {result_dir}")
        
    session_ids = [d for d in os.listdir(result_dir) if d.startswith('ses-')]
    
    if not session_ids:
        raise ValueError(f"No session directories (starting with 'ses-') found in {result_dir}")
        
    return sorted(session_ids)  # Sort for consistent ordering

def setup_logging() -> None:
    """Configure logging to write to both file and console."""
    # Load environment variables
    load_dotenv()
    base_dir = os.getenv('BASE_DIR')
    
    # Create logs directory in BASE_DIR
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"friends_plot_betas_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to: {log_file}")

def plot_session_distribution(session_distribution: dict, figure_dir: str, sub_id: str) -> None:
    """Plot the distribution of signal voxels across sessions.
    
    Args:
        session_distribution: Dictionary containing voxel counts and percentages per session
        figure_dir: Directory to save the plot
        sub_id: Subject identifier
    """
    plt.figure(figsize=DISTRIBUTION_FIGURE_SIZE)
    plt.bar(session_distribution.keys(), 
            [d['n_voxels'] for d in session_distribution.values()])
    plt.xlabel('Number of Sessions')
    plt.ylabel('Number of Voxels')
    plt.title('Distribution of Signal Voxels Across Sessions')

    # Add value labels on top of each bar
    for i, v in enumerate(session_distribution.values()):
        plt.text(i + 1, v['n_voxels'], 
                f"{v['n_voxels']}\n({v['percentage']:.1f}%)", 
                ha='center', va='bottom')

    if SAVE_PLOTS:
        plt.savefig(os.path.join(figure_dir, f"{sub_id}_session_distribution.png"))
        plt.close()
    else:
        plt.show()

def plot_cumulative_distribution(x_values: range, y_values: list, percentages: list, 
                               figure_dir: str, sub_id: str) -> None:
    """Plot the cumulative distribution of signal voxels.
    
    Args:
        x_values: Range of session numbers
        y_values: List of voxel counts
        percentages: List of percentage values
        figure_dir: Directory to save the plot
        sub_id: Subject identifier
    """
    fig, ax1 = plt.subplots(figsize=CUMULATIVE_FIGURE_SIZE)
    ax2 = ax1.twinx()

    # Plot voxel counts
    line1 = ax1.plot(x_values, y_values, 'b-', label='Voxel Count')
    ax1.set_xlabel('Number of Sessions')
    ax1.set_ylabel('Number of Voxels', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot percentages
    line2 = ax2.plot(x_values, percentages, 'r-', label='Percentage')
    ax2.set_ylabel('Percentage of Total Voxels (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Cumulative Distribution of Signal Voxels Across Sessions')
    plt.grid(True, alpha=0.3)

    if SAVE_PLOTS:
        plt.savefig(os.path.join(figure_dir, f"{sub_id}_cumulative_distribution.png"))
        plt.close()
    else:
        plt.show()

def analyze_signal_distribution(
    signal_masks: np.ndarray, 
    figure_dir: str, 
    sub_id: str
) -> tuple[dict, tuple[range, list, list]]:
    """Analyze and plot the distribution of signal voxels across sessions.
    
    Args:
        signal_masks: Boolean array of shape (n_sessions, n_voxels) indicating signal voxels
        figure_dir: Directory to save the generated plots
        sub_id: Subject identifier
        
    Returns:
        tuple containing:
            - dict: Session distribution
            - tuple: (x_values, y_values, percentages) for cumulative distribution
    """
    # Count occurrences across sessions
    voxel_counts = np.sum(signal_masks, axis=0)
    n_sessions = signal_masks.shape[0]
    
    # Calculate session distribution
    session_distribution = {}
    for k in range(1, n_sessions + 1):
        n_voxels = np.sum(voxel_counts == k)
        percentage = (n_voxels / len(voxel_counts)) * 100
        session_distribution[k] = {
            'n_voxels': n_voxels,
            'percentage': percentage
        }
        print(f"Voxels appearing in exactly {k} sessions: {n_voxels} ({percentage:.2f}%)")

    # Calculate cumulative distribution
    x_values = range(1, n_sessions + 1)
    y_values = []
    percentages = []
    
    for k in x_values:
        n_voxels = np.sum(voxel_counts >= k)
        percentage = (n_voxels / len(voxel_counts)) * 100
        y_values.append(n_voxels)
        percentages.append(percentage)
        print(f"Voxels appearing in {k} or more sessions: {n_voxels} ({percentage:.2f}%)")

    # Create plots
    plot_session_distribution(session_distribution, figure_dir, sub_id)
    plot_cumulative_distribution(x_values, y_values, percentages, figure_dir, sub_id)
    
    return session_distribution, (x_values, y_values, percentages)

def create_analysis_dataframes(
    session_ids: list[str], 
    main_characters: list[str],
    result_dir: str,
    cn_dict: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create DataFrames for beta values and noisepool analysis."""
    betas_df = pd.DataFrame(index=main_characters, columns=session_ids)
    noisepool_df = pd.DataFrame(
        index=session_ids,
        columns=['mask', 'signal_count', 'noise_count', 'signal_percentage']
    )

    for ses_id in session_ids:
        betas, noisepool = get_betas_for_main_char(
            ses_id, main_characters, result_dir, cn_dict
        )
        
        for char in main_characters:
            betas_df.loc[char, ses_id] = betas[char]
        
        noisepool_df.loc[ses_id] = {
            'mask': noisepool,
            'signal_count': (~noisepool).sum(),
            'noise_count': noisepool.sum(),
            'signal_percentage': (~noisepool).mean() * 100
        }
    
    return betas_df, noisepool_df

def analyze_signal_changes(signal_masks: np.ndarray, session_ids: list[str], 
                         figure_dir: str, sub_id: str) -> pd.DataFrame:
    """Analyze how signal voxels change between consecutive sessions.
    
    Args:
        signal_masks: Boolean array of shape (n_sessions, n_voxels)
        session_ids: List of session identifiers
        figure_dir: Directory to save plots
        sub_id: Subject identifier
        
    Returns:
        pd.DataFrame containing signal change statistics
    """
    changes_df = pd.DataFrame(index=session_ids[1:],  # Start from second session
                            columns=['new_signals', 'lost_signals', 'maintained_signals', 
                                   'total_signals', 'net_change'])
    
    for i in range(1, len(session_ids)):
        prev_mask = signal_masks[i-1]
        curr_mask = signal_masks[i]
        
        # Calculate changes
        new_signals = np.sum(~prev_mask & curr_mask)  # Voxels that became signal
        lost_signals = np.sum(prev_mask & ~curr_mask)  # Voxels that were lost
        maintained_signals = np.sum(prev_mask & curr_mask)  # Voxels that stayed signal
        total_signals = np.sum(curr_mask)  # Total signal voxels in current session
        net_change = new_signals - lost_signals
        
        changes_df.loc[session_ids[i]] = {
            'new_signals': new_signals,
            'lost_signals': lost_signals,
            'maintained_signals': maintained_signals,
            'total_signals': total_signals,
            'net_change': net_change
        }
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot bars for new and lost signals
    x = np.arange(len(session_ids[1:]))
    width = 0.35
    
    plt.bar(x, changes_df['new_signals'], width, label='New Signals', color='green', alpha=0.6)
    plt.bar(x, -changes_df['lost_signals'], width, label='Lost Signals', color='red', alpha=0.6)
    
    # Add net change line
    plt.plot(x, changes_df['net_change'], 'k--', label='Net Change')
    
    plt.xlabel('Sessions')
    plt.ylabel('Number of Voxels')
    plt.title('Signal Voxel Changes Between Consecutive Sessions')
    plt.xticks(x, session_ids[1:], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i in x:
        # New signals (positive values)
        plt.text(i, changes_df['new_signals'].iloc[i], 
                f'+{changes_df["new_signals"].iloc[i]}', 
                ha='center', va='bottom')
        # Lost signals (negative values)
        plt.text(i, -changes_df['lost_signals'].iloc[i], 
                f'-{changes_df["lost_signals"].iloc[i]}', 
                ha='center', va='top')
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(os.path.join(figure_dir, f"{sub_id}_signal_changes.png"))
        plt.close()
    else:
        plt.show()
        
    return changes_df

if __name__ == "__main__":
    try:
        sub_id = "sub-01"
        setup_logging()  # Call before setup_directories
        output_dir, result_dir = setup_directories(sub_id)
        
        # Load design matrix once for all sessions
        designmatrix = np.load(os.path.join(os.path.dirname(os.path.dirname(result_dir)), "design_matrix_face.npz"))
        cn_dict = designmatrix['cn_dict']
        
        # Setup figure directory
        figure_dir = os.path.join(output_dir, "GLMsingle", "figures", sub_id)
        os.makedirs(figure_dir, exist_ok=True)
        
        # Process each session
        session_ids = get_session_ids(result_dir)
        for ses_id in session_ids:
            print(f"Processing session: {ses_id}")
            main(ses_id=ses_id, 
                 sub_id=sub_id,
                 cn_dict=cn_dict,
                 result_dir=result_dir, 
                 figure_dir=figure_dir)
            
        betas_df, noisepool_df = create_analysis_dataframes(
            session_ids, MAIN_CHARACTERS, result_dir, cn_dict
        )
        # After processing sessions and creating DataFrames
        signal_masks = np.array([~noisepool_df.loc[ses_id, 'mask'] for ses_id in session_ids])
        
        # Analyze and plot signal distribution
        session_dist, cumulative_dist = analyze_signal_distribution(
            signal_masks, figure_dir, sub_id
        )
        
        plt.figure(figsize=(10, 6))
        plt.bar(noisepool_df.index, noisepool_df['signal_percentage'])
        plt.xlabel('Session ID')
        plt.ylabel('Signal Percentage (%)')
        plt.title('Signal Percentage by Session')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plt.savefig(os.path.join(figure_dir, f"{sub_id}_signal_percentage_by_session.png"))
            plt.close()
        else:
            plt.show()
        
        # After creating signal_masks
        changes_df = analyze_signal_changes(signal_masks, session_ids, figure_dir, sub_id)
        
        # Print summary statistics
        print("\nSignal Change Summary:")
        print(changes_df)
        
        # Calculate and print some additional statistics
        print("\nAverage Changes:")
        print(f"Average new signals per session: {changes_df['new_signals'].mean():.2f}")
        print(f"Average lost signals per session: {changes_df['lost_signals'].mean():.2f}")
        print(f"Average maintained signals per session: {changes_df['maintained_signals'].mean():.2f}")
        print(f"Average net change per session: {changes_df['net_change'].mean():.2f}")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise