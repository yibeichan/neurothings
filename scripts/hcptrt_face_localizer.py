import os
import glob
import pandas as pd
import numpy as np

import hcp_utils as hcp

from nilearn.glm.contrasts import Contrast
import nibabel as nib
import nilearn
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm

from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

from dotenv import load_dotenv
from argparse import ArgumentParser

def get_subject_files(sub_id, run_id, task_id, data_dir, nese_dir):
    image_files = glob.glob(os.path.join(data_dir, f"{sub_id}/{sub_id}*task-{task_id}_run-{int(run_id)}*.dtseries.nii"))
    if len(image_files) == 0:
        raise ValueError(f"No files found for subject {sub_id} run {run_id} task {task_id}")
    else:
        event_files =[]
        for file in image_files:
            ses_id = os.path.basename(file).split("_space")[0].split("-")[2].replace("_task", "")
            event_files.append(os.path.join(nese_dir, "cneuromod", "hcptrt", f"{sub_id}", f"ses-{ses_id}", "func", f"{sub_id}_ses-{ses_id}_task-{task_id}_run-{int(run_id):02}_events.tsv"))
    return image_files, event_files

def get_design_matrix(event_file, n_scans, tr, hrf_model, drift_model, drift_order):
    event_df = pd.read_csv(event_file, sep="\t")
    if 'trial_type' in event_df.columns:
        event_df1 = event_df.dropna(subset=['duration']).drop(columns=['trial_type']).rename(columns={'stim_type': 'trial_type'})
    # Define the sampling times for the design matrix
    frame_times = np.arange(n_scans) * tr
    # Build design matrix with the reviously defined parameters
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=event_df1,
        hrf_model=hrf_model,
        drift_model=drift_model,
        drift_order=drift_order,
    )
    return design_matrix

def make_localizer_contrasts(design_matrix):
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }

    contrasts['face-body'] = contrasts['Face'] - contrasts['Body']
    contrasts['face-tools'] = contrasts['Face'] - contrasts['Tools']
    contrasts['face-place'] = contrasts['Face'] - contrasts['Place']
    contrasts['face-all'] = (contrasts['Face'] - (1/3)*contrasts['Body'] - (1/3)*contrasts['Tools'] - (1/3)*contrasts['Place'])

    return contrasts

def get_run_contrast(image_files, event_files, sub_id, task_id, tr, hrf_model, drift_model, drift_order):
    contrast_results_per_run = []
    for i, (img, event_file) in enumerate(zip(image_files, event_files)):
        data = nib.load(img).dataobj[:]
        n_scans = data.shape[0]
        design_matrix = get_design_matrix(event_file, n_scans, tr, hrf_model, drift_model, drift_order)
        labels, estimates = run_glm(data, design_matrix.values)
        contrasts = make_localizer_contrasts(design_matrix)
        contrast_results = dict.fromkeys(contrasts.keys())
        # Iterate on contrasts
        for contrast_id, contrast_val in contrasts.items():
            print(f"\tcontrast id: {contrast_id}")
            # compute the contrasts
            contrast = compute_contrast(labels, estimates, contrast_val,
                                        stat_type='t')
            contrast_results[contrast_id] = contrast

            print(f"Contrast {contrast_id} for {sub_id} task-{task_id} for session-{i+1} saved")
        contrast_results_per_run.append(contrast_results)
        print(f"Contrasts for {sub_id} task-{task_id} saved")
    return contrast_results_per_run

def compute_fixed_effects(contrasts, variances, precision_weighted=True):
    """Compute the fixed effects t/F-statistic, contrast, variance, given arrays of effects and variance."""
    
    tiny = 1.0e-16
    contrasts = np.squeeze(np.asarray(contrasts))
    variances = np.squeeze(np.asarray(variances))
    
    # Ensure variances are not zero or negative
    variances = np.maximum(variances, tiny)

    # Precision-weighted averaging
    if precision_weighted:
        weights = 1.0 / variances
        fixed_fx_variance = 1.0 / np.sum(weights, axis=0)
        fixed_fx_contrasts = np.sum(contrasts * weights, axis=0) * fixed_fx_variance
    else:
        # Simple averaging (unweighted)
        fixed_fx_variance = np.mean(variances, axis=0) / len(variances)
        fixed_fx_contrasts = np.mean(contrasts, axis=0)

    # Determine if contrast is t or F based on dimensionality
    dim = 1
    stat_type = "t"
    fixed_fx_contrasts_ = fixed_fx_contrasts
    if len(fixed_fx_contrasts.shape) == 2:
        dim = fixed_fx_contrasts.shape[0]
        if dim > 1:
            stat_type = "F"
    else:
        fixed_fx_contrasts_ = fixed_fx_contrasts[np.newaxis]

    # Handle degrees of freedom if provided (optional)
    dof = np.sum([100]*len(contrasts))

    # Use nilearn's Contrast class to compute stats
    con = Contrast(
        effect=fixed_fx_contrasts_,
        variance=fixed_fx_variance,
        dim=dim,
        dof=dof,
        stat_type=stat_type,
    )
    fixed_fx_z_score = con.z_score()
    fixed_fx_stat = con.stat_

    return (
        fixed_fx_contrasts,
        fixed_fx_variance,
        fixed_fx_stat,
        fixed_fx_z_score,
    )

def get_fixed_effects(contrast_name, contrast_results):
    effect = [contrasts[contrast_name].effect_size() for contrasts in contrast_results]
    variance = [contrasts[contrast_name].effect_variance() for contrasts in contrast_results]
    
    fx_contrasts, fx_variance, fx_stat, fx_z_score = compute_fixed_effects(effect, variance)

    return fx_contrasts, fx_variance, fx_stat, fx_z_score

def get_fixed_contrasts(contrasts_of_interest, contrast_results_per_run):
    fx_effects = dict.fromkeys(contrasts_of_interest)
    fx_variance = dict.fromkeys(contrasts_of_interest)
    for c in contrasts_of_interest:
        fx_effects[c], fx_variance[c], _, _ = get_fixed_effects(c, contrast_results_per_run)
    return fx_effects, fx_variance

def save_cifti(contrast_id, activation_map, file_type, sub_id, template_file, save_dir):
    # Load the template CIFTI file to get the appropriate brain model axis
    template_cifti = nib.load(template_file)
    _, brain_model_axis = [template_cifti.header.get_axis(i) for i in range(template_cifti.ndim)]
    # Ensure the activation map is at least 2D
    data = np.atleast_2d(activation_map)
    # Create a ScalarAxis to represent the single contrast/statistic map
    scalar_axis = nib.cifti2.ScalarAxis([file_type]) 
    new_header = nib.Cifti2Header.from_axes([scalar_axis, brain_model_axis])
    new_cifti_img = nib.Cifti2Image(data, new_header)
    filename = os.path.join(save_dir, f'{sub_id}_contrast-{contrast_id.lower()}_{file_type}.dscalar.nii')
    nib.save(new_cifti_img, filename)
    print(f"Saved: {filename}")

def main(sub_id, task_id, data_dir, nese_dir, save_dir):

    tr = 1.49
    drift_model = "polynomial"
    hrf_model = "glover"
    drift_order = 6

    contrasts_of_interest = ['Face', 'face-all', 'face-body', 'face-tools', 'face-place']

    fx_effects_runs = []
    fx_variance_runs = []
    for run_id in ["1", "2"]:
        image_files, event_files = get_subject_files(sub_id=sub_id, run_id=run_id, task_id=task_id, data_dir=data_dir, nese_dir=nese_dir)
        contrast_results_per_run = get_run_contrast(image_files, event_files, sub_id, task_id, tr, hrf_model, drift_model, drift_order)
        fx_effects_run, fx_variance_run = get_fixed_contrasts(contrasts_of_interest, contrast_results_per_run)

        fx_effects_runs.append(fx_effects_run)
        fx_variance_runs.append(fx_variance_run)

    fx_effects = {contrast: None for contrast in contrasts_of_interest}
    fx_variance = {contrast: None for contrast in contrasts_of_interest}
    fx_stat = {contrast: None for contrast in contrasts_of_interest}
    fx_z_score = {contrast: None for contrast in contrasts_of_interest}
    fx_p_value = {contrast: None for contrast in contrasts_of_interest} 
    fx_fdr_corrected = {contrast: None for contrast in contrasts_of_interest}  

    for c in contrasts_of_interest:
        # Compute fixed effects, variance, stat, and z-scores
        fx_effects[c], fx_variance[c], fx_stat[c], fx_z_score[c] = compute_fixed_effects([effect[c] for effect in fx_effects_runs], [variance[c] for variance in fx_variance_runs])

        # Convert z-scores to one-tailed p-values
        fx_p_value[c] = norm.sf(fx_z_score[c])

        # Apply FDR correction to p-values
        p_values_flat = fx_p_value[c].flatten()
        _, fdr_corrected_p_values = fdrcorrection(p_values_flat, alpha=0.05)
        fx_fdr_corrected[c] = fdr_corrected_p_values.reshape(fx_p_value[c].shape)

        for metric, data in zip(["effect_size", "effect_variance", "stat", "z_score", "p_value", "fdr_corrected_p_value"],  
                                [fx_effects[c], fx_variance[c], fx_stat[c], fx_z_score[c], fx_p_value[c], fx_fdr_corrected[c]]): 
            save_cifti(c, data, metric, sub_id, image_files[0], save_dir)

        print(f"Contrast {c} for {sub_id} task-{task_id} saved")

if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser(description="Run fmriprep data")
    parser.add_argument("sub_id", help="Subject ID (e.g., sub-001)", type=str)
    parser.add_argument("--task_id", default="wm", help="Task name (e.g., hcptrt)", type=str)
    args = parser.parse_args()

    sub_id = args.sub_id
    task_id = args.task_id

    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    nese_dir = os.getenv("NESE_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    data_dir = os.path.join(output_dir, "hcptrt_postproc")
    save_dir = os.path.join(output_dir, "hcptrt_face_localizer")
    os.makedirs(save_dir, exist_ok=True)

    main(sub_id, task_id, data_dir, nese_dir, save_dir)    