#################################################################
# Extract time series from fMRIprep output directly for GLMsingle
#################################################################

import os
import glob
import nibabel as nib
import hcp_utils as hcp
import numpy as np
from dotenv import load_dotenv
from argparse import ArgumentParser
import logging
from datetime import datetime
from joblib import Parallel, delayed

def save_as_cifti(data, filename, hemisphere):
    brain_model_axis = nib.cifti2.BrainModelAxis.from_mask(np.ones(data.shape[1]), name=hemisphere)
    scalar_axis = nib.cifti2.ScalarAxis([f"TR {i+1}" for i in range(data.shape[0])])
    dscalar_img = nib.Cifti2Image(data, header=[scalar_axis, brain_model_axis])
    nib.save(dscalar_img, filename)

def process_file(task_file, label_data, save_dir):
    try:
        data = nib.load(task_file).get_fdata()
        lh_data = np.array([hcp.left_cortex_data(data[i]) for i in range(data.shape[0])])
        rh_data = np.array([hcp.right_cortex_data(data[i]) for i in range(data.shape[0])])
        task_basename = os.path.basename(task_file)

        for label, mask in label_data.items():
            output_filename = os.path.join(save_dir, f"{task_basename.replace('_bold.dtseries.nii', '')}_{label}_tseries.nii")
            if label.startswith("lh"):
                task_data = lh_data * mask
                save_as_cifti(task_data, output_filename, "CORTEX_LEFT")
            elif label.startswith("rh"):
                task_data = rh_data * mask
                save_as_cifti(task_data, output_filename, "CORTEX_RIGHT")
        logging.info(f"Processed {task_file}")
    except Exception as e:
        logging.error(f"Error processing {task_file}: {e}", exc_info=True)

def process_subject(subject_files, froi_labels, save_dir):
    label_data = {label: nib.load(file).get_fdata()[0] for label, file in froi_labels.items()}
    print(label_data)
    Parallel(n_jobs=12)(
        delayed(process_file)(task_file, label_data, save_dir) for task_file in subject_files
    )

if __name__ == "__main__":
    try:
        load_dotenv()

        parser = ArgumentParser(description="Extract time series.")
        parser.add_argument("sub_id", help="Subject ID", type=str)
        args = parser.parse_args()
        sub_id = args.sub_id

        base_dir = os.getenv("BASE_DIR")
        nese_dir = os.getenv("NESE_DIR")
        scratch_dir = os.getenv("SCRATCH_DIR")
        output_dir = os.path.join(scratch_dir, "output")
        save_dir = os.path.join(output_dir, "time_series_v1", f"{sub_id}")
        os.makedirs(save_dir, exist_ok=True)

        subject_files = glob.glob(os.path.join(nese_dir, f"friends.fmriprep/{sub_id}/*/func/*fsLR_den-91k_bold.dtseries.nii"))
        
        froi_labels = {
            f"{hemi}_{roi}": os.path.join("/nese/mit/group/sig/projects/yibei/friends/output", "froi_v1", f"{sub_id}_{hemi}_{roi}_mask.dscalar.nii")
            for hemi in ["lh", "rh"]
            for roi in ["ofa", "ffa", "sts", "v1", "tl"]
        }

        # Set up logging
        log_dir = os.path.join(base_dir, "logs", "extract_time_series_v1")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{sub_id}_{timestamp}.log")
        
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Processing started for {sub_id}")
        
        process_subject(subject_files, froi_labels, save_dir)
        
        logging.info(f"Processing completed for {sub_id}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}", exc_info=True)
