import os
import numpy as np
import nibabel as nib

def make_nib_vol(data, ref_vol):
    return nib.Nifti1Image(data, ref_vol.affine, ref_vol.header)

# Configuration
data_dir = "/home/francois/Projects/data/Totalsegmentator_dataset_v201/s0011"
pth = os.path.join(data_dir, "segmentations")
template_labelmap_filename = "multilabel_segmentation.nii.gz"
heart_labelmap_filename = "heart_segmentation_translated.nii.gz"
output_filename = "combined_labelmap.nii.gz"

# Organ values
VAL_HEART = 115
VAL_AORTA = 6
VAL_IVC = 7
VAL_SVC = 125

# Load images
template_path = os.path.join(pth, template_labelmap_filename)
heart_path = heart_labelmap_filename

template_img = nib.load(template_path)
heart_img = nib.load(heart_path)

# Get data (casting to int for label manipulation)
template_data = template_img.get_fdata().astype(np.int32)
heart_data = heart_img.get_fdata().astype(np.int32)

# 1. Remove Heart from the template
template_data[template_data == VAL_HEART] = 0

# 2. Add Heart, Aorta, Vena Cava from the specific labelmap
# We create a mask specifically for the desired labels
target_labels = [VAL_HEART, VAL_AORTA, VAL_IVC, VAL_SVC]
roi_mask = np.isin(heart_data, target_labels)

# Overwrite template data using the specific mask
template_data[roi_mask] = heart_data[roi_mask]

# Save result
final_img = make_nib_vol(template_data, template_img)
nib.save(final_img, output_filename)