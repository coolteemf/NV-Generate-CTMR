import glob
import os.path
import nibabel as nib
import numpy as np

segmentation_names = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}


def make_nib_vol(data, ref_vol):
    return nib.Nifti1Image(data, ref_vol.affine, ref_vol.header)


def create_multilabel_volume(pth, segmentation_map):
    """
    Merges individual organ segmentation files into a single multilabel volume.
    
    Args:
        pth (str): Path to the directory containing the segmentation files.
        segmentation_map (dict): Dictionary mapping {label_id: "organ_name"}.
        
    Returns:
        nib.Nifti1Image: The merged multilabel volume.
    """
    merged_data = None
    ref_vol = None
    
    # Iterate over the dictionary to find files and assign labels
    for label_id, organ_name in segmentation_map.items():
        # Construct the search pattern (e.g., "spleen*.nii.gz")
        organ_pattern = f"{organ_name}*.nii.gz"
        file_paths = glob.glob(os.path.join(pth, organ_pattern))
        
        if not file_paths:
            # Skip if no file is found for this organ
            continue
        
        # Load data for the current organ
        # We handle cases where there might be multiple files for one organ (merge them first)
        organ_mask = None
        
        for fp in file_paths:
            current_vol = nib.load(fp)
            current_data = current_vol.get_fdata()
            
            if organ_mask is None:
                organ_mask = current_data
                # Use the first valid file found as the reference for shape/affine
                if ref_vol is None:
                    ref_vol = current_vol
            else:
                # Combine masks if multiple files match the pattern (logical OR)
                organ_mask = np.maximum(organ_mask, current_data)
        
        # Initialize the master volume array on the first successful find
        if merged_data is None:
            if ref_vol is None:
                raise ValueError("Reference volume could not be established.")
            # Use uint8 as labels are small integers (0-117)
            merged_data = np.zeros(ref_vol.shape, dtype=np.uint8)
        
        # Assign the label ID to the voxels where the organ is present
        merged_data[organ_mask > 0] = label_id

    if merged_data is None:
        raise ValueError("No segmentation files found in the provided directory.")

    return make_nib_vol(merged_data, ref_vol)


def main():
    # Configuration
    data_dir = "/home/francois/Projects/data/Totalsegmentator_dataset_v201/s0011"
    pth = os.path.join(data_dir, "segmentations")
    output_filename = "multilabel_segmentation.nii.gz"
    
    print(f"Processing directory: {pth}")
    
    try:
        merged_vol = create_multilabel_volume(pth, segmentation_names)
        output_path = os.path.join(pth, output_filename)
        nib.save(merged_vol, output_path)
        print(f"Successfully saved multilabel segmentation to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
