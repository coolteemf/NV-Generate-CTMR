import glob
import os.path
import nibabel as nib
import numpy as np
import json

# Standard output filenames of TotalSegmentator and label values
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

FILES_TO_PROCESS = list(segmentation_names.values())

def make_nib_vol(data, ref_vol):
    return nib.Nifti1Image(data, ref_vol.affine, ref_vol.header)


def find_matching_label(filename_stem, label_dict):
    """
    Tries to find the integer label in label_dict that corresponds to the
    filename_stem (e.g. maps 'kidney_right' -> 'right kidney' -> 5).
    """
    # 1. Exact match (unlikely given naming conventions)
    if filename_stem in label_dict:
        return label_dict[filename_stem]

    # 2. Underscore to space substitution (e.g. 'small_bowel' -> 'small bowel')
    name_spaced = filename_stem.replace("_", " ")
    if name_spaced in label_dict:
        return label_dict[name_spaced]

    # 3. Sorted word matching (e.g. 'kidney_right' -> 'kidney right' == 'right kidney')
    stem_words = sorted(name_spaced.split())
    for label_name, label_id in label_dict.items():
        label_words = sorted(label_name.split())
        if stem_words == label_words:
            return label_id

    # 4. Specific manual overrides for known mismatches

    # Urinary System
    if filename_stem == "urinary_bladder" and "bladder" in label_dict:
        return label_dict["bladder"]

    # Vascular System (Latin/English inconsistencies)
    # Maps 'iliac_vein_left' -> 'left iliac vena'
    if "iliac_vein" in filename_stem:
        # Swap vein->vena and try again
        latin_name = filename_stem.replace("iliac_vein", "iliac_vena").replace("_", " ")
        latin_name_sorted = sorted(latin_name.split())
        for label_name, label_id in label_dict.items():
            if sorted(label_name.split()) == latin_name_sorted:
                return label_id

    # Musculoskeletal (TotalSegmentator v1 vs v2)
    if filename_stem == "erector_spinae_left" and "left autochthon" in label_dict:
        return label_dict["left autochthon"]
    if filename_stem == "erector_spinae_right" and "right autochthon" in label_dict:
        return label_dict["right autochthon"]

    # Gastrointestinal
    if filename_stem == "small_intestine" and "small bowel" in label_dict:
        return label_dict["small bowel"]

    return None


def create_multilabel_volume(pth, file_patterns, label_map):
    """
    Merges individual organ segmentation files into a single multilabel volume
    using the IDs defined in label_map.

    Args:
        pth (str): Path to the directory containing the segmentation files.
        file_patterns (list): List of filename stems to look for.
        label_map (dict): Dictionary mapping {target_organ_name: target_id}.

    Returns:
        nib.Nifti1Image: The merged multilabel volume.
    """
    merged_data = None
    ref_vol = None

    print(f"Scanning {pth} for {len(file_patterns)} organ patterns...")

    for organ_name in file_patterns:
        # Resolve the new ID from the label dictionary
        target_label_id = find_matching_label(organ_name, label_map)

        if target_label_id is None:
            print(
                f"Warning: Could not find a matching label in label_dict for file '{organ_name}'. Skipping."
            )
            continue

        # Construct the search pattern (e.g., "spleen*.nii.gz")
        organ_pattern = f"{organ_name}*.nii.gz"
        file_paths = glob.glob(os.path.join(pth, organ_pattern))

        if not file_paths:
            continue

        # Load data for the current organ
        organ_mask = None

        for fp in file_paths:
            current_vol = nib.load(fp)
            current_data = current_vol.get_fdata()

            if organ_mask is None:
                organ_mask = current_data
                if ref_vol is None:
                    ref_vol = current_vol
            else:
                organ_mask = np.maximum(organ_mask, current_data)

        if merged_data is None:
            if ref_vol is None:
                raise ValueError("Reference volume could not be established.")
            # Use uint8 (or uint16 if IDs > 255)
            merged_data = np.zeros(ref_vol.shape, dtype=np.uint8)

        # Assign the target label ID to the voxels
        count = np.count_nonzero(organ_mask)
        if count > 0:
            merged_data[organ_mask > 0] = target_label_id
            print(f"Merged {organ_name} -> ID {target_label_id} ({count} voxels)")

    if merged_data is None:
        raise ValueError("No segmentation files found in the provided directory.")

    return make_nib_vol(merged_data, ref_vol)


def main():
    # Configuration
    data_dir = "/home/francois/Projects/data/Totalsegmentator_dataset_v201/s0011"
    pth = os.path.join(data_dir, "segmentations")

    # Path to the JSON containing the target mapping
    label_dict_path = "configs/label_dict.json"

    output_filename = "multilabel_segmentation.nii.gz"

    print(f"Processing directory: {pth}")

    if not os.path.exists(label_dict_path):
        print(f"Error: {label_dict_path} not found.")
        return

    try:
        with open(label_dict_path, "r") as f:
            label_map = json.load(f)

        merged_vol = create_multilabel_volume(pth, FILES_TO_PROCESS, label_map)
        output_path = os.path.join(pth, output_filename)
        nib.save(merged_vol, output_path)
        print(f"Successfully saved multilabel segmentation to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
