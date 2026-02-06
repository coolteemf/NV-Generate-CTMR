import glob
import os.path
import nibabel as nib
import numpy as np
import json

from scripts.segmentation_utils import segmentation_names
from scripts.utils import make_nib_vol

FILES_TO_PROCESS = list(segmentation_names.values())

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
