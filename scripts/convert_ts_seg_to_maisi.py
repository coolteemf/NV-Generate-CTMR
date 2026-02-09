import json
import nibabel as nib
import numpy as np
from scripts.segmentation_utils import build_translation_table, segmentation_names


def convert_nifti(input_path, output_path, translation_map):
    print(f"Loading {input_path}...")
    nii = nib.load(input_path)
    data = nii.get_fdata().astype(np.uint16)

    if data.ndim == 4:
        data = data.squeeze()

    print("Translating labels...")

    # Handle potential out-of-bounds keys in input data
    max_val_in_data = np.max(data)
    if max_val_in_data >= len(translation_map):
        print(
            f"Warning: Input contains ID {max_val_in_data} which exceeds known TS labels."
        )
        print("Extending map with zeros to prevent crash.")
        padding = np.zeros(max_val_in_data - len(translation_map) + 1, dtype=np.uint16)
        translation_map = np.concatenate([translation_map, padding])

    # Fast numpy vector translation
    translated_data = translation_map[data]

    print(f"Saving to {output_path}...")
    new_nii = nib.Nifti1Image(translated_data.astype(np.uint8), nii.affine, nii.header)
    nib.save(new_nii, output_path)
    print("Done.")


if __name__ == "__main__":
    # Path to the JSON containing the target mapping
    label_dict_path = "configs/label_dict.json"

    input_filename = "/home/francois/Projects/NV-Generate-CTMR/heart_segmentation.nii"
    output_filename = "heart_segmentation_translated.nii.gz"

    try:
        with open(label_dict_path, "r") as f:
            labelmap_maisi = json.load(f)
            mapper = build_translation_table(segmentation_names, labelmap_maisi)

            # Process the NIfTI
            convert_nifti(input_filename, output_filename, mapper)
    except Exception as e:
        print(f"An error occurred: {e}")