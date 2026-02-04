import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def generate_aligned_pad_volume(
    heart_path, chest_path, output_path, translation_vec, fill_value=-1024
):
    """
    Resamples a heart CT volume into the geometry of a chest CT volume after
    applying a translation.

    Parameters:
    - translation_vec: list or array [x, y, z] in world coordinates (mm).
    - fill_value: value for padding outside the original FOV (default -1024 for CT air).
    """

    # 1. Load the volumes
    heart_img = nib.load(heart_path)
    chest_img = nib.load(chest_path)

    # 2. Apply the translation to the Heart's affine
    # This moves the heart in world space without changing data/resampling yet.
    # New World Pos = Old World Pos + Translation
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = translation_vec

    # Calculate the new affine: T * A_heart
    aligned_heart_affine = transform_matrix @ heart_img.affine

    # Create a proxy image with the new spatial position
    aligned_heart_proxy = nib.Nifti1Image(heart_img.get_fdata(), aligned_heart_affine)

    # 3. Resample into Chest geometry
    # resample_from_to maps the input (aligned_heart) into the voxel grid
    # defined by the reference (chest_img).
    resampled_img = resample_from_to(
        from_img=aligned_heart_proxy,
        to_vox_map=chest_img,  # Defines the target shape and affine
        order=1,  # 1=Linear interpolation (standard for CT)
        cval=fill_value,  # Background value for padding
    )

    # 4. Save the result
    nib.save(resampled_img, output_path)
    print(f"Saved aligned volume to: {output_path}")


# --- Example Usage ---
if __name__ == "__main__":
    my_translation = [0.0, 30.0, 210.0]

    generate_aligned_pad_volume(
        heart_path="/home/francois/Projects/data/raw_data/LnRobo/CT_Fluoro/240313/10 CT Coro   DS_CorCTA  0.6  Bv40  0 - 90 %  Matrix 256 - 10 frames Volume Sequence by CardiacCycle 0.nii.gz",
        chest_path="/home/francois/Projects/data/Totalsegmentator_dataset_v201/s0011/ct.nii.gz",
        output_path="./heart_aligned.nii.gz",
        translation_vec=my_translation,
    )
