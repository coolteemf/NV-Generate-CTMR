import torch
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SaveImaged,
)

# Configuration from your target MAISI config (e.g., config_infer_24g_512x512x512.json)
# If using a different config, update these values to match 'spacing' in that JSON.
TARGET_SPACING = (0.75, 0.75, 1.0) 

files = [{"image": "/home/francois/Projects/data/raw_data/10 CT Coro   DS_CorCTA  0.6  Bv40  0 - 90 %  Matrix 256 - 10 frames Volume Sequence by CardiacCycle 0.nii.gz"}]

preprocess = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    
    # 1. Orientation: Ensure RAS
    Orientationd(keys=["image"], axcodes="RAS"),
    
    # 2. Spacing: Resample
    Spacingd(keys=["image"], pixdim=TARGET_SPACING, mode="bilinear"),
    
    # 3. Intensity: Clip
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-1000, 
        a_max=1000, 
        b_min=-1000, 
        b_max=1000, 
        clip=True
    ),
    
    # Corrected: Use SaveImaged for dictionary pipelines
    SaveImaged(
        keys=["image"],       # Essential: tells MONAI which key contains the image to save
        output_dir=".", 
        output_postfix="preprocessed", 
        output_ext=".nii.gz", 
        separate_folder=False,
        resample=False
    )
])

print("Preprocessing...")
preprocess(files)
print("Done. Use the 'preprocessed' file for outpainting.")