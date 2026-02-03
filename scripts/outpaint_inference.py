# scripts/outpaint_inference.py
# Optimized for volumes size:     "spacing": [
#     0.75,
#     0.75,
#     4.0
# ] *     "output_size": [
#     512,
#     512,
#     128
# ], = 384, 384, 412mm
# Our volume size: 179.96, 179.96, 172 mm, resolution: 256x256x430, spacing: 0.703125, 0.703125, 0.400000mm
# => resample to spacing, then pad

from __future__ import annotations

import argparse
import logging
import os
import sys
import random
from datetime import datetime

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
import torch
import torch.distributed as dist
from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from tqdm import tqdm

# Import utilities from the existing repo structure
# Note: These require running as a module (python -m scripts.outpaint_inference)
from .diff_model_setting import (
    initialize_distributed,
    load_config,
    setup_logging,
    run_torchrun,
)
from .sample import ReconModel
from .utils import define_instance, dynamic_infer

# -------------------------------------------------------------------------
# Reuse Helper Functions
# -------------------------------------------------------------------------


def set_random_seed(seed: int) -> int:
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_models(args, device, logger):
    """Loads Autoencoder, UNet, and Scale Factor using repo utilities."""
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # Load Autoencoder weights
    checkpoint_autoencoder = torch.load(
        args.trained_autoencoder_path, map_location=device
    )
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    logger.info(f"Checkpoints {args.trained_autoencoder_path} loaded.")

    # Load UNet weights
    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(
        "/home/francois/Projects/NV-Generate-CTMR/models/diff_unet_3d_rflow-ct.pt",
        map_location=device,
        weights_only=False,
    )
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    unet.eval()
    logger.info(f"Checkpoints {args.trained_diffusion_path} loaded.")

    # Load Scale Factor
    scale_factor = checkpoint.get("scale_factor", 1.0)
    logger.info(f"Scale Factor -> {scale_factor}.")
    return autoencoder, unet, scale_factor


def prepare_tensors(args, device):
    """Prepares conditioning tensors (spacing, body regions) from config."""
    # Robustly get the inference config dict
    if hasattr(args, "diffusion_unet_inference"):
        infer_conf = args.diffusion_unet_inference
    else:
        infer_conf = vars(args)

    if "top_region_index" not in infer_conf and "body_region" in infer_conf:
        # Map strings to region indices: 0=Head/Neck, 1=Thorax, 2=Abdomen, 3=Pelvis
        region_map = {
            "head": 0,
            "neck": 0,
            "chest": 1,
            "thorax": 1,
            "abdomen": 2,
            "pelvis": 3,
        }

        # Find all mentioned regions
        mentioned_indices = []
        for br in infer_conf["body_region"]:
            br_lower = br.lower()
            for key, val in region_map.items():
                if key in br_lower:
                    mentioned_indices.append(val)

        # Default to Chest (1) if nothing recognized
        if not mentioned_indices:
            mentioned_indices = [1]

        min_idx = min(mentioned_indices)
        max_idx = max(mentioned_indices)

        # Create one-hot vectors (size 4)
        top_region_list = [0, 0, 0, 0]
        top_region_list[min_idx] = 1

        bot_region_list = [0, 0, 0, 0]
        bot_region_list[max_idx] = 1

        infer_conf["top_region_index"] = top_region_list
        infer_conf["bottom_region_index"] = bot_region_list

    top_region = np.array(infer_conf["top_region_index"]).astype(float) * 1e2
    bottom_region = np.array(infer_conf["bottom_region_index"]).astype(float) * 1e2
    spacing = np.array(infer_conf["spacing"]).astype(float) * 1e2

    top_tensor = torch.from_numpy(top_region[np.newaxis, :]).half().to(device)
    bot_tensor = torch.from_numpy(bottom_region[np.newaxis, :]).half().to(device)
    sp_tensor = torch.from_numpy(spacing[np.newaxis, :]).half().to(device)

    # Handle modality embedding if present
    modality_val = infer_conf.get("modality", 0)
    mod_tensor = modality_val * torch.ones((len(sp_tensor)), dtype=torch.long).to(
        device
    )

    return top_tensor, bot_tensor, sp_tensor, mod_tensor


def save_image(data, output_spacing, output_path, logger):
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = output_spacing[i]

    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)
    logger.info(f"Saved {output_path}.")


# -------------------------------------------------------------------------
# Core Outpainting Logic
# -------------------------------------------------------------------------


def run_outpainting(
    args,
    device,
    autoencoder,
    unet,
    scale_factor,
    conditioning_tensors,
    input_ct_crop,
    input_mask,
    output_size,
    logger,
):
    """
    Executes the MAISI v2 Rectified Flow Outpainting loop.
    """
    top_tensor, bot_tensor, spacing_tensor, modality_tensor = conditioning_tensors

    # Get inference config safely
    if hasattr(args, "diffusion_unet_inference"):
        infer_conf = args.diffusion_unet_inference
    else:
        infer_conf = vars(args)

    # 1. Prepare Inputs
    # Calculate latent dimensions (Downsampled by 4)
    latent_shape = (
        1,
        args.latent_channels,
        output_size[0] // 4,
        output_size[1] // 4,
        output_size[2] // 4,
    )
    D, H, W = latent_shape[2:]

    # 2. Encode the Known Crop
    # Normalize input [-1000, 1000] -> [0, 1] for VAE
    input_norm = (input_ct_crop + 1000.0) / 2000.0
    input_norm = torch.clamp(input_norm, 0.0, 1.0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            # Encode and scale
            z0 = autoencoder.encode_stage_2_inputs(input_norm) * scale_factor

    mask = torch.nn.functional.interpolate(
        input_mask,
        size=latent_shape[-3:], 
        mode='nearest',
    )

    # 4. Initialize Noise
    # Ideally, we use the SAME noise map for the initial state and the trajectory of the known region
    noise_canvas = torch.randn(latent_shape, device=device)
    latents = noise_canvas.clone()

    # 5. Setup Scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")
    num_steps = infer_conf.get("num_inference_steps", 30)

    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=num_steps,
            input_img_size_numel=torch.prod(torch.tensor(latent_shape[2:])),
        )
    else:
        raise ValueError("This outpainting script requires RFlowScheduler (MAISI v2).")

    # 6. Inference Loop
    all_timesteps = noise_scheduler.timesteps
    timestep_pairs = zip(all_timesteps[:-1], all_timesteps[1:])

    cfg_scale = infer_conf.get("cfg_guidance_scale", 1.5)
    include_body = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    logger.info(f"Starting Rectified Flow Inference ({len(all_timesteps)} steps)...")

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in tqdm(timestep_pairs, total=len(all_timesteps) - 1):
            t_tensor = torch.Tensor([t]).to(device)

            # A. Prepare UNet Inputs
            unet_inputs = {
                "x": latents,
                "timesteps": t_tensor,
                "spacing_tensor": spacing_tensor,
            }
            if include_body:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_tensor,
                        "bottom_region_index_tensor": bot_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update({"class_labels": modality_tensor})

            # B. Classifier-Free Guidance (Optimized for Memory)
            if cfg_scale > 0:
                # 1. Run Conditional Pass
                # unet_inputs already contains the conditional data setup in step A
                out_cond = unet(**unet_inputs)

                # 2. Run Unconditional Pass
                # Create a copy of inputs for the unconditional pass
                unet_inputs_uncond = {k: v.clone() for k, v in unet_inputs.items()}

                # Update class_labels to null (zeros) for unconditional
                if "class_labels" in unet_inputs_uncond:
                    unet_inputs_uncond["class_labels"] = torch.zeros_like(
                        modality_tensor
                    )

                out_uncond = unet(**unet_inputs_uncond)

                # 3. Combine
                v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)

                # Clean up to free memory immediately
                del out_cond, out_uncond, unet_inputs_uncond
            else:
                v_pred = unet(**unet_inputs)

            # C. Step (Euler step via Scheduler)
            # RFlow: latents_next = latents + (next_t - t) * v_pred
            latents_next, _ = noise_scheduler.step(v_pred, t, latents, next_t)

            # D. Replacement (The Outpainting Magic)
            # Calculate where the "known" heart should be at time `next_t`
            # Trajectory: z_t = t * noise + (1-t) * data
            z_known_next = next_t * noise_canvas + (1.0 - next_t) * z0

            # Combine: Keep generated (mask=1) parts, Force known (mask=0) parts
            latents = latents_next * mask + z_known_next * (1.0 - mask)

    # 7. Decode
    logger.info("Decoding final volume...")
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    # Use config parameters for sliding window inference
    sw_roi_size = infer_conf.get("autoencoder_sliding_window_infer_size")
    sw_overlap = infer_conf.get("autoencoder_sliding_window_infer_overlap")

    print(f"sw_roi_size {sw_roi_size}")
    print(f"sw_overlap {sw_overlap}")
    inferer = SlidingWindowInferer(
        roi_size=sw_roi_size,
        sw_batch_size=1,
        progress=True,
        mode="gaussian",
        overlap=sw_overlap,
        sw_device=device,
        device=device,
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            synthetic_images = dynamic_infer(inferer, recon_model, latents)

    # Post-process to HU
    data = synthetic_images.squeeze().cpu().detach().numpy()
    a_min, a_max = -1000, 1000
    data = data * (a_max - a_min) + a_min
    data = np.clip(data, a_min, a_max)

    return np.int16(data)


# -------------------------------------------------------------------------
# Main Interface
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MAISI v2 Outpainting Inference")
    parser.add_argument("-e", "--env_config", type=str, required=True)
    parser.add_argument("-c", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--model_def", type=str, required=True)
    parser.add_argument(
        "-i",
        "--input_crop",
        type=str,
        required=True,
        help="Path to NIfTI file of the crop (e.g. heart).",
    )
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--output_prefix", type=str, default="outpaint_result")

    # Helper argument to prevent crash when run_torchrun injects --out_index
    parser.add_argument(
        "--out_index", type=str, default=None, help="Internal use for torchrun"
    )

    args = parser.parse_args()

    # Automatically switch to torchrun if using multiple GPUs and not yet distributed
    if args.num_gpus > 1 and "RANK" not in os.environ:
        run_torchrun("scripts.outpaint_inference", sys.argv[1:], args.num_gpus)
        return

    # Load Configurations
    config = load_config(args.env_config, args.model_config, args.model_def)

    # Check if 'diffusion_unet_inference' exists. If not, assume the config is flat
    # and map 'diffusion_unet_inference' to the config object's internal dict.
    if not hasattr(config, "diffusion_unet_inference"):
        # Use dict() to create a shallow copy of the vars dict to avoid
        # circular reference recursion during MONAI ConfigParser resolution.
        config.diffusion_unet_inference = dict(vars(config))

    # Merge argparse args into config args if needed
    config.output_dir = args.output_dir
    config.output_prefix = args.output_prefix

    # Initialize Distributed (or Single GPU)
    local_rank, world_size, device = initialize_distributed(args.num_gpus)
    logger = setup_logging("outpaint_inference")

    seed = set_random_seed(
        config.diffusion_unet_inference.get("random_seed", 0) + local_rank
    )
    logger.info(
        f"Initialized Rank {local_rank}/{world_size} on {device} with seed {seed}"
    )

    if "autoencoder_tp_num_splits" in config.diffusion_unet_inference:
        if hasattr(config, "autoencoder_def"):
            config.autoencoder_def["num_splits"] = config.diffusion_unet_inference[
                "autoencoder_tp_num_splits"
            ]
        if hasattr(config, "mask_generation_autoencoder_def"):
            config.mask_generation_autoencoder_def["num_splits"] = (
                config.diffusion_unet_inference["autoencoder_tp_num_splits"]
            )
    print(f"num tp splits autoencoder_def: {config.autoencoder_def['num_splits']}")
    print(
        f"num tp splits mask_generation_autoencoder_def: {config.mask_generation_autoencoder_def['num_splits']}"
    )
    # Output Specs
    output_size = (
        tuple(config.diffusion_unet_inference["dim"])
        if "dim" in config.diffusion_unet_inference
        else tuple(config.diffusion_unet_inference["output_size"])
    )

    output_spacing = tuple(config.diffusion_unet_inference["spacing"])

    # Load Models
    autoencoder, unet, scale_factor = load_models(config, device, logger)

    # Prepare Tensors
    cond_tensors = prepare_tensors(config, device)

    # Load Input Crop
    logger.info(f"Loading input crop from {args.input_crop}...")
    nii_img = nib.load(args.input_crop)
    min_image = nii_img.get_fdata().min()
    # Adapt to expected spacing
    nii_img = nibproc.resample_to_output(
        nii_img, voxel_sizes=output_spacing, order=3, mode="constant", cval=min_image
    )
    # Adapt to expected shape (padding)
    pad_amount = np.array(output_size) - np.array(nii_img.shape)
    inferior_padding = pad_amount // 2  # //2 or /2 to keep original aligned ?
    superior_padding = pad_amount - inferior_padding
    assert np.all(pad_amount > 0)
    origin_delta = nii_img.affine[:3, :3] @ inferior_padding
    new_origin = nii_img.affine[:3, 3] - origin_delta
    new_affine = np.eye(4)
    new_affine[:3, :3] = nii_img.affine[:3, :3]
    new_affine[:3, 3] = new_origin
    padded_img_data = np.zeros(output_size) + min_image
    padded_img_data[
        inferior_padding[0] : inferior_padding[0] + nii_img.shape[0],
        inferior_padding[1] : inferior_padding[1] + nii_img.shape[1],
        inferior_padding[2] : inferior_padding[2] + nii_img.shape[2],
    ] = nii_img.get_fdata()
    padded_img = nib.nifti1.Nifti1Image(padded_img_data, new_affine)
    input_ct_crop = torch.from_numpy(padded_img.get_fdata()).float()
    # Generate mask with ones where outpainting happens
    input_mask = np.ones(output_size)
    input_mask[
        inferior_padding[0] : inferior_padding[0] + nii_img.shape[0],
        inferior_padding[1] : inferior_padding[1] + nii_img.shape[1],
        inferior_padding[2] : inferior_padding[2] + nii_img.shape[2],
    ] = 0
    input_mask = torch.from_numpy(input_mask).float()

    # Ensure shape (1, 1, D, H, W)
    input_mask = input_mask.unsqueeze(0).unsqueeze(0)
    input_ct_crop = input_ct_crop.unsqueeze(0).unsqueeze(0)
    input_ct_crop = input_ct_crop.to(device)
    input_mask = input_mask.to(device)

    # Run Outpainting
    result_data = run_outpainting(
        config,
        device,
        autoencoder,
        unet,
        scale_factor,
        cond_tensors,
        input_ct_crop,
        input_mask,
        output_size,
        logger,
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_name = f"{args.output_prefix}_{timestamp}.nii.gz"
    out_path = os.path.join(args.output_dir, out_name)

    save_image(result_data, output_spacing, out_path, logger)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
