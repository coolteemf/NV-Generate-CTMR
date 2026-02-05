# scripts/outpaint_inference.py
# Optimized for volumes size:     "spacing": [
#     0.75,
#     0.75,
#     4.0
# ] * "output_size": [
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
from datetime import datetime

import monai
import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
import torch
import torch.distributed as dist
from monai.inferers import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from tqdm import tqdm

from scripts.diff_model_infer import save_image, set_random_seed

# Note: These require running as a module (python -m scripts.outpaint_inference)
from .diff_model_setting import (
    initialize_distributed,
    load_config,
    setup_logging,
    run_torchrun,
)
from .sample import ReconModel
from .utils import define_instance, dynamic_infer, binarize_labels, plot_volume_grid
from .find_masks import find_masks


def get_volume_bb(img, min_val):
    mask = img > min_val

    def get_bbox_limits(mask, axis):
        # Collapse other axes
        reduce_axes = tuple(i for i in range(mask.ndim) if i != axis)
        valid_indices = np.where(np.any(mask, axis=reduce_axes))[0]
        if valid_indices.size > 0:
            return valid_indices[0], valid_indices[-1]
        return 0, 0

    if np.any(mask):
        x_start, x_end = get_bbox_limits(mask, 0)
        y_start, y_end = get_bbox_limits(mask, 1)
        z_start, z_end = get_bbox_limits(mask, 2)

    return x_start, x_end, y_start, y_end, z_start, z_end


def prepare_tensors(args, device, logger):
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

    logger.info(f"Top Region: {top_region}, Bottom Region: {bottom_region}")

    return top_tensor, bot_tensor, sp_tensor, mod_tensor


def load_models(args, device, logger):
    """Loads Autoencoder, UNet, Scale Factor, and optionally ControlNet."""
    autoencoder = define_instance(args, "autoencoder_def").to(device)
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
        args.trained_diffusion_path,
        map_location=device,
        weights_only=False,
    )
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    unet.eval()
    logger.info(f"Checkpoints {args.trained_diffusion_path} loaded.")

    controlnet = None
    if hasattr(args, "controlnet_def") and hasattr(args, "trained_controlnet_path"):
        try:
            controlnet = define_instance(args, "controlnet_def").to(device)
            checkpoint_controlnet = torch.load(
                args.trained_controlnet_path, map_location=device, weights_only=False
            )
            monai.networks.utils.copy_model_state(controlnet, unet.state_dict())
            controlnet.load_state_dict(
                checkpoint_controlnet["controlnet_state_dict"], strict=False
            )
            controlnet.eval()
            logger.info(f"ControlNet loaded from {args.trained_controlnet_path}.")
        except Exception as e:
            logger.warning(f"Could not load ControlNet: {e}")

    # Load Scale Factor
    scale_factor = checkpoint.get("scale_factor", 1.0)
    logger.info(f"Scale Factor -> {scale_factor}.")

    return autoencoder, unet, controlnet, scale_factor


def prepare_control_mask(config, output_size, spacing, device, logger):
    """
    Loads a mask from file OR finds a candidate mask from the database.
    Resamples/Pads it to match the output_size.
    Returns: Binarized ControlNet condition tensor.
    """
    mask_data = None

    # Option A: Explicit Mask Path
    if hasattr(config, "mask_path") and config.mask_path:
        logger.info(f"Loading mask from {config.mask_path}...")
        nii_mask = nib.load(config.mask_path)
        mask_data = nii_mask

    # Option B: Find Mask in Database
    elif hasattr(config, "find_mask") and config.find_mask:
        logger.info("Searching for a suitable mask in database...")
        # Use find_masks from scripts.find_masks
        # We need to map config args to find_masks expectations
        candidates = find_masks(
            body_region=config.diffusion_unet_inference.get("body_region", ["thorax"]),
            anatomy_list=config.diffusion_unet_inference.get("anatomy_list", []),
            spacing=spacing,
            output_size=output_size,
            check_spacing_and_output_size=False,  # Allow loose matching to resize later
            json_file=config.all_mask_files_json,
            data_root=config.all_mask_files_base_dir,
        )
        if not candidates:
            logger.warning("No candidate mask found. Proceeding without ControlNet.")
            return None

        # Pick the first one (arbitrary)
        selected = candidates[0]
        logger.info(f"Selected mask: {selected['mask_file']}")
        nii_mask = nib.load(selected["mask_file"])
        mask_data = nii_mask

    if mask_data is None:
        return None

    # Resample Mask to Target Spacing/Size
    target_affine = np.eye(4)
    for i in range(3):
        target_affine[i, i] = spacing[i]

    # Resample to specific grid (output_size)
    dummy_data = np.zeros(output_size)
    target_img = nib.Nifti1Image(dummy_data, target_affine)

    resampled_mask = nibproc.resample_from_to(
        mask_data,
        target_img,
        order=0,
        mode="constant",
        cval=0,
    )

    mask_np = resampled_mask.get_fdata()

    # Convert to Tensor for ControlNet (Batch, Channel, X, Y, Z)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    # Binarize (using utils.binarize_labels)
    # This assumes the mask contains integer labels (1-132)
    controlnet_cond = binarize_labels(mask_tensor.long()).half()

    return controlnet_cond


def get_repaint_schedule(timesteps, jump_len=10, jump_n_sample=10):
    """
    Generates a RePaint resampling schedule.
    [cite_start]Reference: RePaint Paper [cite: 504-515].

    Args:
        timesteps (Tensor): Monotonically decreasing list of timesteps.
        jump_len (int): Number of steps to jump back (j).
        jump_n_sample (int): Number of times to resample (r).

    Returns:
        List[float]: The full execution schedule of timesteps including jumps.
    """
    ts_list = timesteps.cpu().tolist()
    final_indices = []

    # We start at index 0 (High Noise / Time T)
    curr_idx = 0
    final_indices.append(curr_idx)

    # Iterate until we reach the last step (Data / Time 0)
    # Note: ts_list indices [0, 1, ... N-1] correspond to steps [T, ..., 0]
    while curr_idx < len(ts_list) - 1:
        # 1. Standard Denoise Step (Forward in index, Backward in time)
        curr_idx += 1
        final_indices.append(curr_idx)

        # 2. Resampling Check
        # If we have completed a block of 'jump_len' steps, we resample.
        # We check (curr_idx % jump_len == 0) to align with blocks of size j.
        if curr_idx % jump_len == 0:
            # Repeat the process (r-1) times, because we just performed the 1st pass
            for _ in range(jump_n_sample - 1):
                # A. Diffuse: Jump back 'j' steps (Backward in index, Forward in time)
                for _ in range(jump_len):
                    curr_idx -= 1
                    final_indices.append(curr_idx)

                # B. Denoise: Walk forward 'j' steps again
                for _ in range(jump_len):
                    curr_idx += 1
                    final_indices.append(curr_idx)

    # Map indices back to actual timestep values
    schedule_ts = [ts_list[i] for i in final_indices]
    return schedule_ts


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
    controlnet=None,
    controlnet_cond=None,
):
    """
    Executes the MAISI v2 Rectified Flow Outpainting loop with RePaint support.
    """
    top_tensor, bot_tensor, spacing_tensor, modality_tensor = conditioning_tensors

    if hasattr(args, "diffusion_unet_inference"):
        infer_conf = args.diffusion_unet_inference
    else:
        infer_conf = vars(args)

    # 1. Prepare Inputs
    latent_shape = (
        1,
        args.latent_channels,
        output_size[0] // 4,
        output_size[1] // 4,
        output_size[2] // 4,
    )

    # 2. Encode the Known Crop
    input_norm = (input_ct_crop + 1000.0) / 2000.0
    input_norm = torch.clamp(input_norm, 0.0, 1.0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            z0 = autoencoder.encode_stage_2_inputs(input_norm) * scale_factor

    mask = torch.nn.functional.interpolate(
        input_mask,
        size=latent_shape[-3:],
        mode="nearest",
    )


    # 4. Initialize Noise
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
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"]
        )

    # 6. RePaint Schedule Generation
    # Get standard decreasing timesteps
    standard_timesteps = noise_scheduler.timesteps  # e.g. [999, ..., 0]

    # Get RePaint parameters (defaults to 1 = no resampling if not specified)
    jump_len = infer_conf.get("jump_length", 1)
    jump_n_sample = infer_conf.get("jump_n_sample", 1)

    if jump_len > 1 and jump_n_sample > 1:
        logger.info(
            f"RePaint Strategy Enabled: Jump Length={jump_len}, Resamples={jump_n_sample}"
        )
        repaint_timesteps = get_repaint_schedule(
            standard_timesteps, jump_len, jump_n_sample
        )
    else:
        repaint_timesteps = standard_timesteps.tolist()

    # Create pairs (current, next)
    # Note: repaint_timesteps contains the sequence of 't'.
    # The loop runs transitions.
    timestep_pairs = zip(repaint_timesteps[:-1], repaint_timesteps[1:])
    max_timestep = standard_timesteps[0]  # Used for alpha calculation

    cfg_scale = infer_conf.get("cfg_guidance_scale", 0.0)
    include_body = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    if controlnet is not None and controlnet_cond is not None:
        logger.info("ControlNet is enabled for inference.")
        # Ensure cond is on device
        controlnet_cond = controlnet_cond.to(device)

    logger.info(f"Starting Inference with {len(repaint_timesteps) - 1} transitions...")

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            for t_val, next_t_val in tqdm(
                timestep_pairs, total=len(repaint_timesteps) - 1
            ):
                # Convert to tensor for model input
                t_tensor = torch.Tensor([t_val]).to(device)

                # Detect Direction
                is_denoising = next_t_val < t_val

                # A. Prepare UNet + ControlNet Inputs
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

                if controlnet is not None and controlnet_cond is not None:
                    controlnet_inputs = {
                        "x": latents,
                        "timesteps": t_tensor,
                        "controlnet_cond": controlnet_cond,
                    }
                    if include_modality:
                        controlnet_inputs["class_labels"] = modality_tensor

                    # Run ControlNet
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        **controlnet_inputs
                    )
                    unet_inputs.update(
                        {
                            "down_block_additional_residuals": down_block_res_samples,
                            "mid_block_additional_residual": mid_block_res_sample,
                        }
                    )

                # B. Predict Velocity
                if cfg_scale > 0:
                    out_cond = unet(**unet_inputs)
                    unet_inputs_uncond = {k: v.clone() for k, v in unet_inputs.items()}
                    if "class_labels" in unet_inputs_uncond:
                        unet_inputs_uncond["class_labels"] = torch.zeros_like(
                            modality_tensor
                        )
                    out_uncond = unet(**unet_inputs_uncond)
                    v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
                    del out_cond, out_uncond, unet_inputs_uncond
                else:
                    v_pred = unet(**unet_inputs)

                # C. Step (Bidirectional)
                # noise_scheduler.step handles both forward (dt > 0) and backward (dt < 0)
                # based on the passed timesteps.
                # RFlow: latents_next = latents + v_pred * (next_t - t) / T
                if not isinstance(noise_scheduler, RFlowScheduler):
                    latents, _ = noise_scheduler.step(v_pred, t_val, latents)  # type: ignore
                else:
                    latents, _ = noise_scheduler.step(
                        v_pred, t_val, latents, next_t_val
                    )  # type: ignore

                # D. Force Known Region (Harmonization)
                # We enforce the known region conditions.
                # RePaint suggests doing this during the reverse (denoise) steps
                # to correct the generated content[cite: 160].

                # Calculate the analytical state of the 'known' region at 'next_t_val'
                if not isinstance(noise_scheduler, RFlowScheduler):
                    # DDPM (RePaint) Logic
                    # RePaint samples the known region using the DDPM forward process properties.
                    # Formula: x_t = sqrt(alpha_cumprod) * x0 + sqrt(1 - alpha_cumprod) * epsilon.
                    
                    # Retrieve alpha_cumprod (bar_alpha) for the specific timestep
                    alpha_prod_t = noise_scheduler.alphas_cumprod[next_t_val]
                    
                    z_known_next = (alpha_prod_t ** 0.5) * z0 + ((1 - alpha_prod_t) ** 0.5) * noise_canvas

                else:
                    # Rectified Flow Logic
                    alpha = next_t_val / max_timestep
                    z_known_next = alpha * noise_canvas + (1.0 - alpha) * z0

                # Apply mask: Keep generated (mask=1) parts, Force known (mask=0) parts
                # We apply this primarily when Denoising (moving towards data).
                # When Diffusing (moving towards noise), we allow the regions to mix
                # so the subsequent Denoise step can harmonize the boundary.

                if is_denoising:
                    latents = latents * mask + z_known_next * (1.0 - mask)
                else:
                    pass

    # 7. Decode
    logger.info("Decoding final volume...")
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    sw_roi_size = infer_conf.get("autoencoder_sliding_window_infer_size")
    sw_overlap = infer_conf.get("autoencoder_sliding_window_infer_overlap")

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

    data = synthetic_images.squeeze().cpu().detach().numpy()
    plot_volume_grid(data, 16)
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

    # Controlnet args
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to a specific mask file to use.",
    )
    parser.add_argument(
        "--find_mask",
        action="store_true",
        help="If set, searches database for a mask matching anatomy_list.",
    )

    args = parser.parse_args()

    # Automatically switch to torchrun if using multiple GPUs and not yet distributed
    if args.num_gpus > 1 and "RANK" not in os.environ:
        run_torchrun("scripts.outpaint_inference", sys.argv[1:], args.num_gpus)
        return

    # Load Configurations
    config = load_config(args.env_config, args.model_config, args.model_def)

    if not hasattr(config, "diffusion_unet_inference"):
        config.diffusion_unet_inference = dict(vars(config))

    config.output_dir = args.output_dir
    config.output_prefix = args.output_prefix

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

    output_size = (
        tuple(config.diffusion_unet_inference["dim"])
        if "dim" in config.diffusion_unet_inference
        else tuple(config.diffusion_unet_inference["output_size"])
    )

    autoencoder, unet, controlnet, scale_factor = load_models(config, device, logger)
    # IMPORTANT: this function parses the body_region list from the config (e.g., ["thorax"]) and maps it to indices
    cond_tensors = prepare_tensors(config, device, logger)
    controlnet_cond = None
    if controlnet is not None:
        # Determine spacing explicitly if needed, or use the config spacing
        spacing_val = config.diffusion_unet_inference["spacing"]
        controlnet_cond = prepare_control_mask(
            config, output_size, spacing_val, device, logger
        )

    logger.info(f"Loading input crop from {args.input_crop}...")
    nii_img = nib.load(args.input_crop)

    # --- Start: Crop input to match target physical FOV ---
    target_shape = np.array(config.diffusion_unet_inference["output_size"])
    target_spacing = np.array(config.diffusion_unet_inference["spacing"])

    # Calculate the physical dimensions (FOV) of the target output
    target_fov_mm = target_shape * target_spacing
    
    # Calculate how many voxels in the INPUT image cover that physical distance
    input_spacing = np.array(nii_img.header.get_zooms())
    crop_size_voxels = np.round(target_fov_mm / input_spacing).astype(int)
    
    # Calculate center crop indices
    current_shape = np.array(nii_img.shape)
    start_indices = np.maximum(0, (current_shape - crop_size_voxels) // 2)
    end_indices = np.minimum(current_shape, start_indices + crop_size_voxels)

    # Perform the crop
    slices = tuple(slice(s, e) for s, e in zip(start_indices, end_indices))
    cropped_data = nii_img.get_fdata()[slices]

    # Update the affine matrix: shift the origin to match the new top-left corner
    new_origin_affine = nii_img.affine.copy()
    new_origin_affine[:3, 3] += np.dot(new_origin_affine[:3, :3], start_indices)

    # Replace nii_img with the cropped version
    nii_img = nib.Nifti1Image(cropped_data, new_origin_affine, nii_img.header)
    logger.info(f"Cropped input volume to {nii_img.shape} to match target configuration spacing.")
    # --- End: Crop input ---

    min_image = nii_img.get_fdata().min()

    # Recalculate zoom factors based on the cropped shape
    # Since the crop aligns the physical FOVs, this zoom will result in the correct target spacing
    zoom_factors = np.array(nii_img.shape) / target_shape
    new_affine = nii_img.affine @ np.diag(list(zoom_factors) + [1])

    nii_img = nibproc.resample_from_to(
        nii_img, (target_shape, new_affine), order=3, mode="constant", cval=min_image
    )

    output_spacing = tuple(nib.affines.voxel_sizes(new_affine))
    logger.info(
        f"Resampled from {nii_img.shape} to fixed shape {target_shape}. Image spacing: {nii_img.header.get_zooms()}, expected output spacing: {target_spacing}"
    )

    input_ct = torch.from_numpy(nii_img.get_fdata()).float()

    data_bb = get_volume_bb(nii_img, -1000)
    input_mask = np.ones(output_size)
    input_mask[
        data_bb[0] : data_bb[1],
        data_bb[2] : data_bb[3],
        data_bb[4] : data_bb[5],
    ] = 0
    input_mask = torch.from_numpy(input_mask).float()

    input_mask = input_mask.unsqueeze(0).unsqueeze(0)
    input_ct = input_ct.unsqueeze(0).unsqueeze(0)
    input_ct = input_ct.to(device)
    input_mask = input_mask.to(device)

    result_data = run_outpainting(
        config,
        device,
        autoencoder,
        unet,
        scale_factor,
        cond_tensors,
        input_ct,
        input_mask,
        output_size,
        logger,
        controlnet=controlnet,
        controlnet_cond=controlnet_cond
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_name = f"{args.output_prefix}_{timestamp}.nii.gz"
    out_path = os.path.join(args.output_dir, out_name)

    save_image(result_data, output_size, output_spacing, out_path, logger)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
