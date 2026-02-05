import numpy as np
import matplotlib.pyplot as plt


def inspect_difference(name, val_old, val_new):
    """
    Intelligently visualizes the difference based on data type.
    """
    print(f"\n🔍 INSPECTING: {name}")

    # 1. Handle Arrays (Plotting)
    if isinstance(val_old, np.ndarray) and isinstance(val_new, np.ndarray):
        # Only plot if it has spatial dimensions
        if val_old.ndim >= 2:
            plot_array_diff(name, val_old, val_new)
        else:
            print(f"   Old: {val_old}")
            print(f"   New: {val_new}")

    # 2. Handle Lists/Primitives (Printing)
    else:
        print(f"   🔻 OLD value: {val_old}")
        print(f"   🔻 NEW value: {val_new}")

        # If list lengths differ, print lengths explicitly
        if isinstance(val_old, (list, tuple)) and isinstance(val_new, (list, tuple)):
            print(f"   📏 Lengths: Old={len(val_old)}, New={len(val_new)}")


def plot_array_diff(name, arr_old, arr_new):
    """
    Plots Old, New, and Difference maps.
    Handles high-dim tensors by slicing the center of the first channel.
    """
    arr_old = arr_old.astype(float)
    arr_new = arr_new.astype(float)
    # Squeeze batch dim if present (1, C, D, H, W) -> (C, D, H, W)
    if arr_old.shape[0] == 1:
        arr_old = arr_old.squeeze(0)
        arr_new = arr_new.squeeze(0)

    # Select first channel if multi-channel (C, D, H, W) -> (D, H, W)
    if arr_old.ndim > 3:
        print(f"   ℹ️ Slicing channel 0 for visualization.")
        arr_old = arr_old[0]
        arr_new = arr_new[0]

    # Select center slice of depth if 3D (D, H, W) -> (H, W)
    if arr_old.ndim == 3:
        mid_slice = arr_old.shape[0] // 2
        print(f"   ℹ️ Slicing depth index {mid_slice} for visualization.")
        arr_old = arr_old[mid_slice]
        arr_new = arr_new[mid_slice]

    # Ensure we are down to 2D
    if arr_old.ndim != 2:
        print(
            f"   ⚠️ Could not simplify array to 2D image (Dims: {arr_old.shape}). Skipping plot."
        )
        return

    # Create Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Mismatch: {name}", fontsize=16)

    # Plot Old
    im1 = axes[0].imshow(arr_old, cmap="gray")
    axes[0].set_title("Old Version")
    plt.colorbar(im1, ax=axes[0])

    # Plot New
    im2 = axes[1].imshow(arr_new, cmap="gray")
    axes[1].set_title("New Version")
    plt.colorbar(im2, ax=axes[1])

    # Plot Difference
    diff = arr_old - arr_new
    limit = np.max(np.abs(diff))  # center colormap at 0
    im3 = axes[2].imshow(diff, cmap="seismic", vmin=-limit, vmax=limit)
    axes[2].set_title(f"Difference (MSE: {np.mean(diff**2):.2e})")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


def compare_recursive(val_old, val_new, path=""):
    """
    Recursively compares and triggers inspection on mismatch.
    """
    # [Dictionary and List traversal logic remains mostly the same...]

    if isinstance(val_old, dict) and isinstance(val_new, dict):
        all_keys = set(val_old.keys()) | set(val_new.keys())
        for k in sorted(all_keys):
            curr_path = f"{path}.{k}" if path else k
            if k not in val_old:
                print(f"❓ {curr_path}: ONLY in NEW")
            elif k not in val_new:
                print(f"❓ {curr_path}: ONLY in OLD")
            else:
                compare_recursive(val_old[k], val_new[k], curr_path)

    elif isinstance(val_old, list) and isinstance(val_new, list):
        if len(val_old) != len(val_new):
            print(f"❌ {path}: Length mismatch {len(val_old)} vs {len(val_new)}")
            # TRIGGER INSPECTION
            inspect_difference(path, val_old, val_new)
        else:
            for i, (item_old, item_new) in enumerate(zip(val_old, val_new)):
                compare_recursive(item_old, item_new, f"{path}[{i}]")

    elif isinstance(val_old, np.ndarray) and isinstance(val_new, np.ndarray):
        # Unwrap 0-d object arrays
        if val_old.dtype == object and val_old.shape == ():
            compare_recursive(val_old.item(), val_new.item(), path)
            return

        if val_old.shape != val_new.shape:
            print(f"❌ {path}: Shape mismatch {val_old.shape} vs {val_new.shape}")
            inspect_difference(path, val_old, val_new)
        else:
            if np.issubdtype(val_old.dtype, np.number) and np.issubdtype(
                val_new.dtype, np.number
            ):
                if not np.allclose(val_old, val_new, atol=1e-5):
                    mse = np.mean((val_old - val_new) ** 2)
                    print(f"❌ {path}: Mismatch! (MSE: {mse:.2e})")
                    # TRIGGER INSPECTION
                    inspect_difference(path, val_old, val_new)
            else:
                if not np.array_equal(val_old, val_new):
                    print(f"❌ {path}: Content mismatch")
                    inspect_difference(path, val_old, val_new)

    else:
        if val_old != val_new:
            print(f"❌ {path}: Value mismatch")
            inspect_difference(path, val_old, val_new)


def compare_states(path_old, path_new):
    print(f"Loading {path_old}...")
    old_archive = np.load(path_old, allow_pickle=True)
    print(f"Loading {path_new}...")
    new_archive = np.load(path_new, allow_pickle=True)

    print("-" * 50)

    # Unwrap main dictionaries
    old_dict = {
        k: v.item() if v.shape == () and v.dtype == object else v
        for k, v in old_archive.items()
    }
    new_dict = {
        k: v.item() if v.shape == () and v.dtype == object else v
        for k, v in new_archive.items()
    }

    compare_recursive(old_dict, new_dict)

if __name__ == "__main__":
    compare_states("debug_working.npz", "debug_non-working.npz")