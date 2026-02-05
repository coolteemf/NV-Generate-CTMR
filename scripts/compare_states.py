import numpy as np
import matplotlib.pyplot as plt

# Sentinel object to represent a missing key
MISSING = object()


def inspect_difference(name, val_old, val_new):
    """
    Visualizes differences with size checks to avoid console flooding.
    """
    print(f"\n🔍 INSPECTING: {name}")

    # --- Helper to safely get size/repr ---
    def get_info(val):
        if val is MISSING:
            return "MISSING", 0
        if isinstance(val, np.ndarray):
            return val, val.size
        if isinstance(val, (list, tuple, dict)):
            return val, len(val)
        return val, 1

    obj_old, size_old = get_info(val_old)
    obj_new, size_new = get_info(val_new)

    # --- Rule 1: Handle Missing Keys ("ONLY IN...") ---
    if val_old is MISSING:
        print(f"   ❌ Key missing in OLD state.")
        print(f"   👉 NEW Value: {obj_new}")
        return
    if val_new is MISSING:
        print(f"   ❌ Key missing in NEW state.")
        print(f"   👉 OLD Value: {obj_old}")
        return

    # --- Rule 2: Handle Large Arrays/Lists (> 10,000 elements) ---
    threshold = 10000
    is_large = (size_old > threshold) or (size_new > threshold)

    if is_large:
        print(f"   ⚠️ Large Data detected (Size > {threshold}). Skipping text print.")
        print(f"   📏 Shape/Len: Old={np.shape(obj_old)}, New={np.shape(obj_new)}")

        # If it's a numeric array, give us stats instead of raw data
        if isinstance(obj_old, np.ndarray) and isinstance(obj_new, np.ndarray):
            print(
                f"   📊 Stats Old: Min={obj_old.min():.2f}, Max={obj_old.max():.2f}, Mean={obj_old.mean():.2f}"
            )
            print(
                f"   📊 Stats New: Min={obj_new.min():.2f}, Max={obj_new.max():.2f}, Mean={obj_new.mean():.2f}"
            )
            # Trigger Plotting for large arrays
            plot_array_diff(name, obj_old, obj_new)
        return

    # --- Rule 3: Print Actual Values (Small Data) ---
    # This catches your args.anatomy_list, indices, etc.
    print(f"   🔻 OLD value: {obj_old}")
    print(f"   🔻 NEW value: {obj_new}")

    # If they are small numpy arrays, we can explicitly show the difference
    if isinstance(obj_old, np.ndarray) and isinstance(obj_new, np.ndarray):
        # Optional: Plot even small arrays if they are 2D+
        if obj_old.ndim >= 2:
            plot_array_diff(name, obj_old, obj_new)


def plot_array_diff(name, arr_old, arr_new):
    """
    Plots center slice of 3D/4D/5D arrays.
    """
    arr_old = arr_old.astype(float)
    arr_new = arr_new.astype(float)
    try:
        # Squeeze batch dim (1, C, D, H, W) -> (C, D, H, W)
        if arr_old.shape[0] == 1:
            arr_old, arr_new = arr_old.squeeze(0), arr_new.squeeze(0)

        # Select first channel (C, ...) -> (...)
        if arr_old.ndim > 3:
            arr_old, arr_new = arr_old[0], arr_new[0]

        # Select center slice (D, H, W) -> (H, W)
        if arr_old.ndim == 3:
            mid = arr_old.shape[0] // 2
            arr_old, arr_new = arr_old[mid], arr_new[mid]

        if arr_old.ndim != 2:
            return

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Mismatch: {name}", fontsize=14)

        for a, title, data in zip(
            ax, ["Old", "New", "Diff"], [arr_old, arr_new, arr_old - arr_new]
        ):
            im = a.imshow(data, cmap="gray" if title != "Diff" else "seismic")
            a.set_title(title)
            plt.colorbar(im, ax=a)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"   (Could not plot: {e})")


def compare_recursive(val_old, val_new, path=""):
    # 1. Dictionaries
    if isinstance(val_old, dict) and isinstance(val_new, dict):
        all_keys = set(val_old.keys()) | set(val_new.keys())
        for k in sorted(all_keys):
            curr_path = f"{path}.{k}" if path else k
            if k not in val_old:
                inspect_difference(curr_path, MISSING, val_new[k])
            elif k not in val_new:
                inspect_difference(curr_path, val_old[k], MISSING)
            else:
                compare_recursive(val_old[k], val_new[k], curr_path)

    # 2. Lists
    elif isinstance(val_old, list) and isinstance(val_new, list):
        if len(val_old) != len(val_new):
            # Pass full lists to inspector so we see what's inside
            inspect_difference(path, val_old, val_new)
        else:
            for i, (item_old, item_new) in enumerate(zip(val_old, val_new)):
                compare_recursive(item_old, item_new, f"{path}[{i}]")

    # 3. Numpy Arrays
    elif isinstance(val_old, np.ndarray) and isinstance(val_new, np.ndarray):
        if val_old.dtype == object and val_old.shape == ():
            compare_recursive(val_old.item(), val_new.item(), path)
            return

        if val_old.shape != val_new.shape:
            inspect_difference(path, val_old, val_new)
        else:
            if np.issubdtype(val_old.dtype, np.number) and np.issubdtype(
                val_new.dtype, np.number
            ):
                if not np.allclose(val_old, val_new, atol=1e-5):
                    # Found mismatch, trigger inspector
                    inspect_difference(path, val_old, val_new)
            else:
                if not np.array_equal(val_old, val_new):
                    inspect_difference(path, val_old, val_new)

    # 4. Primitives
    else:
        if val_old != val_new:
            inspect_difference(path, val_old, val_new)


def compare_states(path_old, path_new):
    print(f"Comparing '{path_old}' vs '{path_new}'...")
    old_archive = np.load(path_old, allow_pickle=True)
    new_archive = np.load(path_new, allow_pickle=True)

    # Unwrap top-level
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
    compare_states("debug_working.npz", "debug_non-working-reset-config.npz")