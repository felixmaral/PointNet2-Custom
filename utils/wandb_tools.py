import numpy as np
import wandb
from matplotlib import cm

def label_to_color(labels):
    import matplotlib.pyplot as plt
    import numpy as np

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = plt.get_cmap("tab20", num_classes)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    colors = np.array([cmap(label_to_idx[l])[:3] for l in labels]) * 255
    return colors.astype(np.uint8)

def log_pointcloud_to_wandb(points, labels, name="PointCloud", step=0):
    points = np.asarray(points)
    labels = np.asarray(labels)

    if points.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {points.shape}")

    # Map original class labels (e.g. 236, 705, etc.) → 0, 1, ..., N-1
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = np.vectorize(label_to_index.get)(labels)  # same shape as labels

    # Generate a palette and assign colors
    colors = label_to_color(unique_labels)

    print(f"[DEBUG] {name} | Unique classes: {unique_labels}")
    print(f"[DEBUG] points: {points.shape} {points.dtype}")
    print(f"[DEBUG] colors: {colors.shape} {colors.dtype}")

    obj = {
        "type": "lidar/beta",
        "points": points,
        "colors": colors
    }

    wandb.log({name: wandb.Object3D(obj)})