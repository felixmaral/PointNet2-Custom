import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from pathlib import Path

def detect_num_classes(label_dir):
    label_files = sorted(glob(str(Path(label_dir) / "*.seg")))
    unique_labels = set()
    for f in label_files:
        labels = np.loadtxt(f).astype(int)
        unique_labels.update(np.unique(labels))
    return max(unique_labels) + 1

def get_min_num_points(folder):
    min_points = float('inf')
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".npy"):
            points = np.load(os.path.join(folder, fname))
            min_points = min(min_points, points.shape[0])
    return int(min_points)

class ShapeNetDataset(Dataset):
    def __init__(self, x_path, y_path, num_points=4096):
        all_x_files = sorted([os.path.join(x_path, f) for f in os.listdir(x_path) if f.endswith('.npy')])
        all_y_files = sorted([os.path.join(y_path, f) for f in os.listdir(y_path) if f.endswith('.seg')])
        assert len(all_x_files) == len(all_y_files), "Mismatch between .npy and .seg files"

        self.x_files = []
        self.y_files = []
        self.num_points = num_points

        for x_file, y_file in zip(all_x_files, all_y_files):
            points = np.load(x_file)
            if points.shape[0] >= num_points:
                self.x_files.append(x_file)
                self.y_files.append(y_file)

        print(f"✅ Loaded {len(self.x_files)} samples with ≥ {num_points} points.")

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        points = np.load(self.x_files[idx]).astype(np.float32)[:, :3]
        labels = np.loadtxt(self.y_files[idx], dtype=np.uint32)

        if len(points) != len(labels):
            raise ValueError(f"Mismatch between points and labels in {self.x_files[idx]}")

        # Muestreo aleatorio sin repetición
        choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        points = points[choice]
        labels = labels[choice]

        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)