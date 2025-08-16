import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.pointnet2 import PointNet2SemSeg

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Warning: Neither MPS nor CUDA is available. Using CPU.")
        device = torch.device("cpu")

    model = PointNet2SemSeg(num_classes=9).to(device)
    x = torch.randn(2, 3, 5000).to(device)
    out = model(x)
    print("Output shape:", out.shape)  # Expected: (2, 9, 5000)