import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_mIoU(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """
    Compute mean Intersection over Union (mIoU) for semantic segmentation.

    Args:
        preds (torch.Tensor): Predicted labels (B, N)
        labels (torch.Tensor): Ground truth labels (B, N)
        num_classes (int): Number of classes

    Returns:
        float: Mean IoU across all classes
    """
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    ious = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls

        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


def compute_precision(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute average precision over all classes.

    Args:
        preds (torch.Tensor): Predicted labels (B, N)
        labels (torch.Tensor): Ground truth labels (B, N)

    Returns:
        float: Macro-averaged precision
    """
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    return precision_score(labels, preds, average='macro', zero_division=0)


def compute_recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute average recall over all classes.

    Args:
        preds (torch.Tensor): Predicted labels (B, N)
        labels (torch.Tensor): Ground truth labels (B, N)

    Returns:
        float: Macro-averaged recall
    """
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    return recall_score(labels, preds, average='macro', zero_division=0)


def compute_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute F1-score over all classes.

    Args:
        preds (torch.Tensor): Predicted labels (B, N)
        labels (torch.Tensor): Ground truth labels (B, N)

    Returns:
        float: Macro-averaged F1-score
    """
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    return f1_score(labels, preds, average='macro', zero_division=0)