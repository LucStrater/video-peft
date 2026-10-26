import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

class BoxTrackingLoss(nn.Module):
    """Smooth L1 + GIoU over the tracked boxes, weighted as in InternVideo-Next
    (arXiv 2512.01342). Boxes are xyxy normalized to [0, 1]."""

    def __init__(self, l1_weight=1.0, giou_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

    def forward(self, pred, target, valid):
        # pred/target: (B, Q, T-1, 4), valid: (B, Q, T-1)
        pred, target = pred[valid], target[valid]
        if pred.numel() == 0:
            return pred.sum()

        l1 = F.smooth_l1_loss(pred, target)

        # giou is only defined for x2 > x1; clamping (not sorting) keeps an inverted
        # box penalized instead of scoring it as its own mirror image
        pred_valid = torch.cat([pred[:, :2], torch.max(pred[:, 2:], pred[:, :2])], dim=-1)
        giou = generalized_box_iou_loss(pred_valid, target, reduction='mean')
        return self.l1_weight * l1 + self.giou_weight * giou

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, min_depth=1e-3, max_depth=10.0):
        super().__init__()
        self.lambd = lambd
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred, target):
        valid = (target > self.min_depth) & (target < self.max_depth)
        pred_valid = pred[valid].clamp(min=self.min_depth)
        target_valid = target[valid]

        log_diff = torch.log(pred_valid) - torch.log(target_valid)
        silog = torch.mean(log_diff ** 2) - self.lambd * (torch.mean(log_diff) ** 2)
        return silog