import torch
import torch.nn as nn


def lossfunc(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    kiszámoljuk a maszkolt l1 veszteséget a ritka mélységtérképen.
    preds: háló tippje [Batch, 1, H, W]
    targets: valódi mélység (depth) [Batch, 1, H, W]
    masks: bináris maszk (gt_mask) [Batch, 1, H, W]
    """
    absolute_error=torch.abs(preds-targets)
    masked_error=absolute_error*masks

    total_loss=torch.sum(masked_error)
    valid_pixels=torch.sum(masks)

    if valid_pixels==0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    return total_loss/valid_pixels