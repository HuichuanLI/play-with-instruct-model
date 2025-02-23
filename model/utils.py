from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


def masked_mean(tensor: torch.Tensor,
                mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # [1,1,1,1,0,0,0]
    tensor = tensor * mask
    s = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = s / (mask_sum + 1e-8)
    return mean
