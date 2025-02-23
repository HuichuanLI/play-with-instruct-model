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


def log_probs_from_logits(logits: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
    # [batch, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    # [batch, seq_len] -> [batch, seq_len, 1]
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    # [batch, seq_len]
    return log_probs_labels.squeeze(-1)
