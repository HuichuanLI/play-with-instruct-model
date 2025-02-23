from typing import Optional
import torch
from torch import Tensor
from torch import nn


# pair wise
class LogSigLoss(nn.Module):
    def forward(self, chosen_reward: torch.Tensor,
                reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = - log_probs.mean()
        return loss
