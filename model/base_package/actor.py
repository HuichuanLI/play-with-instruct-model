# model
# -- base
# -- -- actor.py

from typing import Optional
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, model: nn.Module):
        super(Actor, self).__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **model_kwargs,
                ) -> torch.Tensor:
        output = self.model(input_ids,
                            attention_mask=attention_mask, **model_kwargs)
        return output
