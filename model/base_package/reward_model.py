# models
# -- base_package
# -- -- reward_model.py

from typing import Optional
import torch
from torch import nn


class RewardModel(nn.Module):
    def __init__(self, model: nn.Module,
                 value_head: Optional[nn.Module] = None) -> None:
        super(RewardModel, self).__init__()
        self.model = model
        if value_head is None:
            self.value_head = nn.Linear(model.config.n_embd, 1)
        else:
            if value_head.out_features != 1:
                raise ValueError('output dim is not 1')
            self.value_head = value_head

    def forward(self, sequences: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        # [batch, seq_len, n_embd]
        # [batch, seq_len]
        # [batch, seq_len-1]
        values = self.value_head(last_hidden_states)[:, :-1]
        # [batch, 1] -> [batch]
        value = values.mean(dim=1).squeeze(1)
        return value
