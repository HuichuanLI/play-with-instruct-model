# model
# -- base
# -- -- critic.py
from typing import Optional
import torch
import torch.nn as nn

from ..utils import masked_mean


class Critic(nn.Module):
    def __init__(self, model: nn.Module,
                 value_head: nn.Module,
                 use_action_mask: bool = False,
                 ):
        super(Critic, self).__init__()
        self.model = model
        self.value_head = value_head
        self.use_action_mask = use_action_mask

    def forward(self, input_ids: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # 把数据穿过去
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # 最后一层mha[batch, seq_len, d_dim]
        last_hidden_states = outputs['last_hidden_state']
        # 把所有的结果输出给reward model计算出一个值
        # [batch, seq_len, 1] -> [batch, seq_len]
        values = self.value_head(last_hidden_states).squeeze(-1)
        # 如果action_mask
        if action_mask is not None and self.use_action_mask:
            num_actions = action_mask.size(1)
            # [batch, seq_len-num_action]
            prompt_mask = attention_mask[:, :-num_actions]
            values = values[:, :-num_actions]
            value = masked_mean(values, prompt_mask, dim=1)
            return value

        # [batch, seq_len-1]
        values = values[:, :-1]
        value = values.mean(dim=1)
        return value
