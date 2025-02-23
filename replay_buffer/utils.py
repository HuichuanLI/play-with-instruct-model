# replay_buffer
# -- utils.py

# experience [sequences[16,xx]/reward[16]/values[16]]
# buffer [ sequences[xx]/reward[1]/values[1]]

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from exp_maker.base import Experience


@dataclass
class BufferItem:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.BoolTensor]


def split_experience_batch(exp: Experience) -> List[BufferItem]:
    batch_size = exp.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = ('sequences',
            'action_log_probs',
            'values',
            'reward',
            'advantages',
            'attention_mask',
            'action_mask')
    for k in keys:
        val = getattr(exp, k)
        if isinstance(val, torch.Tensor):
            vals = torch.unbind(val)
        else:
            vals = [val for _ in range(batch_size)]
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][k] = v
    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor],
                       side: str = 'left') -> torch.Tensor:
    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_seqs.append(F.pad(seq, padding))
    # [Tensor[seq_len], Tensor[seq_len]]
    # Tensor[batch, seq_len]
    return torch.stack(padded_seqs, dim=0)


def make_experience_batch(items: List[BufferItem]) -> Experience:
    kwargs = {}
    to_pad_keys = ('action_log_probs', 'action_mask')
    keys = ('sequences',
            'action_log_probs',
            'values',
            'reward',
            'advantages',
            'attention_mask',
            'action_mask')
    for k in keys:
        vals = [getattr(item, k) for item in items]
        if k in to_pad_keys:
            batch_data = zero_pad_sequences(vals)
        else:
            batch_data = torch.stack(vals, dim=0)
        kwargs[k] = batch_data
    return Experience(**kwargs)
