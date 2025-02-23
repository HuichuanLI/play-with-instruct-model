# model
# -- generate.py

from typing import Any, Callable, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


def prepare_logits_precessor(top_k: Optional[int] = None,
                             top_p: Optional[float] = None,
                             temperature: Optional[float] = None) -> LogitsProcessorList:
    precessor_list = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        # [batch, 1, vocab_size]
        precessor_list.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        precessor_list.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        precessor_list.append(TopPLogitsWarper(top_p))
    return precessor_list


def _is_sequence_finished(unfinished_sequences: torch.Tensor) -> bool:
    return unfinished_sequences.max() == 0


def generate(model: nn.Module,
             input_ids: torch.LongTensor,
             max_length: int,
             early_stopping: bool = False,
             eos_token_id: Optional[int] = None,
             pad_token_id: Optional[int] = None,
             tok_k: Optional[int] = None,
             top_p: Optional[float] = None,
             temperature: Optional[float] = None,
             prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
             update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
             **model_kwargs) -> torch.Tensor:
    if input_ids.size(1) >= max_length:
        return input_ids
    logits_processor = prepare_logits_precessor(tok_k, top_p, temperature)
    # [batch]
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    for _ in range(input_ids.size(1), max_length):
        model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) \
            if prepare_inputs_fn is not None else {'input_ids': input_ids}
        outputs = model(**model_inputs)
        next_token_logits = outputs['logits'][:, -1, :]
        next_token_logits = logits_processor(input_ids, next_token_logits)
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        # [batch]
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError('eos or pad choose one')
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        # [batch, 1]
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, model_kwargs)
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        if early_stopping and _is_sequence_finished(unfinished_sequences):
            break
    return input_ids


@torch.no_grad()
def generate_with_actor(actor_model: nn.Module,
                        input_ids: torch.LongTensor,
                        return_action_mask: bool = True,
                        **kwargs):
    sequences = generate(actor_model, input_ids, **kwargs)
    attention_mask = None
    pad_token_id = kwargs.get('pad_token_id', None)
    if pad_token_id is not None:
        attention_mask = sequences.not_equal(pad_token_id
                                             ).to(dtype=torch.long, device=sequences.device)
    if not return_action_mask:
        return sequences, attention_mask, None
    input_len = input_ids.size(1)
    eos_token_id = kwargs.get('eos_token_id', None)
    if eos_token_id is None:
        action_mask = torch.ones_like(sequences, dtype=torch.bool)
    else:
        # [bacth, seq_len-input_len] -> [0,0,0, 1,1,1] -> [1,1,1,0,0,0]
        action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
        # [1,1,1,   1,1,1,0,0]
        action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)
        # [0,0,1,   1,1,1,0,0]
        action_mask[:, :input_len] = False
        # [  0,1,   1,1,1,0,0]
        action_mask = action_mask[:, 1:]
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):]
