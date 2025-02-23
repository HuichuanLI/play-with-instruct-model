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
    # logits：模型输出的原始分数，形状为[batch, seq_len, vocab_size]。
    # labels：目标Token的ID序列，形状为[batch, seq_len]。
    # [batch, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    # [batch, seq_len] -> [batch, seq_len, 1]
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    # [batch, seq_len]
    return log_probs_labels.squeeze(-1)


def calc_action_log_probs(output: torch.Tensor,
                          sequences: torch.Tensor,
                          num_actions: int) -> torch.Tensor:
    # output：语言模型的输出，包含logits（形状[batch, seq_len, vocab_size]）。
    # sequences：完整的输入序列（Prompt + 生成的Action），形状 [batch, seq_len]。
    # num_actions：生成动作的Token数量。
    logits = output['logits']
    # [batch, seq_len[:-1], vocab_size]
    # [batch, seq_len[1:], vocab_size]
    log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
    # [batch, seq_len[-num_action:]]
    return log_probs[:, -num_actions:]


def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_old: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    log_ratio = log_probs - log_probs_old
    # p * log (q/p)
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1)
    return approx_kl


def compute_reward(r, kl_coef: float,
                   log_probs: torch.Tensor,
                   log_probs_old: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if kl_coef <= 0.0:
        return r
    kl = compute_approx_kl(log_probs, log_probs_old, action_mask=action_mask)
    reward = r - kl_coef * kl
    return reward
