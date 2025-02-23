from typing import Optional
import torch
from torch import Tensor
from torch import nn
from .utils import masked_mean


# pair wise
class LogSigLoss(nn.Module):
    def forward(self, chosen_reward: torch.Tensor,
                reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = - log_probs.mean()
        return loss


# ptx
# CausalLM.forward()
class LMLoss(nn.Module):
    def __init__(self):
        super(LMLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, vocab_size]
        # [a, b, c]
        # [b, c, d]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # [batch*seq_len, vocab_size]
        # [batch*seq_len]
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)),
                         shift_labels.view(-1))


# Actor策略损失 Policy gradient的函数
# 目标：通过限制策略更新的幅度，确保新策略不会偏离旧策略太远（避免训练不稳定）。
# 关键操作：
# 概率比（ratio）：计算新旧策略下动作概率的比值 exp(log_probs - old_log_probs)。
# 替代目标（surrogate objectives）：
# surr1：未截断的替代目标（直接使用概率比）。
# surr2：截断后的替代目标（将概率比限制在 [1-clip_eps, 1+clip_eps] 范围内）。
# 损失计算：取 surr1 和 surr2 的最小值，确保更新不会过于激进。
class PolicyLoss(nn.Module):
    def __init__(self, clip_eps: float = 0.2):
        super(PolicyLoss, self).__init__()
        self.clip_eps = clip_eps

    def forward(self,
                log_probs: Tensor,
                old_log_probs: Tensor,
                advantages: Tensor,
                action_mask: Optional[Tensor] = None
                ) -> Tensor:
        # new/old
        # log new/old
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = - torch.min(surr1, surr2)
        if action_mask is not None:
            loss = masked_mean(loss, action_mask)
            return loss
        loss = loss.mean()
        return loss


# ValueLoss（价值损失） crtic loss
# 目标：优化价值网络（Critic），使其预测的价值更接近真实回报（reward）。
# 关键操作：
# 截断价值（values_clip）：限制新价值的更新幅度不超过 clip_eps。
# 损失计算：取截断和未截断损失的较大值，确保更新保守。
class ValueLoss(nn.Module):
    # MSE
    def __init__(self, clip_eps: float = 0.4):
        super(ValueLoss, self).__init__()
        self.clip_eps = clip_eps

    def forward(self,
                values: Tensor,
                old_values: Tensor,
                reward: Tensor
                ) -> Tensor:
        values_clip = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        surr1 = (values_clip - reward) ** 2
        surr2 = (values - reward) ** 2
        loss = torch.max(surr1, surr2)
        loss = loss.mean() * 0.5
        return loss

# 策略损失（policy_loss）：使用 PolicyLoss 计算 PPO 的策略损失。
# 预训练损失（lm_loss）：使用 LMLoss（语言模型损失）保持模型的语言生成能力。
# 联合损失：将两者加权求和，权重由 pretrain_coef 控制。
class PPOPtxActorLoss(nn.Module):
    def __init__(self, policy_clip_eps: float = 0.2,
                 pretrain_coef: float = 0.0,
                 pretrain_loss_fn=LMLoss()) -> None:
        super(PPOPtxActorLoss, self).__init__()
        self.pretrain_coef = pretrain_coef
        self.pretrain_loss_fn = pretrain_loss_fn
        self.policy_loss_fn = PolicyLoss(policy_clip_eps)

    def forward(self,
                log_probs: Tensor,
                old_log_probs: Tensor,
                advantages: Tensor,
                lm_logits: Tensor,
                lm_input_ids: Tensor,
                action_mask: Optional[Tensor] = None
                ) -> Tensor:
        policy_loss = self.policy_loss_fn(log_probs,
                                          old_log_probs, advantages, action_mask=action_mask)
        lm_loss = self.pretrain_loss_fn(lm_logits, lm_input_ids)
        return policy_loss + self.pretrain_coef * lm_loss
