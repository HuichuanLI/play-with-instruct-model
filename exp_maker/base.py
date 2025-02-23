# exp_maker
# -- base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from model.base_package.actor import Actor

# 封装单次交互的完整经验数据，用于后续策略（Actor）和价值（Critic）模型的优化。
# sequences：输入提示（Prompt）+ 模型生成的动作（Action）组成的完整序列，形状为 [batch_size, seq_len]。
#
# action_log_probs：模型生成动作时每个Token的对数概率，形状 [batch_size, num_actions]。
#
# values：Critic模型对输入提示（状态）的价值估计，形状 [batch_size]。
#
# reward：奖励模型给出的奖励值（结合人类偏好和KL惩罚），形状 [batch_size]。
#
# advantages：优势函数值（用于PPO策略更新），形状 [batch_size]。
#
# attention_mask：标识有效Token位置（如忽略填充部分），形状同sequences。
#
# action_mask：标识动作部分的Token位置，形状同sequences。

@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.reward = self.reward.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.reward = self.reward.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self

# 1. 用Actor模型生成动作（文本）。
#
# 2. 用Critic模型计算状态价值。
#
# 3.用奖励模型计算奖励（结合KL散度惩罚）。
#
# 4. 计算优势函数。
#
# 5. 封装所有数据为Experience对象。
class ExpMaker(ABC):
    def __init__(self, actor: Actor,
                 critic: nn.Module,
                 reward_model: nn.Module,
                 initial_model: nn.Module,
                 kl_coef: float = 0.1):
        super(ExpMaker, self).__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.kl_coef = kl_coef

        # actor：策略模型（如GPT），负责生成动作（文本回复）。
        #
        # critic：价值模型，评估输入提示的价值。
        #
        # reward_model：奖励模型，为生成内容打分（基于人类偏好）。
        #
        # initial_model：初始模型（预训练模型），用于计算KL散度惩罚，防止策略偏离太远。
        #
        # kl_coef：KL散度的惩罚系数，平衡奖励与策略稳定性。

    @abstractmethod
    def make_experience(self, input_ids: torch.Tensor,
                        **generate_kwargs):
        raise NotImplementedError('make experience not imp')
