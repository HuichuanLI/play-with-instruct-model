# trainer
# -- ppo.py

from typing import Dict, List
import torch
from torch import nn
from exp_maker.naive import Experience, NaiveExpMaker
# from exp_maker.
from model.base_package import Actor, Critic, RewardModel
from model.loss import LMLoss, PolicyLoss, ValueLoss
from model.utils import calc_action_log_probs
from replay_buffer.naive import NaiveReplayBuffer
from torch import Tensor
from torch.optim import Optimizer
from .base import OnPolicyTrainer, to_device


# 这个PPOTrainer类是基于OnPolicyTrainer框架实现的PPO（Proximal Policy Optimization）算法，专为强化学习与语言模型结合的场景设计（如对话生成）。以下是逐层解析：
# （1）模型与优化器
# Actor：策略网络，生成动作（如生成文本）。
#
# Critic：价值网络，评估状态价值。
#
# Reward Model：计算即时奖励（如文本质量评估）。
#
# Initial Model：参考模型（通常为初始策略），用于计算KL散度防止策略突变。
#
# (2) 损失函数
# PolicyLoss：PPO-Clip策略损失，限制新旧策略差异。
#
# ValueLoss：价值函数损失，带clip机制稳定训练。
#
# LMLoss：语言模型损失（监督学习），防止遗忘预训练知识。
# (3) 超参数
# kl_coef：KL散度惩罚系数。
#
# ptx_coef：预训练损失混合比例。
#
# eps_clip：策略更新的clip范围。
#
# vf_coef：价值损失权重。
class PPOTrainer(OnPolicyTrainer):
    def __init__(self,
                 actor: Actor,
                 critic: Critic,
                 reward_model: nn.Module,
                 initial_model: nn.Module,
                 actor_optim: Optimizer,
                 critic_optim: Optimizer,
                 kl_coef: float = 0.1,
                 ptx_coef: float = 0.9,
                 train_batch_size: int = 8,
                 buffer_limit: int = 0,
                 buffer_cpu_offload: bool = True,
                 eps_clip: float = 0.2,
                 vf_coef: float = 1.0,
                 value_clip: float = 0.4,
                 offload_inference_models: bool = True,
                 device='cpu',
                 **generate_kwargs
                 ):
        buffer = NaiveReplayBuffer(train_batch_size,
                                   buffer_limit, buffer_cpu_offload)
        super(PPOTrainer, self).__init__(buffer)
        self.generate_kwargs = generate_kwargs
        self.experience_maker = NaiveExpMaker(actor, critic, reward_model, initial_model, kl_coef)
        self.offload_inference_models = offload_inference_models
        self.actor = actor
        self.critic = critic
        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.vf_coef = vf_coef
        self.ptx_loss_fn = LMLoss()
        self.ptx_coef = ptx_coef
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.device = torch.device(device)

    def _make_experience(self, collect_step: int) -> Experience:
        prompts = self.prompt_dataloader.next()
        if self.offload_inference_models:
            self.experience_maker.initial_model.to(self.device)
            self.experience_maker.reward_model.to(self.device)
        if isinstance(prompts, Tensor):
            return self.experience_maker.make_experience(prompts,
                                                         **self.generate_kwargs)
        elif isinstance(prompts, dict):
            return self.experience_maker.make_experience(**prompts,
                                                         **self.generate_kwargs)
        else:
            raise ValueError(f'unsupported type {type(prompts)}')

    def _training_step(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()
        # [batch, seq_len]
        num_actions = experience.action_mask.size(1)
        actor_output = self.actor(experience.sequences,
                                  attention_mask=experience.attention_mask)
        action_log_probs = calc_action_log_probs(actor_output,
                                                 experience.sequences, num_actions)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)

        # ptx loss
        if self.ptx_coef != 0:
            batch = self.pretrain_dataloader.next()
            batch = to_device(batch, self.device)
            ptx_log_probs = self.actor(batch['input_ids'],
                                       attention_mask=batch['attention_mask'])['logits']
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, batch['labels'])
            actor_loss = ptx_loss * self.ptx_coef + actor_loss * (1 - self.ptx_coef)

        actor_loss.backward()
        self.actor_optim.step()
        self.actor_optim.zero_grad()

        # value loss
        values = self.critic(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask
                             )
        critic_loss = self.critic_loss_fn(values, experience.values,
                                          experience.reward, action_mask=experience.action_mask)
        critic_loss = critic_loss * self.vf_coef
        critic_loss.backward()
        self.critic_optim.step()
        self.critic_optim.zero_grad()

        return {'reward': experience.reward.mean().item()}

    def _learn(self, update_step: int):
        if self.offload_inference_models:
            self.experience_maker.initial_model.to('cpu')
            self.experience_maker.reward_model.to('cpu')
        experience = self.buffer.sample()
        experience.to_device(self.device)
        self._training_step(experience)
