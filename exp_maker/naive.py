# exp_maker
# -- naive.py
import torch
from model.generate import generate_with_actor
from model.utils import calc_action_log_probs, compute_reward

from .base import Experience, ExpMaker


class NaiveExpMaker(ExpMaker):

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor,
                        **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()
        # 使用Actor模型（如GPT）根据输入提示生成回复（动作）。
        # 标识生成部分（动作）的Token位置（如[False, ..., True, True]）
        sequences, attention_mask, action_mask = generate_with_actor(
            self.actor, input_ids, return_action_mask=True,
            **generate_kwargs
        )

        # action_log_probs：当前策略（Actor）生成动作的Token对数概率。
        # base_action_log_probs：初始模型（旧策略）生成相同动作的概率。
        num_actions = action_mask.size(1)
        actor_output = self.actor(sequences, attention_mask)
        action_log_probs = calc_action_log_probs(
            actor_output, sequences, num_actions)
        base_model_output = self.initial_model(sequences, attention_mask)
        base_action_log_probs = calc_action_log_probs(
            base_model_output, sequences, num_actions)
        # 作用：Critic模型评估输入序列（状态）的价值（value），即预期长期回报。
        value = self.critic(sequences, action_mask, attention_mask)
        # 原始奖励：奖励模型（reward_model）对生成内容打分（如是否符合人类偏好）。
        # KL惩罚项：计算当前策略与初始策略的KL散度，加权后从奖励中扣除。
        # 掩码聚合：仅对动作部分的Token计算KL散度（通过action_mask）。
        r = self.reward_model(sequences, attention_mask)
        reward = compute_reward(r, self.kl_coef,
                                action_log_probs, base_action_log_probs, action_mask=action_mask)
        # tde = r + value(next) - value(current)
        advantage = reward - value
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)
        return Experience(sequences, action_log_probs,
                          value, reward, advantage, attention_mask, action_mask)
