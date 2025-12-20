import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeepSeekMOE(nn.Module):
    def __init__(self,
                 h_model,
                 num_experts,
                 num_shared,
                 top_k,
                 hidden_dim
                 ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_shared = num_shared

        self.shared_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, h_model)
                )
            ] for _ in range(len(num_shared))
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, h_model)
                )
            ] for _ in range(len(num_experts))
        )
        self.router = nn.Linear(h_model, num_experts)

    def forward(self, x: torch.Tensor):
        share_out = torch.sum([expert(x) for expert in self.shared_experts])

        logits = self.router(x)
        scores = torch.softmax(logits, dim=-1)

        top_scores, top_indices = torch.topk(scores, k=self.top_k)

        expert_out = torch.zeros(share_out.shape)
        for index in top_indices:
            expert_out += self.experts[index](x)

        return share_out + expert_out
