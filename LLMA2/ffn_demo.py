from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # gate
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # down_projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # up_projection
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ffn = FeedForward(dim=512,hidden_dim=2048, multiple_of=64, ffn_dim_multiplier=1)
# x  = torch.randn(32,100,512)
# print(ffn(x).shape)
