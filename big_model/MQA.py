# MQA

import torch
import torch.nn as nn
import math


class MQA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must divisible by num_heads"
        self.head_dim = d_model // num_heads

        self.qeury = nn.Linear(d_model, self.d_model)
        self.key = nn.Linear(d_model, self.head_dim)
        self.val = nn.Linear(d_model, self.head_dim)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        query = self.qeury(x)
        key = self.key(x).unsqueeze(1)  # batch_size, 1,seq_len, head_dim
        val = self.val(x).unsqueeze(1)  # batch_size, 1, seq_len, head_dim

        Q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                     2)  # [batch_size, num_heads, seq_length, head_dim]
        print(Q.size())
        print(key.transpose(-2, -1).size())
        # attention
        scores = torch.matmul(Q, key.transpose(-2, -1)) / math.sqrt(d_model)
        print(scores.size())

        scores = torch.softmax(scores, dim=-1)  # batch_size, seq_len, seq_len
        attn_output = torch.matmul(scores, val)  # batch_size, num_heads, seq_len, head_dim

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        attn_output = self.linear(attn_output)

        return attn_output, scores


batch_size, seq_len, d_model = 16, 10, 768

mqa = MQA(d_model, 12)

x = torch.randn(batch_size, seq_len, d_model)

output, _ = mqa(x)

print(f"output is {output.size()}")
