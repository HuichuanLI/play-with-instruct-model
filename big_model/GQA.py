# GQA

import torch
import torch.nn as nn
import math
class GQA(nn.Module):
    def __init__(self, d_model, head_dim, num_q_heads, num_kv_groups=None):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.num_q_heads = num_q_heads

        assert num_q_heads % num_kv_groups == 0, "num_q_heads must be divisible by num_kv_groups"
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, num_kv_groups * head_dim)
        self.val = nn.Linear(d_model, num_kv_groups * head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.query(x)  # batch_szie, seq_len, d_model
        K = self.key(x)  # batch_size, seq_len, num_kv_groups*head_dim
        V = self.val(x)  # batch_szie, seq_len, num_kv_groups*head_dim

        # Q*KT
        # Q ：batch_size, num_q_heads, seq_len, head_dim
        # K: batch_size, num_kv_groups, seq_len, head_dim
        Q = Q.view(batch_size, seq_len, self.num_q_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        K = torch.repeat_interleave(K, self.num_q_heads // self.num_kv_groups,
                                    1)  # batch_size, num_q_heads, seq_len, head_dim

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        # batch_size, num_q_heads, seq_len, seq_len
        # V batch_szie, seq_len, num_kv_groups*head_dim

        V = V.view(batch_size, seq_len, num_kv_groups, head_dim).transpose(1, 2)
        V = torch.repeat_interleave(V, self.num_q_heads // self.num_kv_groups,
                                    1)  # batch_size, num_q_heads, seq_len, head_dim
        scores = torch.softmax(scores, dim=-1)

        attn_out = torch.matmul(scores, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, num_q_heads * head_dim)

        output = self.out_proj(attn_out)
