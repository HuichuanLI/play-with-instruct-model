import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadLatentAttention(nn.Module):
    def __init__(self,
                 d_model=1024,
                 n_heads=8,
                 d_k=128,
                 d_c=32,
                 d_hR=32):
        '''
        '''

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_c = d_c
        self.d_hR = d_hR

        # 权重矩阵
        self.W_DQ = nn.Linear(d_model, d_c)
        self.W_UQ = nn.Linear(d_c, n_heads * d_k)
        self.W_DKV = nn.Linear(d_model, d_c)
        self.W_UK = nn.Linear(d_c, n_heads * d_k)
        self.W_UV = nn.Linear(d_c, n_heads * d_k)
        self.W_QR = nn.Linear(d_c, n_heads * d_hR)
        self.W_KR = nn.Linear(d_model, d_hR)

        self.W_O = nn.Linear(n_heads * d_k, d_model)

    def forward(self, h_t, mask=None):
        batch_size, seqlen, _ = h_t.shape

        c_t_Q = self.W_DQ(h_t)
        q_t_c = self.W_UQ(c_t_Q).view(batch_size, seqlen, self.n_heads, self.d_k).transpose(1, 2)

        c_t_KV = self.W_DKV(h_t)
        k_t_C = self.W_UK(c_t_KV).view(batch_size, seqlen, self.n_heads, self.d_k).transpose(1, 2)
        v_t_C = self.W_UV(c_t_KV).view(batch_size, seqlen, self.n_heads, self.d_k).transpose(1, 2)

        q_t_R = self.W_QR(c_t_Q).view(batch_size, seqlen, self.n_heads, self.d_k).transpose(1, 2)
        q_t_R = self.apply_rope(q_t_R)

        k_t_R = self.W_KR(h_t).view(batch_size, seqlen, 1, self.d_k).transpose(1, 2)
        k_t_R = self.apply_rope(k_t_R)
        k_t_R = k_t_R.expand(-1, self.n_heads, -1, -1)  # batch_szie, n_heads, seq_len, d_k
        q_t = torch.concat([q_t_c, q_t_R], dim=-1)
        k_t = torch.concat([k_t_C, k_t_R], dim=-1)

        attn_scores = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(self.d_k + self.d_hR)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        o_t = torch.matmul(attn_weights, v_t_C)  # batch_size, n_heads, seq_len, d_k
        o_t = o_t.transpose(1, 2).contiguous().view(batch_size, seqlen, self.n_heads * self.d_k)

        u_t = self.W_O(o_t)

        return u_t
