import math

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from rope_demo import apply_rotary_emb, precompute_freqs_cis


# 定义 ModelArgs 类
@dataclasses.dataclass
class ModelArgs:
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    max_batch_size: int = 32
    max_seq_len: int = 2048


def repeat_kv(x, n_rep):
    bz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bz, seqlen, n_kv_heads, n_rep, head_dim)
            .reshape(bz, seqlen, n_kv_heads * n_rep, head_dim)
            )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        model_parallel_size = 1
        # 将多头的分配到每一张gpu
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_kv_heads = self.n_kv_heads // model_parallel_size
        # 假设kv的头和q的头的数量是不一致的，所以需要将kv的头的数量复制到和q的头相同的数量
        self.n_rep = self.n_local_heads // self.n_kv_heads

        self.head_dim = args.dim // args.n_heads

        # [512,512]
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # [512,512]
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # [512,512]
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # [512,512]
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim)

        # [32,2048,8,64]
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)

        # [32,2048,8,64]
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)

    def forward(self, x, start_pos, freqs_cis, mask):
        # 1.x->wq，wk、wv-》q、k、v
        # 2.q、k、v 【b,seq,dim】-》view ->dim -> head*head_dim(拆多头) ->[b,seq,head*head_dim]
        # 3.q、k ->rope-> softmax(q*k^T/dim)*v ->output->wo->outpot->[b,seq,dim】
        bz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # [32, 2048, 8, 64]  k/v[32,100,8,64] [32,1,8,24] start_pos=100 seqlen=1  101
        self.cache_k[:bz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bz, start_pos:start_pos + seqlen] = xv

        keys = self.cache_k[:bz, :start_pos + seqlen]
        values = self.cache_v[:bz, :start_pos + seqlen]

        # 分组的时候，都说 kv_heads*n_rep-> n_local_heads
        # [b,seq,n_local_head,head_dim] q,k,v
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # transpose(1,2) [b,n_local_head,seq,head_dim] q,k,v
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # q[b,n_local_head,seq,head_dim]  @ k^T [b,n_local_head,head_dim，seq] -》[b,n_local_head,seq，seq]
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # scores [b,n_local_head,seq，seq] @ values [b,n_local_head,seq,head_dim] ->[b,n_local_head,seq,head_dim]
        output = torch.matmul(scores, values)
        # transpose(1,2)->[b,seq,n_local_head,head_dim]->[b,seq,dim】
        output = output.transpose(1, 2).contiguous().view(bz, seqlen, -1)
        return self.wo(output)


# 第一次是需要mask
def create_mask(seq_len, n_heads):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    mask = mask.repeat(n_heads, 1, 1)
    return mask


args = ModelArgs(dim=512, n_heads=8, max_batch_size=32, max_seq_len=200)
attention = Attention(args)

x = torch.randn(1, 50, 512)

# 创建mask
mask = create_mask(50, 8)
mask = mask.unsqueeze(0).expand(1, -1, -1, -1)

# [400,64]
freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)
freqs_cis_1 = freqs_cis[:50, :]

# 第一次forward
ouput = attention(x, start_pos=0, freqs_cis=freqs_cis_1, mask=mask)
print(ouput.shape)

# 第二次forward
x_2 = torch.randn(1, 1, 512)

freqs_cis_2 = freqs_cis[50:50 + 1, :]

output_2 = attention(x_2, start_pos=50, freqs_cis=freqs_cis_2, mask=None)
print(output_2.shape)
