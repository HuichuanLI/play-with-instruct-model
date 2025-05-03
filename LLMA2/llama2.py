from  rope_demo import precompute_freqs_cis,apply_rotary_emb
from norm import RMSNorm
from  attention_demo import Attention
from ffn_demo import  FeedForward
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 2
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 1000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

class TransformerBlock(nn.Module):
    def __init__(self,layer_id:int,args:ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self,x,start_pos,freqs_cis,mask):
        h = x+self.attention(self.attention_norm(x),start_pos,freqs_cis,mask)

        out = h+self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self,params:ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        # DECODER  layer
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
          self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
    def forward(self,tokens,start_pos):
        _bsz,seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]

        mask = None
        # a,b
        #[[0,inf]]
        #[[0,0]]
        if seqlen >1:
            mask = torch.full((seqlen,seqlen),float('-inf'))
            mask = torch.triu(mask,diagonal=1)

            mask = torch.hstack([torch.zeros((seqlen,start_pos)),mask])
        for layer in self.layers:
            h = layer(h,start_pos,freqs_cis,mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


args = ModelArgs()
llama2= Transformer(args)
x = torch.randint(0,1000,(2,200))
print(llama2(x,0).shape)
