import torch
import torch.nn as nn


# batch, sequence_length, C
# batch, sequence_length, dim = 3, 4, 5
#
#
# input = torch.randn(batch, sequence_length, dim)


# # batch_norm
# batch_norm = nn.BatchNorm1d(dim,affine=False)
# # (N, C, L)   ###(N,L,C) ->trans->(N, C, L)->(N, L, c)
# batch_norm_input_api = batch_norm(input.transpose(-1,-2)).transpose(-1,-2)

# (N, L, c) (N,L) mean std
# batch_norm_mean = input.mean(dim=(0,1),keepdim=True)
# batch_norm_std = input.std(dim=(0,1),keepdim=True,unbiased=False)
#
# batch_norm_input_1 = (input-batch_norm_mean)/(batch_norm_std+1e-5)
#
# print(batch_norm_input_api)
# print(batch_norm_input_1)


# layer_norm
# layer_norm = nn.LayerNorm(dim,elementwise_affine=False)
# layer_norm_api = layer_norm(input)
#
# layer_norm_mean = input.mean(dim=-1,keepdim=True)
# layer_norm_std = input.std(dim=-1,keepdim=True,unbiased=False)
#
# layer_norm_input_1 = (input-layer_norm_mean)/(layer_norm_std+1e-5)
#
# print(layer_norm_api)
# print(layer_norm_input_1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()) * self.weight
        return output

# rms_norm = RMSNorm(dim)
# print(rms_norm(input))
