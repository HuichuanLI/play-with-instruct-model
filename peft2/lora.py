import torch
from torch import nn
import torch.nn.functional as F


class LoraModel(nn.Module):
    def __init__(self, config, model):
        super(LoraModel, self).__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()

    def _find_and_replace(self):
        # embedding
        pass


def mark_only_lora_as_trainable():
    pass


class LoraLayer:
    def __init__(self,
                 r: int,
                 # W + alpha * w_delta
                 lora_alpha: int,
                 lora_dropout: float,
                 merge_weights: bool
                 ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # current
        self.merged = False
        self.merge_weights = merge_weights
        # lora
        self.disable_adapters = False


def transpose(weight):
    return weight.T


class Linear(nn.Linear, LoraLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 0,  # 4
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.0,
                 merge_weights: bool = True,
                 **kwargs
                 ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)
        if r > 0:
            # A*B = W
            # [r, in]
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # [out, r]
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            if self.r > 0:
                # torch.matmul()
                # [out, in]
                self.weight.data += (
                        transpose(self.lora_B.weight @ self.lora_A.weight) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= (
                        transpose(self.lora_B.weight @ self.lora_A.weight) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                        transpose(self.lora_B.weight @ self.lora_A.weight) * self.scaling
                )
                self.merged = False
            return F.linear(x, transpose(self.weight), bias=self.bias)
        elif self.r > 0 and not self.merged:
            # W + w_delta
            # xW + x*w_delta
            result = F.linear(x, transpose(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight), bias=self.bias)


from typing import List


class MergedLinear(nn.Linear, LoraLayer):
    # q,k,v = xW
    # [True, False, True]
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 0,  # 4
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.0,
                 enable_lora: List[bool] = [False],
                 merge_weights: bool = True,
                 **kwargs):

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)
        self.enable_lora = enable_lora
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
            # [1,2,3,4]
            self.lora_B = nn.Conv1d(
                r * sum(enable_lora),
                out_features // len(enable_lora) * sum(enable_lora),
                kernel_size=1,
                groups=2,
                bias=False
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            # [out]->[3, out/3]
            self.lora_ind = self.weight.new_zeros((out_features),
                                                  dtype=torch.bool).view(len(enable_lora), -1)
            # [0, out/3], [2, out/3]
            self.lora_ind[enable_lora, :] = True
            # [out]
            self.lora_ind = self.lora_ind.view(-1)

    def zero_pad(self, x):
        # [batch, seq_len, out/3*2]
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[: self.lora_ind] = x.reshape(-1,
                                            self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        # [batch, seq_len, out]
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.weight.data.unsqueeze(0),  # [batch, r*sum(enable_lora), in]
                    self.lora_B.weight.data.unsqueeze(-1),  # [out/3*2, r*sum(enable_lora), 1]
                    # [r1;r2], [r1;r2]
                    groups=sum(self.enable_lora)
                ).squeeze(0)  # [out', in]
                # [in, out/3*2]
                self.weight.data += self.zero_pad(transpose(delta_w * self.scaling))
            self.merged = True
        elif self.merge_weights and self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.weight.data.unsqueeze(0),  # [batch, r*sum(enable_lora), in]
                    self.lora_B.weight.data.unsqueeze(-1),  # [out/3*2, r*sum(enable_lora), 1]
                    # [r1;r2], [r1;r2]
                    groups=sum(self.enable_lora)
                ).squeeze(0)  # [in, out/3*2]
                self.weight.data -= self.zero_pad(transpose(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.weight.data.unsqueeze(0),  # [batch, r*sum(enable_lora), in]
                    self.lora_B.weight.data.unsqueeze(-1),  # [out/3*2, r*sum(enable_lora), 1]
                    # [r1;r2], [r1;r2]
                    groups=sum(self.enable_lora)).squeeze(0)  # [in, out/3*2]
                self.weight.data -= self.zero_pad(transpose(
                    delta_w * self.scaling
                ))
                self.merged = False
            return F.linear(x, transpose(self.weight), bias=self.bias)
        elif self.merged:
            return F.linear(x, transpose(self.weight), bias=self.bias)
        else:
            result = F.linear(x, transpose(self.weight), bias=self.bias)
            if self.r > 0:
                # [seq_len, in] -> [seq_len, r]
                after_A = self.lora_A(self.lora_dropout(x))
                # [r, seq_len] -> [out', seq_len] -> [seq_len, out']
                after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result
