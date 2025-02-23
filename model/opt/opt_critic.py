# model
# -- opt
# -- -- opt_critic.py
from typing import Optional
import torch
from torch import nn
from transformers.models.opt import OPTConfig, OPTModel
from ..base.critic import Critic

class OPTCritic(Critic):
    def __init__(self, pretrained: Optional[str] = None,
            config: Optional[OPTConfig] = None,
            checkpoint: bool = False,
            cache_dir = None,
            **kwargs) -> None:
        if pretrained is not None:
            model = OPTModel.from_pretrained(pretrained, cache_dir=cache_dir)
        elif config is not None:
            model = OPTModel(config)
        else:
            model = OPTModel(OPTConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.word_embed_proj_dim, 1)
        super(OPTCritic, self).__init__(model, value_head, **kwargs)