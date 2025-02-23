# models
# -- opt
# -- -- opt_rm.py

from typing import Optional
import torch.nn as nn
from transformers import OPTModel, OPTConfig
from model.base.reward_model import RewardModel


class OPTRM(RewardModel):
    def __init__(self, pretrained: Optional[str] = None,
                 config: Optional[OPTConfig] = None,
                 checkpoint: bool = False,
                 cache_dir=None
                 ) -> None:
        if pretrained is not None:  # cache/cache/facebook/opt-125m/
            model = OPTModel.from_pretrained(pretrained, cache_dir=cache_dir)
        elif config is not None:
            model = OPTModel(config)
        else:
            model = OPTModel(OPTConfig())
        value_head = nn.Linear(model.config.word_embed_proj_dim, 1)
        value_head.weight.data.normal_(mean=0.0,
                                       std=1 / (model.config.word_embed_proj_dim + 1))
        super(OPTRM, self).__init__(model, value_head)
