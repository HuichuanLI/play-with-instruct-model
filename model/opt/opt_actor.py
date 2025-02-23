# model
# -- opt
# -- -- opt_actor.py
from typing import Optional
from transformers.models.opt import OPTConfig, OPTForCausalLM

from ..base.actor import Actor

class OPTActor(Actor):
    def __init__(self, pretrained: Optional[str] = None,
            config: Optional[OPTConfig] = None,
            checkpoint: bool = False,
            cache_dir = None) -> None:
        if pretrained is not  None:
            model = OPTForCausalLM.from_pretrained(pretrained, cache_dir=cache_dir)
        elif config is not None:
            model = OPTForCausalLM(config)
        else:
            model = OPTForCausalLM(OPTConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super(OPTActor, self).__init__(model)