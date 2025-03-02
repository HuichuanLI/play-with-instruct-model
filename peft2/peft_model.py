import torch
from torch import nn
from transformers import PreTrainedModel
from prompt_tuning import PromptEmbedding
from p_tuning import PromptEncoder
from prefix_tuning import PrefixEncoder
from peft import PeftType, PromptLearningConfig
from lora import LoraModel


class PeftModel(nn.Module):
    #
    def __init__(self, model, peft_config):
        super(PeftModel, self).__init__()
        self.peft_config = peft_config
        self.base_model = model
        if isinstance(self.peft_config, PromptLearningConfig):
            self._setup_prompt_encoder()
        else:
            self.base_model = LoraModel(peft_config, model)

    def _setup_prompt_encoder(self):
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if self.peft_config.num_transformer_submodules is None:
            self.peft_config.num_transformer_submodules = (
                2 if self.peft_config.task_type == 'seq_2_seq_lm' else 1
            )
        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(
                    named_param.replace('.weight', ''))
                break
        if self.peft_config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(self.peft_config, self.word_embeddings)
        elif self.peft_config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(self.peft_config)
        elif self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(self.peft_config)
        else:
            raise ValueError('not support')
        self.prompt_encoder = prompt_encoder
        # [0,1,2,3,4]
        self.prompt_tokens = torch.arange(self.peft_config.num_virtual_tokens *
                                          self.peft_config.num_transformer_submodules).long()

    def get_prompt(self, batch_size):
        # [1, 10] -> [batch_size, 10]
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1)
        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config.num_virtual_tokens]
            if self.peft_config.inference_mode:
                past_key_values = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = self.prompt_encoder(prompt_tokens)
            # [batch_size, num_v_token, num_layers * 2, n_heads, token_dim//n_heads]
            #
            past_key_values = past_key_values.view(
                batch_size,
                self.peft_config.num_virtual_tokens,
                self.peft_config.num_layers * 2,
                self.peft_config.num_attention_heads,
                self.peft_config.token_dim // self.peft_config.num_attention_heads
            )
            if self.peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            # [ num_layers, batch_size, n_heads, num_v_token, token_dim//n_heads]
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                self.peft_config.num_transformer_submodules * 2
            )
            return past_key_values
        else:
            if self.peft_config.inference_mode:
                prompts = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = self.prompt_encoder(prompt_tokens)
        return prompts


class PeftModelForCausalLM(PeftModel):
    def __init__(self, model, peft_config):
        super(PeftModelForCausalLM, self).__init__(model, peft_config)

    def forward(self, input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs
                ):
        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            prefix_att_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens)
            attention_mask = torch.cat((prefix_att_mask, attention_mask), dim=1)

        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.peft_config.num_virtual_tokens), -100)
                kwargs['labels'] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
