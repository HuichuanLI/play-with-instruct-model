
import torch
from torch import nn

class PrefixEncoder(nn.Module):
    def __init__(self, config):
        super(PrefixEncoder, self).__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            self.embedding = nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = nn.Sequential(
                nn.Linear(token_dim, encoder_hidden_size),
                nn.Tanh(),
                # k,v
                nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim)
            )
        else:
            self.embedding = nn.Embedding(num_virtual_tokens,
                num_layers * 2 * token_dim
            )

    def forward(self, prefix):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)

        return past_key_values

