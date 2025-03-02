import torch
from torch import nn


class PromptEncoder(nn.Module):
    def __init__(self, config):
        super(PromptEncoder, self).__init__()
        self.token_dim = config.token_dim
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.hidden_size = config.encoder_hidden_size
        self.total_virtual_tokens = config.num_virtual_tokens * \
                                    config.num_transformer_submodules
        # LSTM + MLP / MLP
        self.embedding = nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not config.inference_mode:
            lstm_dropout = config.encoder_dropout
            num_layers = config.encoder_num_layers
            self.lstm_head = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                bidirectional=True,
                batch_first=True,
            )
            self.mlp_head = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, self.output_size)
            )

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        return output_embeds
