import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    '''Scaled Dot-Product Attention
    计算注意力权重的公式
    attention(Q,K,V) = softmax((Q*K^T)/sqrt(d_model))

    '''

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        '''
        进行注意力权重的计算
        :param query: 查询向量，[batch_size, seq_len, d_model]
        :param key: 键值向量，[batch_size, seq_len. d_model]
        :param value: 值向量，[batch_size, seq_len, d_model]
        :return:
            attention_weight:注意力权重 [batch_size, seq_len, seq_len]
            output 注意力输出 [batch_size, seq_len, d_model]
        '''
        d_q = query.size()[-1]

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_q, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = torch.softmax(scores, dim=-1)

        output = torch.matmul(scores, value)

        return output, scores


sdpta = ScaledDotProductAttention()

batch_size, seq_len, d_model = 16, 20, 768
query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)
output, scores = sdpta(query, key, value)

print(f"output size is {output.size()}")
print(f"score size is {scores.size()}")
print(f"score is {scores}")

# 生成一个mask
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
print(mask)
mask.size()

output, scores = sdpta(query, key, value, mask=mask)

print(f"output size is {output.size()}")
print(f"score size is {scores.size()}")
print(f"score is {scores[0]}")


class MultiheadAttention(nn.Module):
    '''
    multi head attention 模块
    将输入分割为多个头（heads），在每个头上独立计算Scaled Dot-Product Attention，
    最后将结果拼接并通过线性变换输出。
    '''

    def __init__(self, d_model, num_heads):
        '''
        -
        :param d_model: 模型维度
        :param num_heads: 注意力的头数
        :param dropout: Dropout比率
        '''
        super(MultiheadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be divisiable by num_heads"

        self.head_dim = self.d_model // self.num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        '''
        :param x: [batch_size, num_heads, seq_len, head_dim]
        :return:
        '''
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x, mask):
        batch_size, seq_len, d_model = x.size()

        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.val_proj(x)

        query_splited = self.split_heads(query)
        key_splited = self.split_heads(key)
        value_splited = self.split_heads(value)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attention_out, attention_weights = self.attention(query_splited, key_splited, value_splited, mask)
        # attention_out [batch_size, num_heads, seq_len, head_dim]

        attention_out = self.combine_heads(attention_out)

        return self.out_proj(attention_out), attention_weights


class MultiheadAttention(nn.Module):
    '''
    multi head attention 模块
    将输入分割为多个头（heads），在每个头上独立计算Scaled Dot-Product Attention，
    最后将结果拼接并通过线性变换输出。
    '''

    def __init__(self, d_model, num_heads):
        '''
        -
        :param d_model: 模型维度
        :param num_heads: 注意力的头数
        :param dropout: Dropout比率
        '''
        super(MultiheadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be divisiable by num_heads"

        self.head_dim = self.d_model // self.num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        '''
        :param x: [batch_size, num_heads, seq_len, head_dim]
        :return:
        '''
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x, mask):
        batch_size, seq_len, d_model = x.size()

        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.val_proj(x)

        query_splited = self.split_heads(query)
        key_splited = self.split_heads(key)
        value_splited = self.split_heads(value)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attention_out, attention_weights = self.attention(query_splited, key_splited, value_splited, mask)
        # attention_out [batch_size, num_heads, seq_len, head_dim]

        attention_out = self.combine_heads(attention_out)

        return self.out_proj(attention_out), attention_weights


class MaskedMultiHeadAttention(MultiheadAttention):
    """Masked Multi-Head Attention模块

    专用于Transformer decoder的自注意力层，
    添加未来掩码（future-masking）以防止位置i关注位置j>i的token。
    """

    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__(d_model, num_heads)

    def forward(self, x):
        """
        参数:
        - query: 查询向量 [batch_size, seq_len, d_model]
        - key: 键向量 [batch_size, seq_len, d_model]
        - value: 值向量 [batch_size, seq_len, d_model]

        返回:
        - 带掩码的多头注意力输出 [batch_size, seq_len, d_model]
        - 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        # 获取序列长度
        seq_len = x.size(1)

        # 创建未来掩码（下三角矩阵，对角线为1，上三角为0）
        # [1, seq_len, seq_len]
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

        # 调用父类的forward方法并传入掩码
        return super().forward(x, mask)


batch_size, seq_len, d_model = 16, 20, 768

mmha = MaskedMultiHeadAttention(d_model, num_heads=12)
x = torch.randn(batch_size, seq_len, d_model)

output, scores = mmha(x)

print(f"output size is {output.size()}")
print(f"score size is {scores.size()}")
print(f"score is {scores[0]}")

# Transfomer Decoder:
# 1. token embedding
# 2. position embedding
# 3. Multihead attention
# 4. Feed forward Nural network
# 5. attention +FFN ：decoder block
# 6. 多个decoder堆叠起来
# 7.laynorm、activation、残差连接

import torch
import torch.nn as nn


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNeuralNetwork, self).__init__()
        # laynorm
        self.layer_norm = nn.LayerNorm(d_model)
        # proj
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        # 激活函数
        self.activation = nn.GELU()

    def forward(self, x):
        '''

        :param x: [batch_szie, seq_len, hidden_size]
        :return:
        '''
        resiual = x
        output = self.layer_norm(x)  # [batch_szie, seq_len, hidden_size]
        output = self.linear1(output)  # [batch_szie, seq_len, hidden_size*4]
        output = self.activation(output)  # [batch_szie, seq_len, hidden_size*4]
        output = self.linear2(output)  # [batch_szie, seq_len, hidden_size]

        return resiual + output


batch_size, seq_len, hidden_size = 16, 10, 768

x = torch.randn(batch_size, seq_len, hidden_size)

ffn = FeedForwardNeuralNetwork(768, 768 * 4)

output = ffn(x)

print(f"x size is {x.size()}")
print(f"output size is {output.size()}")
print(f"output is {output[0]}")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads

        super(MultiHeadAttention, self).__init__()
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = self.d_model // self.num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        '''

        :param x: [batch_size, seq_len, d_model]
        :return:
            [batch_size, num_heads, seq_len, head_dim]
        '''
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.head_dim * self.num_heads, f"input must in dim {self.num_heads * self.head_dim} but input dim is {d_model}"

        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        '''

        :param x:  [batch_size, num_heads, seq_len, head_dim]
        :return:  [batch_size, seq_len, num_heads*head_dim]
        '''
        batch_size, num_heads, seq_len, head_dim = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x, mask=None):
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        splited_query = self.split_heads(query)
        splited_key = self.split_heads(key)
        splited_value = self.split_heads(value)

        output, scores = self.attention(splited_query, splited_key, splited_value, mask)
        output = self.combine_heads(output)

        return self.out_proj(output), scores


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerDecoderBlock, self).__init__()
        # 核心模块
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feedForward = FeedForwardNeuralNetwork(d_model, d_ff)
        # 后层归一化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attn_output, attn_weights = self.attention(x, attn_mask)

        ff_output = self.feedForward(x + attn_output)

        output = self.layer_norm(ff_output)

        return output, attn_weights


batch_size, seq_len, hidden_size = 16, 10, 768

x = torch.randn(batch_size, seq_len, hidden_size)

tdb = TransformerDecoderBlock(hidden_size, 12, hidden_size * 4)

output, attn_weights = tdb(x)
print(f"x size is {x.size()}")
print(f"output size is {output.size()}")
print(f"output is {output[0]}")

# Transformer实现

import math


class PositionalEncoding(nn.Module):
    """位置编码模块（支持动态序列长度）"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # 动态获取位置编码
        position_emb = self.pe[:, :x.size(1)]
        return x + position_emb  # [batch, seq_len, d_model]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_layers, num_heads, d_ff):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # 堆叠Decoder块
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        # tied embeddings
        self.output_layer.weight = self.token_embedding.weight

        self.init_weights()
        pass

    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # 各层做一下初始化
        for layer in self.layers:
            nn.init.xavier_normal(layer.attention.query_proj.weight)
            nn.init.xavier_normal(layer.attention.key_proj.weight)
            nn.init.xavier_normal(layer.attention.value_proj.weight)

            nn.init.kaiming_normal(layer.feedForward.linear1.weight)
            nn.init.kaiming_uniform(layer.feedForward.linear2.weight)

    def create_causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask

    def forward(self, input_ids):
        '''

        :param x:[batch_size, seq_len]
        :return:
        '''
        batch_size, seq_len = input_ids.size()

        # 嵌入
        embeddings = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        pos_embedding = self.pos_encoder(embeddings)

        embeddings = embeddings + pos_embedding

        mask = self.create_causal_mask(seq_len)
        # 通过所有的Transformer Decoder block
        hidden_states = embeddings
        all_attn_weights = []
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, mask)
            all_attn_weights.append(attn_weights)

        hidden_states = self.final_norm(hidden_states)

        logits = self.output_layer(hidden_states)

        return logits, all_attn_weights


model = TransformerDecoder(
    vocab_size=500, d_model=256, max_len=128, num_layers=12, num_heads=8, d_ff=256 * 4)

batch_size, seq_len = 16, 10

input_ids = torch.randint(0, 500, (batch_size, seq_len))

print(f"input size is {input_ids.size()}")
print(f"input is {input_ids}")

output, _ = model(input_ids)

print(f"output size is {output.size()}")
print(f"output is {output[-1]}")
