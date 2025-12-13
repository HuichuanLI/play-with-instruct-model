import torch
import torch.nn as nn
import math


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


def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emd(x, cos, sin):
    # cos : [seq_len, head_dim]
    # x : [batch_size, num_head, seq_len, head_dim] * [head_dim, seq_len]
    return x * cos + rotate_half(x) * sin


class MultiheadAttention(nn.Module):
    '''
    multi head attention 模块
    将输入分割为多个头（heads），在每个头上独立计算Scaled Dot-Product Attention，
    最后将结果拼接并通过线性变换输出。
    '''

    def __init__(self, d_model, num_heads, base):
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
        self.rotaryEmbedding = RotaryEmbedding(base, self.head_dim)

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
        # 应用旋转位置矩阵
        cos, sin = self.rotaryEmbedding(seq_len)
        # cos : [seq_len, head_dim]
        # query_splited : [batch_size, num_head, seq_len, head_dim]
        query_splited = apply_rotary_pos_emd(query_splited, cos, sin)
        key_splited = apply_rotary_pos_emd(key_splited, cos, sin)
        attention_out, attention_weights = self.attention(query_splited, key_splited, value_splited, mask)
        # attention_out [batch_size, num_heads, seq_len, head_dim]

        attention_out = self.combine_heads(attention_out)

        return self.out_proj(attention_out), attention_weights


class MaskedMultiHeadAttention(MultiheadAttention):
    """Masked Multi-Head Attention模块

    专用于Transformer decoder的自注意力层，
    添加未来掩码（future-masking）以防止位置i关注位置j>i的token。
    """

    def __init__(self, base, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__(d_model, num_heads, base)

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


import math


class RotaryEmbedding(nn.Module):
    def __init__(self, base, head_dim, traning_seq_len):
        super().__init__()
        self.base = base
        self.dim = head_dim
        self.traning_seq_len = traning_seq_len
        pass

    def get_ntk_alpha(self, true_seq_len):
        # train_seq_len
        # ntk_alpha = 2**(log2(true_seq_len/self.traning_seq_len)) +1
        context_value = math.log(true_seq_len / self.traning_seq_len, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def get_mscaling(self, scale=1):
        '''
        true_seq_len/self.traning_seq_len
        :param scale:
        :return:
        '''
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def forward(self, max_seq_len):
        '''
        ntk aware + attention scaling
        (
            cosmth0,
            cosmth1,
            ....
            cosmth63,
            cosmth0,
            cosmth1,
            ....
            cosmth63
        )

         (
            sinmth0,
            sinmth1,
            ....
            sinmth63,
            sinmth0,
            sinmth1,
            ....
            sinmth63
        )
        m就是token所处在的位置
        th_i = 1/ base**(2i/d)
        [1/base**0, 1/base**(2*1/d), 1/base**(2*2/d), ..., 1/base**(2*d/2)/d]
        :param x: 待添加位置信息的Tensor Q或者K
        :param max_seq_len: 最大位置信息
        :return:
        '''
        ntk_alpha = self.get_ntk_alpha(max_seq_len)
        self.base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
        self.mscale = self.get_mscaling(max_seq_len / self.traning_seq_len)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))
        print(f"inv_freq is {inv_freq}")
        seq = torch.arange(max_seq_len)
        emb = torch.outer(seq, inv_freq)
        emb = torch.cat([emb, emb], dim=-1)

        return torch.cos(self.mscale * emb), torch.sin(self.mscale * emb)
