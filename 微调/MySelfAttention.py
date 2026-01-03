import torch
import torch.nn as nn

#W_q, W_k, Q_v
#Q,K,V
#Q = x @ W_q, 其中x的列的维度是embedding_dim，Query的维度呢？自己设置的，64
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))
        self.d_out = d_out

    def forward(self, x):
        #获得q,k,v
        #@表示的就是调用matmul(), 它是一种缩写
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        attn_score = q @ k.T
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        context_vec = attn_weight @ v
        return context_vec

class MySelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attn_score = q @ k.T
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        context_vec = attn_weight @ v
        return context_vec  
    
class MaskSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, mask_matrix_len, drop_rate, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        self.dropout= nn.Dropout(drop_rate)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(mask_matrix_len, mask_matrix_len), diagonal=1)
        )
    
    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        #之前由于输入是二维，所以可以直接使用k.T进行转置
        #现在k是一个三维数据，此时不能直接使用T进行转置了，需要调用transpose交换指定的维度
        attn_score = q @ k.transpose(1,2)
        #masked_fill作用：根据遮掩矩阵，替换相应的值，返回的是一个处理后的新矩阵
        #masked_fill_作用：根据遮掩矩阵，替换相应的值，但它是对原数据进行修改
        attn_score.masked_fill_(self.mask.bool()[:seq_length, :seq_length], -torch.inf)
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        attn_weight = self.dropout(attn_weight) # 增加泛化性
        context_vec = attn_weight @ v
        return context_vec  

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 mask_matrix_len, drop_rate, num_heads, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        self.dropout= nn.Dropout(drop_rate)
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.output = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(mask_matrix_len, mask_matrix_len), diagonal=1)
        )
    
    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        #之前由于输入是二维，所以可以直接使用k.T进行转置
        #现在k是一个三维数据，此时不能直接使用T进行转置了，需要调用transpose交换指定的维度
        attn_score = q @ k.transpose(2,3)
        #masked_fill作用：根据遮掩矩阵，替换相应的值，返回的是一个处理后的新矩阵
        #masked_fill_作用：根据遮掩矩阵，替换相应的值，但它是对原数据进行修改
        attn_score.masked_fill_(self.mask.bool()[:seq_length, :seq_length], -torch.inf)
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        attn_weight = self.dropout(attn_weight) # 增加泛化性
        context_vec = (attn_weight @ v).transpose(1,2)
        context_vec = context_vec.contiguous().view(batch_size, seq_length, self.d_out)
        context_vec = self.output(context_vec)
        return context_vec  

#对于自注意力机制来说，它的输出（seq_length, d_out)
#上节课中，我们知道输入给transformer_block,它的输入是三维的（batch_size, seq_length, embedding_dim)
#最终的输入是二维的（seq_length, embedding_dim）
#cfg[]
d_in = 768
d_out = 64
torch.manual_seed(123)
attn = SelfAttention_v1(d_in, d_out)
#inputs = [[0.1, -0.32 ],[0.06, -0.234]]
#attn(inputs)