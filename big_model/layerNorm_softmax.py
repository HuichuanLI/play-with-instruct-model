import torch
from torch import nn


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        自定义LayerNorm实现

        参数:
        - normalized_shape: 归一化的特征维度
        - eps: 数值稳定性小常数，防止除以零
        - elementwise_affine: 是否使用可学习的缩放(γ)和平移(β)参数
        """
        super().__init__()

        # 检查normalized_shape是int还是tuple
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # 可学习的缩放参数γ，初始化为1
            self.gamma = nn.Parameter(torch.ones(*self.normalized_shape))
            # 可学习的平移参数β，初始化为0
            self.beta = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            # 不使用可学习参数
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        # 有偏估计 (分母n)
        # 方差计算公式：
        # 有偏公式 (1/n)((x1-mean)**2+(x2-mean)**2+...+(xn-mean)**2)
        # 无偏公式 (1/(n-1))((x1-mean)**2+(x2-mean)**2+...+(xn-mean)**2)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # 归一化公式
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # 应用可学习的参数
        if self.elementwise_affine:
            output = self.gamma * x_normalized + self.beta
        else:
            output = x_normalized

        return output


# 创建输入数据: (batch_size, seq_len, features)
batch_size, seq_len, features = 16, 10, 512
x = torch.randn(batch_size, seq_len, features)

print("输入形状:", x.shape)
print("输出形状:", x.shape)
print("输出均值:", x.mean(dim=-1))
print("输出方差:", x.var(dim=-1, unbiased=False))
# 实例化自定义层归一化模块
custom_norm = CustomLayerNorm(features)

# 前向传播
output = custom_norm(x)

print("输入形状:", x.shape)
print("输出形状:", output.shape)
print("输出均值:", output.mean(dim=-1))
print("输出方差:", output.var(dim=-1, unbiased=False))


## softmax 实现
# 多分类网络输出层示例
def softmax_torch(x, dim=-1):
    """PyTorch实现的softmax函数

    参数:
        x - 输入张量
        dim - 计算softmax的维度

    返回:
        softmax激活后的张量
    """
    # 数值稳定性处理
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    e_x = torch.exp(x - max_vals)

    return e_x / torch.sum(e_x, dim=dim, keepdim=True)


seq_len = 10
x = torch.randn(seq_len)

result = softmax_torch(x)
print(f"result is {result}")
print(f"sum of result is {sum(result)}")


def softmax_temperature(x, temperature=1.0, dim=-1):
    """带有温度参数的softmax

    温度参数作用:
        temperature > 1.0: 平滑分布 (增加熵)
        temperature < 1.0: 锐化分布 (降低熵)
    """
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)


result = softmax_temperature(x, 2)
print(f"result is {result}")
print(f"sum of result is {sum(result)}")

result = softmax_temperature(x, 0.2)
print(f"result is {result}")
print(f"sum of result is {sum(result)}")


def batched_softmax(x):
    """处理批量输入的softmax

    输入形状: (batch_size, num_classes)
    输出形状: (batch_size, num_classes)
    """
    max_vals, _ = torch.max(x, dim=1, keepdim=True)
    e_x = torch.exp(x - max_vals)
    return e_x / torch.sum(e_x, dim=1, keepdim=True)


batch_size, seq_len = 16, 10
x = torch.randn(batch_size, seq_len)

result = batched_softmax(x)
print(f"result is {result}")
print(f"sum of result is {torch.sum(result, -1)}")
