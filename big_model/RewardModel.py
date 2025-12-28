import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List, Optional


class RewardModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", use_margin_loss: bool = False):
        super().__init__()
        # 1. 加载预训练模型作为Backbone（共享权重）
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # 2. 添加Reward Head（输出标量分数）
        self.reward_head = nn.Linear(hidden_size, 1)
        self.use_margin_loss = use_margin_loss

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Backbone提取特征
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # 取[EOS]位置的特征（或均值池化）
        eos_embedding = last_hidden_state[:, -1, :]  # 使用序列末尾token
        # 或：mean_embedding = last_hidden_state.mean(dim=1)

        # 计算奖励分数
        reward = self.reward_head(eos_embedding).squeeze(-1)  # [batch]
        return reward


def pairwise_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor,
                  margins: Optional[torch.Tensor] = None) -> torch.Tensor:
    """成对损失函数（支持标准损失和Margin Loss）"""
    diff = chosen_rewards - rejected_rewards
    if margins is not None:  # Margin Loss变体[3](@ref)
        diff = diff - margins
    return -torch.log(torch.sigmoid(diff)).mean()


class PreferenceDataset(Dataset):
    """加载成对偏好数据格式: (prompt, chosen_response, rejected_response)"""

    def __init__(self, data: List[Tuple[str, str, str]], tokenizer: AutoTokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[dict, dict]:
        prompt, chosen, rejected = self.data[idx]

        # 编码chosen序列: [CLS] + prompt + [SEP] + chosen + [SEP]
        chosen_enc = self.tokenizer(
            prompt, chosen,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 编码rejected序列
        rejected_enc = self.tokenizer(
            prompt, rejected,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return chosen_enc, rejected_enc


# 训练超参数
model_name = r""
batch_size = 8
grad_accum_steps = 2  # 梯度累积步数（节省显存）
use_amp = True  # 混合精度训练
use_margin_loss = True  # 是否启用Margin Loss

# 初始化模型、Tokenizer、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RewardModel(model_name, use_margin_loss).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

# 示例数据（实际需替换为真实偏好数据）
train_data = [
    ("解释强化学习", "强化学习是智能体通过奖励信号学习决策的方法。", "强化学习是一种算法。"),
    ("牛顿第三定律是什么？", "作用力与反作用力大小相等、方向相反。", "牛顿定律涉及万有引力。"),
]
train_dataset = PreferenceDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练循环
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
model.train()
for epoch in range(10):
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        chosen_batch, rejected_batch = batch

        # 将数据移至设备
        chosen_inputs = {k: v.squeeze(1).to(device) for k, v in chosen_batch.items()}
        rejected_inputs = {k: v.squeeze(1).to(device) for k, v in rejected_batch.items()}

        # 混合精度前向
        with torch.cuda.amp.autocast(enabled=use_amp):
            # 计算chosen和rejected的奖励分数
            r_chosen = model(**chosen_inputs)
            r_rejected = model(**rejected_inputs)

            # 若用Margin Loss，需从数据中加载margin值（此处示例随机生成）
            margins = torch.rand(len(r_chosen)).to(device) if use_margin_loss else None
            loss = pairwise_loss(r_chosen, r_rejected, margins)

        # 梯度累积
        scaler.scale(loss).backward()
        total_loss += loss.item()

        # 累积梯度后更新
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "reward_model.pt")
