import torch
import torch.nn as nn
import math

from MySelfAttention import MultiHeadAttention

class MyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["max_seq_length"], cfg["embedding_dim"])
        self.transformer_blocks = nn.Sequential(
            *[MyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        #self.layer_norm = MyLayerNorm()
        self.layer_norm = nn.LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
        self.drop = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        #x它是一个矩阵，每一行是段训练数据（也就是一句话）
        #x不是文字，而是文字所对应的token ID 串
        #所以，x中包括了多行训练数据，称为一个批量
        #它的列表示，每一段训练数据的长度
        batch_size, seq_len = x.shape

        #1. batch_size; 2. seq_len; 3. embedding_dim
        token_embeds = self.token_embedding(x) #token_embeds 是一个三维的矩阵

        #position_embedding结果是一个二维矩阵
        #每一行表示arange生成的字符
        #而每一行的列数是由embedding_dim决定的，GPT-2中是768
        postion_embeds = self.position_embedding(torch.arange(seq_len, device=x.device))

        #广播机制（batch_size, seq_len, embedding_dim), (batch_size, seq_len, embedding_dim)
        x = token_embeds + postion_embeds

        #防止过拟合
        x = self.drop(x)

        #(batch_size, seq_len, embedding_dim)
        x = self.transformer_blocks(x)

        x = self.layer_norm(x)

        logits = self.out_head(x)

        return logits

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dim"], 4*cfg["embedding_dim"]),
            NewGELU(),
            #nn.GELU(),
            nn.Linear(4*cfg["embedding_dim"], cfg["embedding_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)

class MyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # MY_GPT_CONFIG = {
        #   "vocab_size": 50257,    #词汇表大小
        #   "max_seq_length": 1024, #每一句训练数据的最大长度
        #   "embedding_dim": 768,   #嵌入向量的维度
        #   "n_heads": 12,          #注意力头个数
        #   "n_layers": 12,         #Transformer 层数
        #   "drop_rate": 0.1,       #Dropout rate
        #   "qkv_bias": False       #bias
        # }
        self.mha = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
            num_heads=cfg["n_heads"],
            drop_rate=cfg["drop_rate"],
            mask_matrix_len=cfg["max_seq_length"],
            qkv_bias=cfg["qkv_bias"])
        self.ffn = FeedForwardNetwork(cfg)
        self.norm_1 = nn.LayerNorm(cfg["embedding_dim"])
        self.norm_2 = nn.LayerNorm(cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        old_x = x
        x = self.norm_1(x)
        x = self.mha(x)
        x = self.dropout(x)
        #残差
        x = x + old_x
        old_x = x #为后面的残差做准备
        x = self.norm_2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + old_x
        return x
    
def text_to_tokenids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_new = torch.tensor(encoded).unsqueeze(0) #在新生成的tensor的最前面增加一个新的维度（batch_size）
    return encoded_new

def tokenids_to_text(tokenids, tokenizer):
    token_new = tokenids.squeeze(0) #它会将tokenids中最前面一个维度为1的去掉, [[1,2,3]]=>[1,2,3]
    return tokenizer.decode(token_new.tolist())

def generate_new(model, prompt, max_new_tokens, context_seq_size,
                 top_k=None, temperature=0.0, eos_id=None):
    for _ in range(max_new_tokens):
        #1024, 1025,我们应该从大小为1025这个文本的后面开如算，取1024个
        #prompt(batch, tokens)
        #[[1, 2, 3], [5678,9967,3344]]
        prompt_slice = prompt[:, -context_seq_size:]
        with torch.no_grad():
            logits = model(prompt_slice)
        #logist(batch_size, new_token, vocab_size)
        logits = logits[:, -1, :]

        if top_k is not None:
            #logits现在已经是一个二维的矩阵，（batch_size, vocab_size)
            top_logits, _ = torch.topk(logits, top_k)
            min_k = top_logits[:, -1]
            logits = torch.where(logits<min_k, 
                        torch.tensor(float("-inf")).to(logits.device), 
                        logits)
        if temperature >0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            new_token = torch.multinomial(probas, num_samples=1)
        else:
            probas = torch.softmax(logits, dim=-1)
            new_token = torch.argmax(probas, dim=-1, keepdim=True)

        if new_token == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        prompt = torch.cat((prompt, new_token), dim=1)
    return prompt

def generate_text(model, prompt, max_new_tokens, context_seq_size):
    for _ in range(max_new_tokens):
        #1024, 1025,我们应该从大小为1025这个文本的后面开如算，取1024个
        #prompt(batch, tokens)
        #[[1, 2, 3], [5678,9967,3344]]
        prompt_slice = prompt[:, -context_seq_size:]
        with torch.no_grad():
            logits = model(prompt_slice)
        #logist(batch_size, new_token, vocab_size)
        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)
        new_token = torch.argmax(probas, dim=-1, keepdim=True)
        prompt = torch.cat((prompt, new_token), dim=1)

    return prompt

def calc_loss_batch(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten())
    return loss

def calc_loss(loader, model, device):
    total_loss = 0
    num_batch = len(loader)
    for inputs, targets in loader:
        loss = calc_loss_batch(inputs, targets, model, device)
        total_loss += loss
    return total_loss / num_batch

def print_train_info(model, tokenizer, device, prompt):
    model.eval()
    encoded = text_to_tokenids(prompt, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model, encoded, 50, 1024)
        decoded_text = tokenids_to_text(token_ids, tokenizer)
        print("new text:", decoded_text.replace("\n", " "))
    model.train()
    return

def eval_model(model, train_loader, eval_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss(train_loader, model, device)
        eval_loss = calc_loss(eval_loader, model, device)
    model.train()
    return train_loss, eval_loss

def train_model(model, train_loader, eval_loader, optimizer, device,
                epochs, tokenizer, prompt, eval_interval):
    batch_step = 0
    train_losses, eval_losses = [], []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step()

            batch_step +=1

            if batch_step % eval_interval ==0:
                train_loss, eval_loss = eval_model(model, train_loader, eval_loader, device)
                train_losses.append(train_loss)
                eval_losses.append(eval_loss)
                print(f"Epoch{epoch+1} step:{batch_step:10d}: "
                      f"Train loss {train_loss:.3f}, Eval loss{ eval_loss:.3f}")
            
        print_train_info(model, tokenizer, device, prompt)
    return train_losses, eval_losses