import torch
from transformers import GPT2LMHeadModel

# ==========================================
# 1. 辅助函数定义 (Refactored Helper Functions)
# ==========================================

def _load_embeddings(my_model, hf_sd):
    """加载词嵌入和位置嵌入"""
    print("  -> 正在加载 Embeddings (wte, wpe)...")
    my_model.token_embedding.weight.copy_(hf_sd["transformer.wte.weight"])
    my_model.position_embedding.weight.copy_(hf_sd["transformer.wpe.weight"])

def _load_attention_layer(my_block, hf_sd, prefix, embed_dim):
    """
    加载 Attention 层权重
    关键逻辑：处理 HuggingFace 的 c_attn (合并的QKV) 和 Conv1D 形状转置
    """
    # 1. 获取 HF 的 QKV 联合权重和偏置
    # HF shape: [768, 2304]
    c_attn_w = hf_sd[f"{prefix}.attn.c_attn.weight"] 
    c_attn_b = hf_sd[f"{prefix}.attn.c_attn.bias"]   

    # 2. 切分 (Split) 成 Q, K, V
    # dim=1 表示在输出维度切分
    q_w, k_w, v_w = c_attn_w.split(embed_dim, dim=1)
    q_b, k_b, v_b = c_attn_b.split(embed_dim, dim=0)

    # 3. 赋值并转置 (.t())
    # MyModel Linear shape: [768, 768] -> (out, in)
    my_block.mha.W_q.weight.copy_(q_w.t())
    my_block.mha.W_q.bias.copy_(q_b)
    
    my_block.mha.W_k.weight.copy_(k_w.t())
    my_block.mha.W_k.bias.copy_(k_b)
    
    my_block.mha.W_v.weight.copy_(v_w.t())
    my_block.mha.W_v.bias.copy_(v_b)

    # 4. 输出投影层 (c_proj -> output)
    c_proj_w = hf_sd[f"{prefix}.attn.c_proj.weight"]
    c_proj_b = hf_sd[f"{prefix}.attn.c_proj.bias"]
    
    my_block.mha.output.weight.copy_(c_proj_w.t()) # 记得转置
    my_block.mha.output.bias.copy_(c_proj_b)

def _load_ffn_layer(my_block, hf_sd, prefix):
    """
    加载 Feed Forward Network (MLP) 权重
    关键逻辑：处理 Conv1D 形状转置
    """
    # 第一层 Linear (升维: c_fc -> layers[0])
    c_fc_w = hf_sd[f"{prefix}.mlp.c_fc.weight"]
    c_fc_b = hf_sd[f"{prefix}.mlp.c_fc.bias"]
    
    my_block.ffn.layers[0].weight.copy_(c_fc_w.t()) # 转置
    my_block.ffn.layers[0].bias.copy_(c_fc_b)
    
    # 第二层 Linear (降维: c_proj -> layers[2])
    # layers[1] 是 GELU，跳过
    c_proj_ffn_w = hf_sd[f"{prefix}.mlp.c_proj.weight"]
    c_proj_ffn_b = hf_sd[f"{prefix}.mlp.c_proj.bias"]
    
    my_block.ffn.layers[2].weight.copy_(c_proj_ffn_w.t()) # 转置
    my_block.ffn.layers[2].bias.copy_(c_proj_ffn_b)

def _load_transformer_block(my_model, hf_sd, layer_idx, embed_dim):
    """
    加载单个 Transformer Block (LayerNorms + Attention + FFN)
    """
    prefix = f"transformer.h.{layer_idx}"
    my_block = my_model.transformer_blocks[layer_idx]

    # 1. LayerNorms
    my_block.norm_1.weight.copy_(hf_sd[f"{prefix}.ln_1.weight"])
    my_block.norm_1.bias.copy_(hf_sd[f"{prefix}.ln_1.bias"])
    
    my_block.norm_2.weight.copy_(hf_sd[f"{prefix}.ln_2.weight"])
    my_block.norm_2.bias.copy_(hf_sd[f"{prefix}.ln_2.bias"])

    # 2. Multi-Head Attention
    _load_attention_layer(my_block, hf_sd, prefix, embed_dim)

    # 3. Feed Forward Network
    _load_ffn_layer(my_block, hf_sd, prefix)

def _load_final_layers(my_model, hf_sd):
    """加载最后的 LayerNorm 和 Output Head"""
    print("  -> 正在加载 Final LayerNorm & Head...")
    
    # Final LayerNorm
    my_model.layer_norm.weight.copy_(hf_sd["transformer.ln_f.weight"])
    my_model.layer_norm.bias.copy_(hf_sd["transformer.ln_f.bias"])

    # Output Head
    if "lm_head.weight" in hf_sd:
        my_model.out_head.weight.copy_(hf_sd["lm_head.weight"])
    else:
        # 权重绑定 (Weight Tying) 情况
        print("     (提示: lm_head 权重与 wte 共享)")
        my_model.out_head.weight.copy_(hf_sd["transformer.wte.weight"])

# ==========================================
# 2. 主加载函数 (Main Function)
# ==========================================

def load_gpt2_weights(my_model, cfg):
    """
    [主函数] 将 Hugging Face 的 GPT-2 权重完美移植到自定义 MyGPTModel 中
    """
    print(f"\n[1/3] 正在从 Hugging Face 下载/加载 gpt2 模型...")
    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    hf_sd = hf_model.state_dict()
    print("[2/3] 开始权重移植...")

    # 使用 no_grad 加速并防止内存浪费
    with torch.no_grad():
        # 1. Embeddings
        _load_embeddings(my_model, hf_sd)

        # 2. Transformer Blocks
        print(f"  -> 正在加载 {cfg['n_layers']} 层 Transformer Block...")
        for i in range(cfg["n_layers"]):
            _load_transformer_block(my_model, hf_sd, i, cfg["embedding_dim"])

        # 3. Final Layers
        _load_final_layers(my_model, hf_sd)

    print("[3/3] 成功！GPT-2 权重已全部加载完成。\n")
    return my_model