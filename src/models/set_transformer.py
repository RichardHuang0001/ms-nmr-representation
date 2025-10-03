## set_transformer.py
# -*- coding: utf-8 -*-
"""
我们的核心模型架构。
该文件上半部分直接整合了Set Transformer官方实现的核心模块，
下半部分是我们为自监督预训练任务设计的模型。
此文件不再依赖任何外部的 'set-transformer' 第三方库。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================================================================
# Part 1: Official Set Transformer Core Modules
# Source: https://github.com/juho-lee/set_transformer
# 作者: Juho Lee et al.
# 我们直接将官方实现的核心组件整合到项目中，以确保稳定性和正确性。
# ==========================================================================================

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None): # <--- [修正1] 增加 mask 参数
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        # --- [修正2] 应用注意力掩码 ---
        # 计算注意力分数
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if mask is not None:
            # 将mask广播到多头注意力需要的形状
            # 原始mask: [B, L] -> [B, 1, L] -> [B*H, 1, L] -> [B*H, L_q, L_k]
            # 这里Q和K的序列长度相同，简化处理
            mask_ = mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1)
            A = A.masked_fill(mask_ == 0, -1e9) # 用一个极大的负数填充mask位置

        A = torch.softmax(A, 2)
        # --------------------------------

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

# --------------------------------
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None): # <--- [修正3] 增加 mask 参数
        return self.mab(X, X, mask=mask)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_in))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_out, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# ==========================================================================================
# Part 2: Our Pre-training Model Implementation
# 这个模型现在直接调用上面定义的官方核心模块。
# ==========================================================================================

class PretrainSetTransformer(nn.Module):
    """
    用于自监督预训练的Set Transformer模型。
    该模型现在是自包含的，不依赖任何外部set-transformer库。
    """
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_hidden: int,
                 num_heads: int,
                 num_inds: int = 32,
                 depth: int = 2,
                 ln: bool = True):
        """
        初始化预训练模型。
        :param dim_input: 输入的Peak Vector维度 (我们设计的是12)。
        :param dim_output: 输出的Peak Vector维度 (同样是12)。
        :param dim_hidden: Set Transformer内部的隐藏维度。
        :param num_heads: 多头注意力机制的头数。
        :param num_inds: 诱导点数量。
        :param depth: Set Transformer编码器的层数。
        :param ln: 是否使用LayerNorm。
        """
        super().__init__()
        
        # 1. 定义一个可学习的 [MASK] 向量
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim_input))

        # 2. 实例化Set Transformer核心作为编码器
        # 我们使用上面定义的官方模块来构建编码器
        encoder_layers = []
        for _ in range(depth):
            encoder_layers.append(
                SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln)
            )
        
        self.encoder_input_proj = nn.Linear(dim_input, dim_hidden)
        self.encoder = nn.Sequential(*encoder_layers)

        # 3. 定义重建头部 (Reconstruction Head)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden * 2),
            nn.GELU(),
            nn.Linear(dim_hidden * 2, dim_output)
        )
        
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        模型的前向传播。
        :param input_tensor: 形状为 [B, L, D] 的输入张量，被掩码的峰已用全零向量填充。
        :param attention_mask: 形状为 [B, L] 的注意力掩码。
        :return: 形状为 [B, L, D] 的预测峰向量。
        """
        # 1. 动态替换 [MASK] token
        is_masked = torch.all(input_tensor == 0, dim=-1)
        if attention_mask is not None:
            is_masked = is_masked & attention_mask.bool()
        
        masked_input = torch.where(
            is_masked.unsqueeze(-1), 
            self.mask_token.expand(input_tensor.shape[0], input_tensor.shape[1], -1), 
            input_tensor
        )
        
        # 将输入投影到隐藏维度
        x = self.encoder_input_proj(masked_input)

        # --- [修正4] 向编码器传递 attention_mask ---
        # 遍历Sequential中的SAB层并传递mask
        encoded_representation = x
        for layer in self.encoder:
            encoded_representation = layer(encoded_representation, mask=attention_mask)
        
        predictions = self.reconstruction_head(encoded_representation)
        
        return predictions


# --- 用于演示和调试的示例代码 ---
if __name__ == "__main__":
    
    # --- 模型超参数 ---
    BATCH_SIZE = 4
    MAX_PEAKS = 512
    PEAK_DIM = 12
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    
    # --- 实例化模型 ---
    model = PretrainSetTransformer(
        dim_input=PEAK_DIM,
        dim_output=PEAK_DIM,
        dim_hidden=HIDDEN_DIM,
        num_heads=NUM_HEADS
    )
    
    print("--- 模型结构 ---")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {num_params / 1e6:.2f} M")

    # --- 创建一些虚拟的输入数据 ---
    input_data = torch.randn(BATCH_SIZE, MAX_PEAKS, PEAK_DIM)
    
    real_peaks_count = [100, 250, 400, 50]
    attention_mask = torch.zeros(BATCH_SIZE, MAX_PEAKS)
    for i, count in enumerate(real_peaks_count):
        attention_mask[i, :count] = 1
        input_data[i, count:] = 0

    for i in range(BATCH_SIZE):
        num_to_mask = int(real_peaks_count[i] * 0.15)
        mask_indices = torch.randperm(real_peaks_count[i])[:num_to_mask]
        input_data[i, mask_indices] = 0

    print("\n--- 输入张量形状检查 ---")
    print(f"输入数据 (input_tensor) 形状: {input_data.shape}")
    print(f"注意力掩码 (attention_mask) 形状: {attention_mask.shape}")
    
    model.eval()
    with torch.no_grad():
        predictions = model(input_data, attention_mask)

    print("\n--- 输出张量形状检查 ---")
    print(f"模型预测 (predictions) 形状: {predictions.shape}")

    # 检查输出维度是否正确
    assert predictions.shape == (BATCH_SIZE, MAX_PEAKS, PEAK_DIM), "输出维度不匹配！"
    print("\n✅ 输出维度正确！")

