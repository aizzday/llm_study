import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. 输入数据
# 形状: (batch_size, seq_len, embedding_dim)
batch_size = 1
seq_len = 5
embedding_dim = 16
x = torch.randn(batch_size, seq_len, embedding_dim)
print(f"输入 (x) Shape: {x.shape}\n")

# 2. 手动创建 Q, K, V 矩阵 (Linear层)
q_linear = nn.Linear(embedding_dim, embedding_dim)
k_linear = nn.Linear(embedding_dim, embedding_dim)
v_linear = nn.Linear(embedding_dim, embedding_dim)

Q = q_linear(x)
K = k_linear(x)
V = v_linear(x)
print(f"Q, K, V 的 Shape (均为): {Q.shape}\n")

# 3. 分步骤计算 Self-Attention

# 步骤 1: 计算 Q 和 K 的点积 (Dot Product)
# Q shape: [1, 5, 16], K.transpose(-2, -1) shape: [1, 16, 5]
scores = torch.matmul(Q, K.transpose(-2, -1))
print(f"步骤 1: 点积分数 (Scores) Shape: {scores.shape}")

# 步骤 2: 除以根号下的维度 (Scaling)
# 这是为了在反向传播时获得更稳定的梯度
d_k = K.size(-1)
scaled_scores = scores / math.sqrt(d_k)
print(f"步骤 2: 缩放后分数 (Scaled Scores) Shape: {scaled_scores.shape}")

# 步骤 3: 做 Softmax 归一化
# 在最后一个维度上进行 softmax，使得每一行的和为 1
attention_weights = F.softmax(scaled_scores, dim=-1)
print(f"步骤 3: 注意力权重 (Attention Weights) Shape: {attention_weights.shape}\n")

# 步骤 4: 用权重乘以 V
output = torch.matmul(attention_weights, V)
print(f"步骤 4: 最终输出 (Output) Shape: {output.shape}\n")


# 重点：打印出最后的 5x5 注意力权重矩阵
print("--- 注意力权重矩阵 (5x5) ---")
# 由于 batch_size 是 1，我们直接打印第一个 (也是唯一一个) 矩阵
print(attention_weights.squeeze(0))
print("-" * 30)

# 解释为什么这个矩阵的每一行加起来必须是 1
print("\n--- 为什么每一行加起来是 1？ ---")
print("这个特性是由 Softmax 函数决定的。")
print("1. Softmax 的作用: Softmax 函数接收一个向量作为输入，并输出一个概率分布。这意味着输出向量中的所有元素都是非负的，并且它们的总和为 1。")
print("2. 应用于注意力: 在 Self-Attention 中，我们对 'scaled_scores' 张量的最后一个维度（dim=-1）应用 Softmax。这个维度对应着序列中的每一个查询（Query）词与所有键（Key）词之间的相关性分数。")
print("3. 逐行归一化: 'dim=-1' 意味着 Softmax 是独立地应用在每个 5x5 矩阵的每一行上的。对于第一行，它将第一个词与所有五个词的相关性分数转换为一个概率分布。对于第二行，它将第二个词与所有五个词的相关性分数转换为概率分布，以此类推。")
print("4. 结果: 因此，每一行都代表了一个独立的概率分布，表示一个词应该'关注'其他所有词（包括它自己）的程度。作为一个概率分布，该行所有元素的总和自然必须为 1。")

# 验证：打印每一行的和
print("\n验证：每一行的和:")
print(attention_weights.sum(dim=-1))
print("-" * 30)
