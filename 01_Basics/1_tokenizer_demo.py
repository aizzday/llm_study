import torch

# 1. 定义一个长字符串作为训练语料
text = "hello world, this is a long string for tokenization. this is a simple example."

# 2. 提取所有不重复字符建立词表
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 实现字符到 ID 的双向映射字典
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# 3. 编写 encode 和 decode 函数
def encode(s):
  """将字符串编码为整数列表"""
  return [stoi[c] for c in s]

def decode(l):
  """将整数列表解码为字符串"""
  return "".join([itos[i] for i in l])

# 4. 将编码后的数字列表转换成 PyTorch 的 Tensor 格式并打印
encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.long)

print("Vocabulary:", "".join(chars))
print("Vocabulary size:", vocab_size)
print("Encoded text:", encoded_text)
print("Decoded text:", decode(encoded_text))
print("PyTorch Tensor:", data)
print("Tensor shape:", data.shape)
print("Tensor dtype:", data.dtype)

# 5. 增加 PyTorch 的 Embedding 操作
embedding_dim = 16
embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
embedded_data = embedding_layer(data)

print("\n--- Embedding 操作 ---")
print("Embedding-Shape:", embedded_data.shape)
print("第一个词的向量:", embedded_data[0])

# 为什么输出的维度是二维的？
# 输入 data 的 shape 是 (79)，这是一个一维张量，代表了 79 个 token 的 ID 序列。
# Embedding 层的作用是将每个 token ID 映射到一个固定维度的向量（这里是 16 维）。
# 因此，对于序列中的每个 token，都会输出一个 16 维的向量。
# 最终，输出的 shape 就是 (序列长度, embedding维度)，即 (79, 16)，所以是二维的。

# 6. 增加位置编码 (Positional Encoding)
seq_len = data.shape[0]  # 获取输入 Tensor 的序列长度
max_seq_len = 100  # 假设模型能处理的最大序列长度

# 创建位置编码层
pos_embedding_layer = torch.nn.Embedding(max_seq_len, embedding_dim)

# 生成位置索引 (0, 1, 2, ..., seq_len-1)
position_indices = torch.arange(seq_len)
position_embeddings = pos_embedding_layer(position_indices)

# 将词向量与位置向量相加
combined_embedding = embedded_data + position_embeddings

print("\n--- 位置编码操作 ---")
print("最终 Combined-Shape:", combined_embedding.shape)

# 为什么是相加而不是相乘？
# 在经典的 Transformer 模型中，位置编码是通过加法融入词向量的。
# 1.  **保留信息**: 加法允许位置信息作为一种“偏移”或“调整”被添加到词向量中，而不会破坏词向量本身携带的语义信息。如果使用乘法，可能会导致原始语义信息的丢失或过度扭曲，特别是当位置编码的某些值为 0 时。
# 2.  **线性组合**: 加法是一种简单的线性操作，使得模型在反向传播时更容易学习和优化。每个维度的梯度可以独立计算。
# 3.  **类比**: 可以将其类比为给每个词的向量表示增加一个独特的、基于其位置的“偏置项”，帮助模型区分相同词在不同位置的含义。
