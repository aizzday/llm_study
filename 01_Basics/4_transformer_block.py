import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Multi-Head Attention
        attn_output, _ = self.attention(x, x, x)

        # 2. 残差连接 和 LayerNorm
        x = self.norm1(x + self.dropout(attn_output))

        # 3. Feed Forward
        ff_output = self.feed_forward(x)

        # 4. 另一个残差连接 和 LayerNorm
        x = self.norm2(x + self.dropout(ff_output))

        return x

# --- 演示 ---

# 参数设置
embed_dim = 16
num_heads = 2
ff_dim = 64  # FeedForward 层的中间维度，通常是 embed_dim 的 4 倍
seq_len = 5
batch_size = 1

# 创建模型
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

# 创建一个随机输入张量
x = torch.randn(batch_size, seq_len, embed_dim)

# 前向传播
output = transformer_block(x)

# 打印输出形状
print(f"输入 Shape: {x.shape}")
print(f"输出 Shape: {output.shape}\n")

# --- 为什么 Transformer 需要这么多重复的 Block 堆叠在一起？ ---
print("--- 为什么 Transformer 需要这么多重复的 Block 堆叠在一起？ ---")
print("将多个 Transformer Block 堆叠在一起，是为了构建一个'深度'模型，这对于学习复杂的数据模式至关重要。原因如下：")
print("1. **逐层抽象特征**: 就像深度卷积网络（如 ResNet）通过堆叠卷积层来从像素中提取越来越抽象的特征（从边缘到纹理再到物体部分）一样，堆叠 Transformer Block 也能让模型学习到更高级的语言特征。")
print("  - **底层 Block**: 可能主要关注局部的、简单的语法关系。")
print("  - **中层 Block**: 可能会开始理解更复杂的短语结构或语义组合。")
print("  - **高层 Block**: 则可能能够捕捉到长距离的依赖关系、上下文逻辑甚至语篇结构。")
print("2. **扩大感受野**: 每经过一个 Block，信息都会在序列中的所有位置之间进行一次交互（通过自注意力）。堆叠多个 Block 意味着信息可以进行多轮的、更复杂的传播和混合。一个词的表示不仅能直接吸收到其他词的信息，还能间接吸收到'其他词吸收到的信息'，从而极大地扩展了每个词的'上下文感受野'。")
print("3. **增强模型容量**: 更多的 Block 意味着更多的参数和更强的非线性能力，这使得模型能够拟合更复杂的数据分布，从而在大型、多样化的数据集上取得更好的性能。残差连接和层归一化是确保这种深度结构能够被有效训练的关键技术。")
