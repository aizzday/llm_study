import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性层，用于生成 Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 最终的输出线性层
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # 1. 通过线性层生成 Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # 2. 拆分成多个头
        # 原始 Q, K, V shape: [batch_size, seq_len, embed_dim]
        # 拆分后 shape: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # 4. 拼接多头结果
        # attention_output shape: [batch_size, num_heads, seq_len, head_dim]
        # 拼接后 shape: [batch_size, seq_len, embed_dim]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 5. 通过最终的线性层
        output = self.fc_out(attention_output)
        return output

# --- 演示 ---

# 输入参数
embed_dim = 16
num_heads = 2
seq_len = 5
batch_size = 1

# 创建模型
multi_head_attention = MultiHeadAttention(embed_dim, num_heads)

# 创建一个随机输入张量
x = torch.randn(batch_size, seq_len, embed_dim)

# 前向传播
output = multi_head_attention(x)

# 打印输出形状
print(f"输入 Shape: {x.shape}")
print(f"输出 Shape: {output.shape}\n")

# --- 为什么多头注意力比单头注意力更强大？ (眼镜的比喻) ---
print("--- 为什么多头注意力比单头注意力更强大？ (眼镜的比喻) ---")
print("单头注意力就像只戴一副普通的眼镜去看待一句话，你只能从一个固定的视角去理解词与词之间的关系。")
print("而多头注意力则像是给你配备了多副功能不同的'特效眼镜'：")
print("  - **眼镜A (语法分析镜)**: 可能专门用来发现句子中的主谓宾结构。")
print("  - **眼镜B (同义词辨识镜)**: 可能专门用来寻找具有相似意义的词。")
print("  - **眼镜C (指代关系镜)**: 可能专门用来连接代词和它所指代的名词。")
print("模型可以同时戴上这些眼镜（并行计算），每一个'头'（一副眼镜）都从自己的专业角度去审视句子，并给出一个关注点矩阵。")
print("最后，模型将所有眼镜看到的结果（多个注意力输出）拼接融合，并通过一个输出线性层（fc_out）形成一个综合了所有视角、信息更丰富的最终理解。这就是多头注意力更强大的原因。")
