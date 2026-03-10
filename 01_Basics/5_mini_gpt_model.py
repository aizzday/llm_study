import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(MiniGPT, self).__init__()
        self.embed_dim = embed_dim

        # 1. 词嵌入 和 位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # 2. Transformer Encoder 层
        # PyTorch 的 TransformerEncoderLayer 已经包含了多头注意力和前馈网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 最后的线性输出头
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # 生成位置索引 (0, 1, 2, ...)
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)

        # 计算词嵌入和位置嵌入并相加
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        # 手动生成因果掩码 (causal mask)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # 通过 Transformer 块 (同时传入 mask 和 is_causal=True)
        # is_causal=True 是新版 PyTorch 的推荐做法，它会自动创建掩码
        # 手动传入 mask 是为了兼容旧版或解决特定环境下的 RuntimeError
        output = self.transformer_encoder(x, mask=mask, is_causal=True)

        # 通过输出头得到最终的 logits
        logits = self.lm_head(output)
        return logits

# --- 模拟预测 ---

# 模型参数
vocab_size = 1000  # 假设词表大小为 1000
embed_dim = 16
num_heads = 2
num_layers = 4
max_seq_len = 100 # 模型能处理的最大序列长度

# 创建模型
model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)

# 创建一个模拟输入 (batch_size=1, seq_len=5)
input_ids = torch.randint(0, vocab_size, (1, 5))

# 前向传播
output_logits = model(input_ids)

# 打印输入和输出形状
print(f"输入ID Shape: {input_ids.shape}")
print(f"模型输出 (Logits) Shape: {output_logits.shape}\n")

# --- 直观解释：为什么输出的最后一个维度必须等于词表的大小？ ---
print("--- 为什么输出的最后一个维度必须等于词表的大小？ ---")
print("GPT这类语言模型的核心任务是'预测下一个词'。")
print("1. **预测的本质**: 对于输入序列中的每一个位置，模型都需要计算出词表中所有单词在'下一个'位置出现的可能性。例如，输入'hello world'，在'world'这个位置，模型需要预测后面可能出现的词。")
print("2. **概率分布**: 这个'可能性'是通过一个概率分布来表示的。我们需要为词表中的每一个单词都分配一个分数（logit），分数越高的单词，出现的可能性就越大。")
print("3. **输出与词表的映射**: 因此，模型在每个位置的输出向量，其维度必须正好等于词表的大小（vocab_size）。向量中的第 i 个元素，就代表词表中第 i 个单词的分数。")
print("4. **从 Logits 到概率**: 这些分数（logits）经过 Softmax 函数处理后，就可以转换成一个和为 1 的概率分布，从而可以清晰地看出模型认为哪个词最有可能作为下一个词出现。")
print(f"所以，输出形状 [1, 5, {vocab_size}] 的含义是：对于一个批次（1）中的一句话（5个词），在每一个词的位置上，都预测了一个包含 {vocab_size} 个分数的向量，代表了对下一个词的预测分布。")
