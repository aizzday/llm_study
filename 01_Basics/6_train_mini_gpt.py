import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. 数据准备 ---

# 定义训练语料
text = 'The quick brown fox jumps over the lazy dog'

# 创建字符级 Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

# 将语料转换为 Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# --- 2. 定义 Mini-GPT 模型 ---
# (为了方便，我们将之前的 MiniGPT 类直接复制到这里)
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(MiniGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        output = self.transformer_encoder(x, mask=mask, is_causal=True)
        logits = self.lm_head(output)
        return logits

# --- 3. 训练设置 ---

# 模型参数
embed_dim = 16
num_heads = 2
num_layers = 4
max_seq_len = len(text)

# 实例化模型
model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# --- 4. 训练循环 ---

print("--- 开始训练 ---")
for i in range(50):
    # 准备输入 (x) 和目标 (y)
    # x 是除了最后一个字符的所有字符
    # y 是除了第一个字符的所有字符 (x 的每个字符对应的下一个字符)
    inputs = data[:-1].unsqueeze(0)  # 增加 batch 维度
    targets = data[1:].unsqueeze(0) # 增加 batch 维度

    # 梯度清零
    optimizer.zero_grad()

    # 前向传播
    logits = model(inputs)

    # 计算损失
    # CrossEntropyLoss 要求 logits 是 (N, C) 格式，targets 是 (N)
    # 所以我们需要调整一下形状
    B, T, C = logits.shape
    loss = loss_fn(logits.view(B*T, C), targets.view(B*T))

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 每 10 次打印一次损失
    if i % 10 == 0:
        print(f"第 {i} 次循环, Loss: {loss.item()}")

print("--- 训练完成 ---\n")

# --- 5. 验证 ---

# 准备第一个词作为输入
first_char_encoded = data[0].unsqueeze(0).unsqueeze(0) # Shape: [1, 1]

# 模型预测
model.eval() # 切换到评估模式
with torch.no_grad():
    logits = model(first_char_encoded)

# 获取最后一个时间步的 logits，并找到概率最高的词
next_char_logit = logits[0, -1, :]
predicted_index = torch.argmax(next_char_logit).item()
predicted_char = decode([predicted_index])

print(f"输入: '{decode([data[0].item()])}'")
print(f"模型预测的下一个字符: '{predicted_char}'")
