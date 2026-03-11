'''
使用 Hugging Face GPT-2 模型生成文本
'''

# 确保已安装 transformers 和 torch
# pip install transformers torch

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# --- 1. 加载预训练模型和分词器 ---

# 指定模型名称
model_name = "gpt2"

print(f"正在加载模型: {model_name}...")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载 GPT-2 语言模型
# GPT2LMHeadModel 是一个标准的 Transformer 模型，其顶部带有一个语言模型头部（LM Head）。
# LM Head 是一个线性层，其输出维度等于词表大小，用于预测下一个 token。
model = GPT2LMHeadModel.from_pretrained(model_name)
print("模型加载完成。")

# --- 2. 尝试“提示工程 (Prompt Engineering)” ---

# 提示工程是指通过精心设计输入文本（Prompt），来引导模型产生我们想要的输出。
# 即使是同一个模型，不同的 Prompt 也会导致结果天差地别。

def generate_answer(prompt_text, max_new_tokens=10):
    """一个辅助函数，用于编码、生成和解码。"""
    print(f'\n--- Prompt: "{prompt_text}" ---')
    
    # 编码
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
    # 生成
    # 注意：我们将 max_length 调整为输入长度 + 新 token 数量
    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + max_new_tokens,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码并打印
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"模型回答: {decoded_output}")

# --- 2.1 尝试“少样本提示 (Few-shot Prompting)” ---
# 这是一种更高级的提示工程技巧。我们不直接提问，而是给模型几个完整的“输入->输出”范例，
# 然后给出一个不完整的范例，让模型根据学习到的模式来“补全”。
# 这种方式能更清晰地告诉模型我们想要它扮演的角色和执行的任务。

prompt_few_shot = "English: Apple, French: Pomme; English: Dog, French: Chien; English: Hello, French:"
generate_answer(prompt_few_shot, max_new_tokens=3) # 只需要生成少量 token 即可


# 结论：
# 通过提供 "Apple -> Pomme" 和 "Dog -> Chien" 这两个例子，
# 我们为模型设置了一个清晰的“英译法”模式。
# 当模型看到 "English: Hello, French:" 时，它会意识到需要遵循这个模式，
# 从而更有可能生成正确的 "Bonjour"。
# 这种方法比直接问 "Translate 'Hello' to French" 更有效，因为它为模型提供了上下文和任务范例，
# 极大地降低了任务的模糊性。
