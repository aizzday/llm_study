'''
演示 Hugging Face Tokenizer (GPT-2)
'''

# 首先，确保你已经安装了 transformers 库
# pip install transformers

from transformers import AutoTokenizer

# 1. 加载预训练的 GPT-2 分词器
# GPT-2 使用的是一种叫做 BPE (Byte-Pair Encoding) 的子词分词算法
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. 准备两个输入字符串
# 一个是常见的词组，另一个是一个罕见的、拼接起来的词
string1 = "Hello world"
string2 = "Tokenizerisawesome"

# 3. 分别处理并打印结果
print(f"--- 正在处理: '{string1}' ---")
# 使用 .tokenize() 方法获取 tokens
tokens1 = tokenizer.tokenize(string1)
# 使用 .encode() 方法获取 ids
ids1 = tokenizer.encode(string1)
print(f"Tokens: {tokens1}")
print(f"Token IDs: {ids1}")
print("-" * 30)

print(f"--- 正在处理: '{string2}' ---")
tokens2 = tokenizer.tokenize(string2)
ids2 = tokenizer.encode(string2)
print(f"Tokens: {tokens2}")
print(f"Token IDs: {ids2}")
print("-" * 30)

# 4. 解释为什么 GPT-2 能处理怪词

# 我们可以看到，对于 "Hello world"，分词器能够轻松地将其拆分为 'Hello' 和 'Ġworld'。
# （前面的 "Ġ" 符号代表这是一个词的开始，通常是空格的替代）

# 但对于 "Tokenizerisawesome"，分词器并没有见过这个词。但它没有报错，而是巧妙地将其拆分成了：
# ['Token', 'izer', 'isa', 'wesome']
# 这些都是它在训练语料中见过的、有意义的子词单元。

# 这就是 BPE (Byte-Pair Encoding) 算法的威力：
# 核心思想：通过统计和合并最高频的字节对，来构建一个既包含常用词也包含大量子词的词表。
# 1. 初始词表：词表一开始只包含基础字符（比如 ASCII 字符）。
# 2. 迭代合并：算法会不断扫描语料，找出出现频率最高的相邻子词对（比如 'T', 'o' -> 'To'），
#    然后将它们合并成一个新的、更长的子词单元，并加入词表。
# 3. 处理未知词：当遇到一个从未见过的词时（如 "Tokenizerisawesome"），
#    BPE 会尝试将这个词分解成它词表里存在的、最长的子词单元的组合。
#    如果实在不行，它最终可以回退到最基础的单个字符级别。

# 结论：
# 通过这种方式，模型既能高效地处理常见词（作为一个整体），又能灵活地表示任何新词、罕见词、甚至是拼写错误的词，
# 从而大大增强了模型的泛化能力，避免了 "未知词" (Unknown Token) 问题。
