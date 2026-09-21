#!/usr/bin/env python3
"""
Generate calibration data for W8A8 quantization.
Uses diverse sources: Chinese, English, code, math, reasoning.
Output: JSONL format with 'text' field.
"""

import json
import os
import random

OUTPUT_PATH = "/home/acceleration/quantization/calibration/calib_data.jsonl"
NUM_SAMPLES = 512
MAX_LENGTH = 2048  # tokens (approximate, will be tokenized later)

# Chinese natural language samples
CHINESE_SAMPLES = [
    "人工智能的发展正在深刻改变我们的生活方式。从智能语音助手到自动驾驶汽车，从医疗诊断到金融分析，AI技术已经渗透到社会的各个领域。",
    "量子计算代表了计算技术的下一代革命。与经典计算机使用比特不同，量子计算机使用量子比特，可以同时处于多个状态。",
    "深度学习模型的量化技术是部署大语言模型到边缘设备的关键。通过将模型权重从浮点数转换为低精度整数，可以显著减少模型大小和推理延迟。",
    "自然语言处理中的注意力机制允许模型关注输入序列中最相关的部分。Transformer架构正是基于这一核心思想构建的。",
    "知识蒸馏是一种模型压缩技术，通过训练小型模型（学生模型）来模仿大型模型（教师模型）的行为，从而在保持性能的同时减少计算资源需求。",
    "强化学习是机器学习的一个重要分支，它通过让智能体与环境交互并根据奖励信号来学习最优策略。AlphaGo的成功就是强化学习的典型案例。",
    "联邦学习是一种分布式机器学习方法，允许多个参与者在不共享原始数据的情况下协作训练模型，有效保护了数据隐私。",
    "图神经网络是处理图结构数据的深度学习模型，在社交网络分析、分子属性预测、推荐系统等领域有广泛应用。",
    "多模态学习是指模型能够同时处理和理解多种类型的数据，如文本、图像、音频和视频。GPT-4V等模型展示了强大的多模态理解能力。",
    "自动化机器学习（AutoML）旨在使机器学习流程更加自动化，包括特征工程、模型选择和超参数优化等步骤。",
    "在医学影像分析中，卷积神经网络已经达到了甚至超越了人类专家的水平。特别是在皮肤癌检测、视网膜病变诊断等方面。",
    "知识图谱是一种结构化的知识表示方法，它将实体和关系组织成图的形式，广泛应用于搜索引擎、问答系统和推荐系统。",
    "迁移学习允许将在大规模数据集上学到的知识迁移到相关但不同的任务中，大大减少了新任务所需的训练数据和时间。",
    "对比学习是一种自监督学习方法，通过学习区分相似和不相似的样本来学习有效的数据表示。",
    "大语言模型的涌现能力是指当模型规模超过某个阈值时，突然出现的新能力，这些能力在较小的模型中不存在。",
]

# English samples
ENGLISH_SAMPLES = [
    "The transformer architecture has revolutionized natural language processing. Its self-attention mechanism allows the model to weigh the importance of different parts of the input sequence when producing each part of the output.",
    "Quantization-aware training (QAT) simulates the effects of quantization during the training process, allowing the model to learn to compensate for the loss of precision. This typically results in better accuracy than post-training quantization.",
    "The key innovation in SmoothQuant is the mathematical insight that activation outliers can be migrated from activations to weights through a smooth transformation, making both easier to quantize.",
    "Mixture of Experts (MoE) models achieve computational efficiency by activating only a subset of parameters for each input token, allowing for larger model capacity without proportional increases in computation.",
    "Chain-of-thought prompting improves the reasoning capabilities of large language models by encouraging them to show their step-by-step reasoning process before arriving at a final answer.",
    "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval, allowing models to access and synthesize information from large document collections.",
    "Gradient checkpointing is a memory optimization technique that trades compute for memory by selectively recomputing activations during the backward pass instead of storing them all.",
    "Flash Attention achieves significant speedups by reorganizing the attention computation to make better use of GPU memory hierarchy, reducing the number of memory reads and writes.",
    "The Grok-1 model demonstrates that sparse mixture-of-experts architectures can achieve competitive performance with dense models while using significantly fewer FLOPs per token.",
    "Low-rank adaptation (LoRA) efficiently fine-tunes large language models by freezing the original weights and training low-rank decomposition matrices, reducing the number of trainable parameters by orders of magnitude.",
    "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model by matching the soft probability distributions, capturing the teacher's dark knowledge.",
    "Structured pruning removes entire attention heads, neurons, or layers from a neural network, resulting in a smaller model that can be further fine-tuned to recover performance.",
    "The regret matching algorithm is fundamental to counterfactual regret minimization, which has been successfully applied to solve large-scale imperfect information games like Texas Hold'em poker.",
    "Convolutional neural networks exploit the spatial structure of images through local receptive fields, shared weights, and translation invariance, making them particularly effective for visual recognition tasks.",
    "The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between model complexity, training error, and generalization performance.",
]

# Code samples
CODE_SAMPLES = [
    """def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Dynamic programming approach
def fibonacci_dp(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_dp(n-1, memo) + fibonacci_dp(n-2, memo)
    return memo[n]""",
    """import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)""",
    """class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_dim=128, value_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.value_dim = value_dim
        self.to_q = nn.Linear(dim, num_heads * qk_dim, bias=False)
        self.to_k = nn.Linear(dim, num_heads * qk_dim, bias=False)
        self.to_v = nn.Linear(dim, num_heads * value_dim, bias=False)
        self.to_out = nn.Linear(num_heads * value_dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        q = self.to_q(x).view(B, L, self.num_heads, self.qk_dim)
        k = self.to_k(x).view(B, L, self.num_heads, self.qk_dim)
        v = self.to_v(x).view(B, L, self.num_heads, self.value_dim)
        q = torch.nn.functional.silu(q)
        k = torch.nn.functional.silu(k)
        kv = torch.einsum("blhd,blhe->bhde", k, v)
        q = q.permute(0, 2, 1, 3)
        out = torch.einsum("bnhd,bhde->bnhe", q, kv)
        out = out.reshape(B, L, -1)
        return self.to_out(out)""",
    """def quantize_weight_per_channel(weight, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scales = weight.abs().max(dim=1, keepdim=True).values / qmax
    quantized = torch.clamp(torch.round(weight / scales), -qmax - 1, qmax).to(torch.int8)
    return quantized, scales

def dequantize_weight(quantized, scales):
    return quantized.float() * scales

def smoothquant_scale(weight, activation, alpha=0.5):
    w_max = weight.abs().max(dim=0, keepdim=True).values
    a_max = activation.abs().max(dim=0, keepdim=True).values
    scale = torch.pow(a_max, alpha) / torch.pow(w_max, 1 - alpha)
    return scale""",
    """def top_k_top_p_sampling(logits, top_k=50, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)""",
]

# Math/Reasoning samples
MATH_SAMPLES = [
    "证明：对于任意正整数n，n^3 - n能被6整除。\n证明：n^3 - n = n(n^2-1) = (n-1)n(n+1)，这是三个连续整数的乘积。三个连续整数中必有一个是2的倍数，也必有一个是3的倍数，因此乘积能被6整除。",
    "求解微分方程 dy/dx = y/x + x/y 的通解。\n令v = y/x，则y = vx，dy/dx = v + x(dv/dx)。代入得：v + x(dv/dx) = v + 1/v，因此x(dv/dx) = 1/v，v·dv = dx/x，积分得v²/2 = ln|x| + C，即y² = 2x²ln|x| + Cx²。",
    "If a train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours, what is the average speed?\nTotal distance = 60 * 2.5 + 80 * 1.5 = 150 + 120 = 270 km\nTotal time = 2.5 + 1.5 = 4 hours\nAverage speed = 270 / 4 = 67.5 km/h",
    "A function f(x) = x^3 - 6x^2 + 11x - 6 has roots at x = 1, 2, 3. Verify using Vieta's formulas:\nSum of roots = 1 + 2 + 3 = 6 = -(-6)/1 ✓\nSum of products of pairs = 1*2 + 1*3 + 2*3 = 11 = 11/1 ✓\nProduct of roots = 1*2*3 = 6 = -(-6)/1 ✓",
    "证明柯西-施瓦茨不等式：|<u,v>|² ≤ <u,u>·<v,v>\n证明：对于任意实数t，有 <u+tv, u+tv> ≥ 0\n展开得：<u,u> + 2t<u,v> + t²<v,v> ≥ 0\n这是一个关于t的二次函数，判别式Δ ≤ 0\n4<u,v>² - 4<u,u><v,v> ≤ 0\n因此 <u,v>² ≤ <u,u><v,v>",
]

# Logic/reasoning samples
REASONING_SAMPLES = [
    "问题：有5个人排成一排，其中A和B不能相邻，有多少种排法？\n解法：总排列数5! = 120。A和B相邻的情况：将A和B捆绑看作一个整体，有4!×2 = 48种。因此A和B不相邻的排列数为120 - 48 = 72种。",
    "逻辑推理：所有的猫都是动物。有些动物是黑色的。因此，有些猫是黑色的。这个推理是否正确？\n答：这个推理是不正确的。从前提只能得出有些动物是黑色的，但不能确定这些黑色动物中是否有猫。这是一个无效的三段论推理。",
    "一个盒子中有3个红球和2个蓝球。随机取出2个球，求两个球颜色相同的概率。\nP(两个红球) = C(3,2)/C(5,2) = 3/10\nP(两个蓝球) = C(2,2)/C(5,2) = 1/10\nP(颜色相同) = 3/10 + 1/10 = 4/10 = 2/5",
    "分析以下论证的逻辑谬误：'张三说这个政策好，张三是经济学家，所以这个政策一定好。'\n这是一个诉诸权威的逻辑谬误。即使张三是经济学家，他的观点也不一定正确。我们需要独立评估政策本身的利弊，而不是仅凭提出者的身份来判断。",
    "Problem: In a group of 10 people, each person shakes hands with every other person exactly once. How many handshakes occur?\nSolution: This is a combination problem. We need to choose 2 people from 10 to form a handshake.\nC(10,2) = 10!/(2!*8!) = (10*9)/2 = 45 handshakes.",
]


def generate_calibration_data():
    """Generate diverse calibration data."""
    random.seed(42)
    samples = []

    all_sources = [
        (CHINESE_SAMPLES, 0.25),   # 25% Chinese
        (ENGLISH_SAMPLES, 0.25),   # 25% English
        (CODE_SAMPLES, 0.25),      # 25% Code
        (MATH_SAMPLES, 0.15),      # 15% Math
        (REASONING_SAMPLES, 0.10), # 10% Reasoning
    ]

    total_needed = NUM_SAMPLES

    for source, ratio in all_sources:
        n = int(total_needed * ratio)
        # Repeat and shuffle to get enough samples
        source_samples = []
        while len(source_samples) < n:
            source_samples.extend(source)
        random.shuffle(source_samples)
        source_samples = source_samples[:n]
        for s in source_samples:
            samples.append({"text": s})

    # Pad remaining if needed
    while len(samples) < total_needed:
        src = random.choice(all_sources)[0]
        samples.append({"text": random.choice(src)})

    random.shuffle(samples)
    samples = samples[:total_needed]

    # Write to file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Generated {len(samples)} calibration samples to {OUTPUT_PATH}")
    print(f"  Chinese: ~{int(total_needed * 0.25)}")
    print(f"  English: ~{int(total_needed * 0.25)}")
    print(f"  Code: ~{int(total_needed * 0.25)}")
    print(f"  Math: ~{int(total_needed * 0.15)}")
    print(f"  Reasoning: ~{int(total_needed * 0.10)}")


if __name__ == "__main__":
    generate_calibration_data()
