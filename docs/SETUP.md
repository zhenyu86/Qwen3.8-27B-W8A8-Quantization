# 环境配置教程

本文档指导你从零开始搭建 Qwen3.8-27B W8A8 量化项目的运行环境。

## 一、硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | Hygon K100AI × 1（64 GB） | Hygon K100AI × 4（256 GB） |
| 内存 | 64 GB | 128 GB 以上 |
| 磁盘 | 100 GB 可用空间 | 200 GB 以上（模型文件较大） |
| 系统 | Linux（支持 HIP/ROCm） | Ubuntu 20.04+ / CentOS 8+ |

> **注意**：本项目针对 Hygon K100AI GPU 开发，使用 HIP（ROCm 兼容层）。如果你使用 NVIDIA GPU，需要相应调整 PyTorch 和 SGLang 的安装方式。

## 二、基础环境

### 2.1 Python 环境

项目使用 **Python 3.10**，推荐使用 conda 创建独立环境：

```bash
# 安装 Miniconda（如果尚未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建项目环境
conda create -n qwen3-w8a8 python=3.10 -y
conda activate qwen3-w8a8
```

### 2.2 系统依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git wget curl

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git wget curl
```

## 三、安装 PyTorch（HIP/ROCm 版本）

本项目需要支持 Hygon GPU 的 PyTorch 版本。根据你的 GPU 驱动情况选择对应版本：

### 方案 A：使用预编译的 ROCm PyTorch

```bash
# 安装 PyTorch 2.10.0 + ROCm 6.3（根据实际 HIP 版本调整）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/rocm6.3
```

### 方案 B：使用 Hygon 官方提供的 PyTorch

如果 Hygon 提供了定制版 PyTorch，按照官方文档安装：

```bash
# 示例：安装 Hygon DTK 中包含的 PyTorch
# 具体命令请参考 Hygon 官方文档
pip install <hygon-pytorch-package>
```

### 验证安装

```bash
python -c "
import torch
print('PyTorch 版本:', torch.__version__)
print('HIP 版本:', torch.version.hip)
print('CUDA 可用:', torch.cuda.is_available())
print('GPU 数量:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:', torch.cuda.get_device_name(i))
"
```

## 四、安装项目依赖

### 4.1 核心依赖

```bash
# Transformers 和相关库
pip install transformers>=5.6.0
pip install safetensors>=0.8.0
pip install accelerate>=1.14.0
pip install datasets>=4.8.4

# compressed-tensors（用于量化支持）
pip install compressed-tensors>=0.15.0
```

### 4.2 SGLang 推理框架

SGLang 用于加载量化后的模型并提供推理服务：

```bash
# 安装 SGLang（需要与 PyTorch 和 HIP 版本匹配）
pip install sglang>=0.5.12

# 如果需要从源码安装（推荐，确保兼容性）
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e .
cd ..
```

### 4.3 其他工具库

```bash
# 请求库（用于 benchmark 和评估脚本）
pip install requests

# 可选：用于数据处理
pip install numpy
```

### 4.4 一键安装所有依赖

```bash
pip install torch transformers safetensors accelerate datasets compressed-tensors sglang requests numpy
```

## 五、完整依赖检查脚本

将以下脚本保存为 `check_env.py` 并运行，确认所有依赖已正确安装：

```python
#!/usr/bin/env python3
"""检查项目运行环境是否配置正确。"""

import sys

def check_package(name, min_version=None):
    """检查包是否已安装，返回 (名称, 版本, 状态)。"""
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "未知")
        return (name, version, "✅")
    except ImportError:
        return (name, "未安装", "❌")

def main():
    packages = [
        ("torch", "2.10.0"),
        ("transformers", "5.6.0"),
        ("safetensors", "0.8.0"),
        ("accelerate", "1.14.0"),
        ("datasets", "4.8.4"),
        ("compressed_tensors", "0.15.0"),
        ("sglang", "0.5.12"),
        ("requests", None),
        ("numpy", None),
    ]

    print("=" * 60)
    print("Qwen3.8-27B W8A8 量化项目 - 环境检查")
    print("=" * 60)

    all_ok = True
    results = []
    for name, min_ver in packages:
        pkg_name, version, status = check_package(name)
        results.append((pkg_name, version, status))
        if status == "❌":
            all_ok = False

    print(f"\n{'包名':<25} {'版本':<20} {'状态'}")
    print("-" * 60)
    for name, version, status in results:
        print(f"{name:<25} {version:<20} {status}")

    # 检查 PyTorch GPU 支持
    print("\n" + "=" * 60)
    print("GPU 检查")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"HIP 版本: {torch.version.hip}")
        print(f"CUDA/HIP 可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU 数量: {gpu_count}")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        else:
            print("⚠️  未检测到可用 GPU")
            all_ok = False
    except Exception as e:
        print(f"❌ GPU 检查失败: {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 所有检查通过，环境配置正确！")
    else:
        print("❌ 部分检查未通过，请根据上方提示修复问题")
    print("=" * 60)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
```

运行检查：

```bash
python check_env.py
```

## 六、下载模型

本项目需要两个模型文件：

| 模型 | 路径 | 说明 |
|------|------|------|
| 原始 BF16 模型 | `/home/weight/Qwen3.8-27B` | 量化脚本的输入 |
| 量化后模型 | `/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8` | 量化脚本的输出（自动生成） |

### 6.1 下载原始 BF16 模型

```bash
# 方式一：使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir /home/weight/Qwen3.8-27B

# 方式二：使用 git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B /home/weight/Qwen3.8-27B
```

> **注意**：请根据实际模型名称调整下载命令。`Qwen3.8-27B` 可能是 `Qwen3-27B` 或其他变体，请确认 Hugging Face 上的准确名称。

### 6.2 创建量化输出目录

```bash
mkdir -p /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8
```

## 七、验证完整工作流

完成以上配置后，按以下顺序验证：

```bash
# 1. 激活环境
conda activate qwen3-w8a8

# 2. 检查环境
python check_env.py

# 3. 运行量化（使用少量样本快速验证）
HIP_VISIBLE_DEVICES=0 python scripts/quantize.py \
  --model-path /home/weight/Qwen3.8-27B \
  --output-path /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8 \
  --calib-data calibration/calib_data.jsonl \
  --num-calib-samples 16 \
  --max-calib-length 512 \
  --bf16-ratio 0.10 \
  --use-gptq \
  --damp 0.01 \
  --device cuda:0

# 4. 启动 SGLang 推理服务
HIP_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --page-size 64 \
  --disable-radix-cache \
  --quantization w8a8_int8

# 5. 测试推理（另开终端）
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "你好，请做一个自我介绍。"}],
    "max_tokens": 128
  }'
```

## 八、常见问题

### Q1: `torch.cuda.is_available()` 返回 False

- 检查 GPU 驱动是否正确安装：`rocm-smi`（Hygon/ROCm）或 `nvidia-smi`（NVIDIA）
- 确认安装的 PyTorch 版本与 GPU 驱动兼容
- Hygon GPU 需要安装 HIP/DTK 运行时

### Q2: SGLang 安装失败

- 确保 PyTorch 已正确安装且版本匹配
- 尝试从源码安装 SGLang：`pip install git+https://github.com/sgl-project/sglang.git`
- 检查是否有编译错误，可能需要安装额外的编译工具

### Q3: 量化时 GPU 显存不足

- 减少校准样本数量：`--num-calib-samples 32`
- 减少最大校准长度：`--max-calib-length 512`
- 使用单卡运行：`HIP_VISIBLE_DEVICES=0`
- 量化脚本在统计激活值后会将模型移到 CPU，显存需求主要在加载阶段

### Q4: Hygon K100AI 与 NVIDIA GPU 的区别

本项目针对 Hygon K100AI GPU 开发，使用 HIP（与 ROCm 兼容的接口）。如果你使用 NVIDIA GPU：

1. 安装 CUDA 版本的 PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/cu121`
2. 将 `HIP_VISIBLE_DEVICES` 替换为 `CUDA_VISIBLE_DEVICES`
3. SGLang 启动参数可能需要调整，参考 SGLang 官方文档

## 九、快速安装脚本

将以下内容保存为 `setup.sh`，一键安装所有依赖：

```bash
#!/bin/bash
set -e

echo "=== Qwen3.8-27B W8A8 量化项目环境配置 ==="

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "请先安装 Miniconda 或 Anaconda"
    exit 1
fi

# 创建环境
echo "创建 conda 环境..."
conda create -n qwen3-w8a8 python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate qwen3-w8a8

# 安装依赖
echo "安装 Python 依赖..."
pip install --upgrade pip
pip install torch transformers safetensors accelerate datasets compressed-tensors sglang requests numpy

# 验证
echo "验证环境..."
python -c "import torch; print('PyTorch:', torch.__version__, '| HIP:', torch.version.hip)"

echo "=== 环境配置完成 ==="
echo "请运行 'conda activate qwen3-w8a8' 激活环境"
```

使用方法：

```bash
chmod +x setup.sh
./setup.sh
```

## 十、环境信息参考

本项目开发时使用的环境：

```text
Python        : 3.10.12
PyTorch       : 2.10.0
HIP           : 6.3.26113
Transformers  : 5.6.0
Safetensors   : 0.8.0
Accelerate    : 1.14.0
Datasets      : 4.8.4
compressed-tensors : 0.15.0.1
SGLang        : 0.5.12+g9a3a3e5f9
GPU           : Hygon K100AI, 每卡约 64 GB
```

> 版本号可能随更新变化，以上仅供参考。安装时请使用 `>=` 指定最低版本。
