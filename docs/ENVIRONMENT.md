# 环境信息

本次量化使用的环境如下。

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

补充说明：

- Python 是系统解释器 `/usr/bin/python`，包安装在 `/usr/local/lib/python3.10/dist-packages`，**没有使用 conda**。
- 机器共 8 张 K100AI，实测固定使用 `4,5,6,7` 四张卡做 TP=4 推理；量化校准只用其中一张。
- PyTorch 由 Hygon DTK 提供（DTK 位于 `/opt/dtk`），`torch.version.hip` 有值、`torch.version.cuda` 为 `None`。
- 驱动/监控工具：`/opt/dtk/bin/rocm-smi`、`/opt/hyhal/bin/hy-smi`。

重要约束：

- 不重新安装 torch、triton、sglang、torchvision、DTK
- 不新建 conda 环境
- 原始 BF16 模型只读
- 所有中间产物和日志放在项目目录下

## 路径

```text
原始 BF16 模型:
/home/weight/Qwen3.8-27B

量化模型:
/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8

本项目:
/home/acceleration/Qwen3.8-27B-W8A8-Quantization
```

## 依赖检查

```bash
python scripts/check_env.py
```

该脚本同时检查依赖版本、GPU、模型目录、校准数据和磁盘空间，只读执行，不安装任何东西。
安装步骤、路径约定和常见问题见 [SETUP.md](SETUP.md)。
