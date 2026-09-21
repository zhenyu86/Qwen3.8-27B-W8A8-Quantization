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
python - <<'PY'
import torch, transformers, safetensors, sglang
print('torch', torch.__version__, 'hip', torch.version.hip)
print('transformers', transformers.__version__)
print('safetensors', safetensors.__version__)
print('sglang', sglang.__version__)
PY
```

