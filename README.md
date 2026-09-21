# Qwen3.8-27B W8A8 Quantization Project

这是一个独立的 Qwen3.8-27B W8A8 量化项目，包含量化脚本、校准数据、评估脚本、benchmark、日志和结果文档。

项目不包含两个大模型：

- 原始 BF16 模型：`/home/weight/Qwen3.8-27B`
- 量化后的模型：`/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8`

这两个模型需要单独保存或单独挂载。

## 目录结构

```text
Qwen3.8-27B-W8A8-Quantization/
├── README.md
├── docs/
│   ├── SETUP.md
│   ├── QUANTIZATION_METHOD.md
│   ├── USAGE.md
│   ├── STARTUP.md
│   ├── RESULTS.md
│   └── ENVIRONMENT.md
├── scripts/
│   ├── quantize.py
│   ├── calibrate.py
│   ├── eval_ppl.py
│   ├── compare_logits.py
│   ├── benchmark_sglang.py
│   ├── eval_capabilities.py
│   ├── check_quant_checkpoint.py
│   ├── fix_selective_bf16.py
│   └── legacy/
├── calibration/
│   ├── calib_data.jsonl
│   └── test_data.jsonl
├── benchmarks/
└── logs/
```

## 快速使用

量化命令：

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 python scripts/quantize.py \
  --model-path /home/weight/Qwen3.8-27B \
  --output-path /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8 \
  --calib-data calibration/calib_data.jsonl \
  --num-calib-samples 128 \
  --max-calib-length 1024 \
  --bf16-ratio 0.10 \
  --use-gptq \
  --damp 0.01 \
  --device cuda:0
```

量化模型启动：

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
  --model-path /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4 \
  --page-size 64 \
  --disable-radix-cache \
  --quantization w8a8_int8
```

详细说明见：

- [环境配置教程](docs/SETUP.md) ⬅️ **新文档**
- [量化方法](docs/QUANTIZATION_METHOD.md)
- [使用方法](docs/USAGE.md)
- [启动方法](docs/STARTUP.md)
- [结果对比](docs/RESULTS.md)
- [环境信息](docs/ENVIRONMENT.md)

## 最终方法摘要

本项目最终采用：

```text
Activation-aware GPTQ Learned Rounding
+ Per-output-channel INT8 Weight Quantization
+ SGLang W8A8 Dynamic Per-token Activation Quantization
+ Packed-group Selective BF16 Sensitive Modules
```

最终运行时格式是标准 SGLang `w8a8_int8`。

早期不可部署的 SmoothQuant 权重缩放方案已经移除，原因是当前 SGLang `w8a8_int8` kernel 不消费静态 per-channel 激活缩放，强行使用会导致模型输出错误。

## 最终结果摘要

| 指标 | BF16 | Enhanced W8A8 |
|---|---:|---:|
| TTFT, prompt=2048, conc=1 | 431.9 ms | 362.9 ms |
| TPOT, prompt=2048, conc=1 | 49.5 ms | 25.1 ms |
| single_user_tok_s | 58.8 | 113.5 |
| concurrency4_tok_s | 225.8 | 372.0 |
| concurrency8_tok_s | 386.2 | 655.4 |
| PPL | 1.6430 | 1.6480 |
| Logits KLD | - | 0.00061 |
| Top-1 agreement | - | 98.05% |
| 模型大小 | 51.75 GB | 29.09 GB |

