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
├── requirements.txt
├── setup.sh
├── docs/
│   ├── SETUP.md
│   ├── QUANTIZATION_METHOD.md
│   ├── USAGE.md
│   ├── STARTUP.md
│   ├── RESULTS.md
│   └── ENVIRONMENT.md
├── scripts/
│   ├── check_env.py
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

环境自检（只读，不安装任何东西）：

```bash
python scripts/check_env.py      # 依赖版本 + GPU + 模型 + 数据 + 磁盘
./setup.sh                       # 解释器/驱动信息，缺依赖时用 ./setup.sh --install
```

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

## 文档导航

建议按下面的顺序阅读：

| 文档 | 内容 |
|------|------|
| [环境配置教程](docs/SETUP.md) | 硬件/软件要求、环境自检、端到端验证步骤、常见问题 |
| [环境信息](docs/ENVIRONMENT.md) | 实测版本清单、路径约定、环境约束 |
| [量化方法](docs/QUANTIZATION_METHOD.md) | 算法组成、校准统计、INT8 量化、selective BF16、导出格式 |
| [使用方法](docs/USAGE.md) | 量化、评估、测速的命令与参数说明 |
| [启动方法](docs/STARTUP.md) | BF16 基线与 W8A8 模型的 SGLang 启动命令 |
| [结果对比](docs/RESULTS.md) | 精度、能力子集、性能与模型大小对比 |

## 方法摘要

本项目采用：

```text
Activation-aware GPTQ Learned Rounding
+ Per-output-channel INT8 Weight Quantization
+ Packed-group Selective BF16 Sensitive Modules
+ SGLang 运行时动态 per-token 激活量化
```

量化在离线完成，只处理权重；激活的量化交给 SGLang 运行时按 token 动态完成。
导出的 checkpoint 是标准 SGLang `w8a8_int8` 格式，用 SGLang 原生加载路径和原生
kernel 即可运行，不需要额外的量化 runtime、自定义算子，也不需要改动推理输入。
算法细节见[量化方法](docs/QUANTIZATION_METHOD.md)。

## 结果摘要

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
