# 结果对比

## 1. 测试条件

```text
GPU          : Hygon K100AI x 4
GPU 编号     : 4,5,6,7
TP           : 4
page-size    : 64
radix cache  : disabled
prompt length: 512 / 2048 / 8192
output length: 256
concurrency  : 1 / 4 / 8
run count    : 3
```

## 2. 核心结果

| 指标 | BF16 | Enhanced W8A8 |
|---|---:|---:|
| TTFT, prompt=2048, conc=1 | 431.9 ms | 362.9 ms |
| TPOT, prompt=2048, conc=1 | 49.5 ms | 25.1 ms |
| single_user_tok_s | 58.8 | 113.5 |
| concurrency4_tok_s | 225.8 | 372.0 |
| concurrency8_tok_s | 386.2 | 655.4 |
| PPL | 1.6430 | 1.6480 |
| PPL degradation | - | 0.0050 |
| Logits KLD | - | 0.00061 |
| Top-1 agreement | - | 98.05% |
| Top-5 all-match agreement | - | 75.39% |
| Logits cosine similarity | - | 0.99936 |
| 模型大小 | 51.75 GB | 29.09 GB |
| 4 卡总显存 | 约 187 GB | 约 188 GB |

## 3. 能力子集

| 任务 | BF16 | Enhanced W8A8 |
|---|---:|---:|
| GSM8K subset | 0.95 | 0.95 |
| MMLU subset | 1.00 | 1.00 |
| HumanEval subset | 1.00 | 1.00 |

这三个是自包含小型子集，不是完整公开 benchmark 全量结果。

## 4. 模型结构

| 项目 | 数值 |
|---|---:|
| Linear 总数 | 497 |
| INT8 Linear | 446 |
| BF16 Linear | 51 |
| BF16 packed groups | 25 |
| INT8 coverage | 89.74% |
| calibration samples | 128 |
| max calibration length | 1024 |

## 5. 结果文件

原始结果文件在 `benchmarks/`：

```text
results.csv
results_corrected.csv
summary.json
logits_comparison.json
ppl_comparison.json
capabilities.json
```

## 6. 结论

Enhanced W8A8 相比 BF16：

- 质量接近：PPL 退化很小，KLD 低于目标，Top-1 agreement 高于目标
- 速度明显提升：decode TPOT 约降低一半
- 模型大小明显下降：51.75 GB 到 29.09 GB
- 生成质量正常：三个能力子集上与 BF16 持平，输出无异常

本文档报告的是本项目的实测对比：Enhanced W8A8 与 BF16 基线。所有数字都来自
`benchmarks/` 下的原始结果文件，可以直接核对；测试条件见第 1 节。

## 7. 如何复现

1. 环境准备与自检：[SETUP.md](SETUP.md)（`python scripts/check_env.py`）
2. 运行量化：[USAGE.md](USAGE.md) 第 3 节
3. 检查量化产物：[USAGE.md](USAGE.md) 第 5 节
4. 启动服务（BF16 与 W8A8 各一次）：[STARTUP.md](STARTUP.md)
5. 精度与能力评估：[USAGE.md](USAGE.md) 第 6 节
6. 性能测速：[USAGE.md](USAGE.md) 第 7 节，结果写入 `benchmarks/`
