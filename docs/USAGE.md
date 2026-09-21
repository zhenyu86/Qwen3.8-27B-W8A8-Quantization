# 使用方法

本文档给出量化、检查、评估、测速的完整命令。环境准备见 [SETUP.md](SETUP.md)，
算法细节见 [QUANTIZATION_METHOD.md](QUANTIZATION_METHOD.md)，服务启动见
[STARTUP.md](STARTUP.md)。

除特别说明外，命令都在项目根目录 `/home/acceleration/Qwen3.8-27B-W8A8-Quantization` 下执行。

## 1. 环境

先跑一次自检，确认依赖、GPU、模型和数据都在位：

```bash
python scripts/check_env.py
```

环境版本见 [ENVIRONMENT.md](ENVIRONMENT.md)。直接使用当前 Python 环境，
不要重新安装或替换 torch、sglang、triton、torchvision、DTK。

## 2. 量化前检查

确认原始 BF16 模型存在：

```bash
ls /home/weight/Qwen3.8-27B
```

确认校准文件存在：

```bash
wc -l calibration/calib_data.jsonl
```

本次使用的是 512 条校准文本，量化时取前 128 条。

## 3. 运行量化

标准命令：

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

量化只需要一张卡：`HIP_VISIBLE_DEVICES` 列出可见的卡，`--device cuda:0` 指定用
其中的第一张。上面列出 4 张卡是为了和推理命令保持一致。

如果要用更多校准样本：

```bash
--num-calib-samples 512 --max-calib-length 2048
```

注意：这会增加显存和量化时间。

## 4. 参数说明

| 参数 | 本次值 | 说明 |
|---|---:|---|
| `--model-path` | `/home/weight/Qwen3.8-27B` | 原始 BF16 模型，只读 |
| `--output-path` | `/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8` | 量化输出目录 |
| `--calib-data` | `calibration/calib_data.jsonl` | 校准数据 |
| `--num-calib-samples` | `128` | 校准样本数量 |
| `--max-calib-length` | `1024` | 每条校准文本最大 token 数 |
| `--bf16-ratio` | `0.10` | 敏感 packed group 保留 BF16 的目标比例 |
| `--use-gptq` | 开启 | 启用 activation-aware learned rounding |
| `--no-gptq` | 关闭 | 关闭 learned rounding，退回普通四舍五入（仅用于对比） |
| `--damp` | `0.01` | Hessian diagonal damping |
| `--device` | `cuda:0` | 校准 forward 使用的卡，指 `HIP_VISIBLE_DEVICES` 里的第一张 |
| `--alpha` | `0.5` | 占位参数，当前算法不使用 |

## 5. 输出检查

检查权重和 scale：

```bash
python scripts/check_quant_checkpoint.py
```

检查 `config.json`：

```bash
python - <<'PY'
import json
c=json.load(open('/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8/config.json'))
print(c['quantization_config']['quant_method'])
print(len(c['quantization_config']['ignore']))
PY
```

## 6. 评估

评估需要同时加载 BF16 和量化模型，显存占用较高，建议在确认没有其它任务时运行。

PPL：

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_ppl.py \
  --bf16-model /home/weight/Qwen3.8-27B \
  --quant-model /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8 \
  --test-data calibration/test_data.jsonl \
  --num-samples 32 \
  --max-length 512
```

Logits 对齐：

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 python scripts/compare_logits.py \
  --bf16-model /home/weight/Qwen3.8-27B \
  --quant-model /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8 \
  --test-data calibration/test_data.jsonl \
  --num-samples 32 \
  --max-length 256 \
  --positions 8
```

能力子集需要先按 [STARTUP.md](STARTUP.md) 启动 SGLang 服务，然后运行：

```bash
python scripts/eval_capabilities.py \
  --url http://localhost:30000 \
  --model-name Qwen3.8-27B-Enhanced-W8A8
```

这里 `--model-name` 只是请求里的 `model` 字段，服务端不校验；填成服务的
`--served-model-name`（默认是模型路径）即可。

## 7. 测速

启动量化模型服务后运行：

```bash
python scripts/benchmark_sglang.py \
  --url http://localhost:30000 \
  --model-name Qwen3.8-27B-Enhanced-W8A8 \
  --quant-method enhanced_w8a8 \
  --prompt-lengths 512 2048 8192 \
  --output-length 256 \
  --concurrency-levels 1 4 8 \
  --num-runs 3 \
  --output-csv benchmarks/results_new.csv
```

## 8. 常见问题

### 输出乱码

优先检查：

1. `config.json` 中 `quantization_config.quant_method` 是否为 `w8a8_int8`。
2. 启动时是否加了 `--quantization w8a8_int8`。
3. checkpoint 是否由本项目 `scripts/quantize.py` 生成（可用 `scripts/check_quant_checkpoint.py` 检查权重与 scale 是否齐全）。
4. `weight_scale` 是否存在且数量正确。

### 显存不足

降低校准规模：

```bash
--num-calib-samples 64 --max-calib-length 512
```

量化校准阶段原始 BF16 模型加载在单卡上，如果单卡显存不够，需要改为多卡或减少样本长度。

### 量化后能力下降

可以尝试：

- 提高 `--num-calib-samples`
- 提高 `--max-calib-length`
- 提高 `--bf16-ratio` 到 0.12 或 0.15
- 调整 `--damp`
