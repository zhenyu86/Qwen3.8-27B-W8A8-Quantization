# 启动方法

以下命令只使用当前安装好的 SGLang 环境，不重新安装 DTK、torch 或 SGLang。

## 1. BF16 基线启动

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
  --model-path /home/weight/Qwen3.8-27B \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4 \
  --page-size 64 \
  --disable-radix-cache
```

## 2. Enhanced W8A8 启动

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

## 3. 参数对应关系

| 参数 | BF16 | Enhanced W8A8 |
|---|---|---|
| GPU | `4,5,6,7` | `4,5,6,7` |
| TP | `4` | `4` |
| port | `30000` | `30000` |
| page-size | `64` | `64` |
| disable-radix-cache | 开启 | 开启 |
| quantization | 无 | `w8a8_int8` |
| model-path | 原始 BF16 | 量化目录 |

## 4. 注意事项

1. 用同一个端口做公平对比时，先停止上一个服务。
2. 如果 `30000` 被占用，可以临时换端口，但 BF16 和 W8A8 必须使用相同端口配置。
3. 第一阶段性能对比不要额外启用 speculative decoding、radix cache、MTP、EAGLE 等参数。
4. 量化模型启动时明确加 `--quantization w8a8_int8`。

