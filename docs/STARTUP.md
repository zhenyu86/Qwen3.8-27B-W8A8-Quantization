# 启动方法

BF16 基线和 Enhanced W8A8 模型使用同一套 SGLang 启动方式，只有 `--model-path`
和 `--quantization` 两个参数不同。环境准备见 [SETUP.md](SETUP.md)，环境版本见
[ENVIRONMENT.md](ENVIRONMENT.md)。

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

## 4. 验证服务是否就绪

日志出现下面两行即表示服务已经就绪：

```text
INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
The server is fired up and ready to roll!
```

```bash
# 列出服务端注册的模型名
curl -s http://localhost:30000/v1/models

# 发一条请求，确认能正常生成
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "用一句话说明什么是模型量化。"}],
    "max_tokens": 64
  }'
```

`/v1/models` 返回的名字默认是 `--model-path`（可用 `--served-model-name` 改名）；
`/v1/chat/completions` 不校验 `model` 字段。

## 5. 停止服务

前台启动时直接 `Ctrl+C`。后台启动时先确认进程归属再结束：

```bash
ps -ef | grep sglang.launch_server | grep -v grep
# 确认是本次启动的进程后再结束
kill <pid>
```

## 6. 注意事项

1. 用同一个端口做公平对比时，先停止上一个服务。
2. 如果 `30000` 被占用，可以临时换端口，但 BF16 和 W8A8 必须使用相同端口配置。
3. 性能对比时不要额外启用 speculative decoding、radix cache、MTP、EAGLE 等参数，保持两次运行一致。
4. 量化模型启动时明确加 `--quantization w8a8_int8`。
5. 启动日志会完整打印生效的 `ServerArgs`，参数是否符合预期可以直接在日志里核对。
