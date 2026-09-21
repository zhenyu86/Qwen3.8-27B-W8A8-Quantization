# 环境配置教程

本文档说明如何准备并验证 Qwen3.8-27B W8A8 量化项目的运行环境。

环境有两种用法：

- **场景 A（推荐）**：机器上已经装好 DTK / PyTorch / SGLang，直接复用现有环境。本文档里的版本号和实测结果都来自这种机器。
- **场景 B**：一台全新的、什么都没有的机器，见第九节“全新机器从零搭建”。

> 复现本项目的结果**不需要重装任何东西**。直接复用现有环境是最稳的路径，原因见第二节的“重要约束”。

## 一、硬件要求

| 组件 | 最低要求 | 本项目实测机器 |
|------|----------|----------------|
| GPU | Hygon K100AI × 4（单卡 64 GB） | Hygon K100AI × 8，实测固定使用 `4,5,6,7` |
| 内存 | 256 GB | 约 1 TB |
| 磁盘 | 120 GB 可用 | `/home` 约 888 GB |
| 系统 | Linux + DTK/HIP | Linux + DTK（`/opt/dtk`），HIP 6.3.26113 |

说明：

- 量化校准阶段会把完整的 BF16 模型（52 GB）加载到**单张卡**上，所以单卡显存必须放得下 52 GB；4 张 64 GB 的卡在推理阶段做 TP=4。
- 磁盘需要同时容纳只读的 BF16 模型（52 GB）、量化产物（30 GB）以及日志和中间文件，建议留 120 GB 以上。
- 本项目只针对 Hygon K100AI（HIP）验证过。换成 NVIDIA GPU 需要自备 CUDA 版 PyTorch 与 SGLang，见第八节 Q4。

## 二、软件环境（实测版本）

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10.12 | 系统解释器 `/usr/bin/python`，没有使用 conda |
| PyTorch | 2.10.0 | 由 Hygon DTK 提供，`torch.version.hip = 6.3.26113`、`torch.version.cuda = None` |
| SGLang | 0.5.12+g9a3a3e5f9 | CLI 在 `/usr/local/bin/sglang` |
| Transformers | 5.6.0 | |
| Safetensors | 0.8.0 | |
| Accelerate | 1.14.0 | |
| Datasets | 4.8.4 | |
| compressed-tensors | 0.15.0.1 | 量化配置解析 |
| requests / numpy | 2.34.2 / 1.25.0 | benchmark 与评估脚本使用 |

Python 包统一安装在 `/usr/local/lib/python3.10/dist-packages`。完整清单见 [ENVIRONMENT.md](ENVIRONMENT.md)。

### 重要约束

项目原始约束（见 [ENVIRONMENT.md](ENVIRONMENT.md)）明确要求：

- **不重新安装** `torch`、`triton`、`sglang`、`torchvision`、DTK
- **不新建** conda 环境
- 原始 BF16 模型只读
- 所有中间产物和日志放在项目目录下

原因：Hygon K100AI 上的 PyTorch 是 DTK 随驱动提供的定制版本（`torch.version.hip` 有值、`torch.version.cuda` 为 `None`）。它**不是** `download.pytorch.org` 上给 AMD 显卡用的 ROCm 轮子。按官网命令重装 torch 极可能直接破坏现有环境，让 SGLang 无法启动，而且很难回滚。

因此：**能跑就不要动**。只有在全新机器上才需要安装，且 torch 必须来自 Hygon DTK。

## 三、路径约定

| 用途 | 路径 | 说明 |
|------|------|------|
| 原始 BF16 模型 | `/home/weight/Qwen3.8-27B` | 量化脚本的输入，只读，52 GB |
| 量化输出模型 | `/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8` | 量化脚本的输出，30 GB |
| 本项目 | `/home/acceleration/Qwen3.8-27B-W8A8-Quantization` | 脚本、校准数据、日志、结果 |

脚本里的默认参数就是这几个路径，直接运行可以不带参数。

## 四、环境自检

仓库自带两个检查入口，都只做只读检查，不会安装或修改任何东西。

```bash
# 完整体检：依赖版本 + GPU + 模型 + 数据 + 磁盘
python scripts/check_env.py

# 解释器/驱动信息 + 依赖检查（缺依赖时可加 --install）
./setup.sh
```

`scripts/check_env.py` 会逐项检查：

1. 9 个 Python 依赖是否可导入、版本是否达到要求；
2. torch 是否带 HIP 支持、可见的 GPU 数量和显存；
3. BF16 模型与量化模型目录是否完整；
4. 量化模型的 `quantization_config.quant_method` 是否为 `w8a8_int8`；
5. 校准/测试数据条数，以及量化目录的剩余磁盘空间。

退出码 0 表示通过，1 表示有阻塞项。在本项目实测机器上，关键结果是：

```text
torch 2.10.0 | HIP 6.3.26113 | 8 张 K100_AI（64.0 GB / 卡）
BF16 模型: 18 个 safetensors 分片
W8A8 模型: 7 个分片, quant_method=w8a8_int8
校准数据: calib_data.jsonl 512 条, test_data.jsonl 256 条
```

只缺第三方依赖时，可以用：

```bash
./setup.sh --install          # 按 requirements.txt 安装，不会碰 torch / DTK / SGLang
```

`requirements.txt` 故意不包含 torch、torchvision、triton、sglang，避免 pip 覆盖掉 DTK 提供的版本。

## 五、模型与校准数据

### 5.1 模型

仓库**不包含**两个大模型权重，需要单独保存或挂载：

| 模型 | 路径 | 来源 |
|------|------|------|
| 原始 BF16 | `/home/weight/Qwen3.8-27B` | 量化输入，保持只读 |
| 量化产物 | `/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8` | 由 `scripts/quantize.py` 生成 |

目录里至少要有 `config.json`、`model.safetensors.index.json` 和对应分片文件。从别的机器迁移时整目录拷贝即可：

```bash
rsync -a --info=progress2 /path/to/Qwen3.8-27B/ /home/weight/Qwen3.8-27B/
```

量化输出目录会被脚本重新写入，**不要**把它指向原始 BF16 模型目录。

### 5.2 校准数据

仓库自带两份数据：

| 文件 | 条数 | 用途 |
|------|------|------|
| `calibration/calib_data.jsonl` | 512 | 量化校准，实测取前 128 条 |
| `calibration/test_data.jsonl` | 256 | PPL / logits 对比，实测取前 32 条 |

需要重新生成时运行 `python scripts/calibrate.py`。注意该脚本的输出路径写死在脚本里（`/home/acceleration/quantization/calibration/calib_data.jsonl`），生成后需要自己复制到仓库的 `calibration/` 目录。

## 六、端到端验证

以下 5 步都在项目根目录执行。

### 步骤 1：环境自检

```bash
cd /home/acceleration/Qwen3.8-27B-W8A8-Quantization
python scripts/check_env.py
```

必须退出码为 0 再继续。

### 步骤 2：运行量化

量化只用一张卡：`HIP_VISIBLE_DEVICES` 决定用哪张物理卡（下面示例用 4 号），`--device cuda:0` 指的是“可见卡中的第一张”。完整命令见 [USAGE.md](USAGE.md)。快速验证用小样本：

```bash
HIP_VISIBLE_DEVICES=4 python scripts/quantize.py \
  --model-path /home/weight/Qwen3.8-27B \
  --output-path /home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8 \
  --calib-data calibration/calib_data.jsonl \
  --num-calib-samples 16 \
  --max-calib-length 512 \
  --bf16-ratio 0.10 \
  --use-gptq \
  --damp 0.01 \
  --device cuda:0
```

正式复现请把 `--num-calib-samples` 改成 128、`--max-calib-length` 改成 1024（即 [USAGE.md](USAGE.md) 里的标准命令）。校准样本越多越慢，显存峰值也越高。

### 步骤 3：检查量化产物

```bash
python scripts/check_quant_checkpoint.py

python - <<'PY'
import json
config = json.load(open('/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8/config.json'))
print(config['quantization_config']['quant_method'])
print('BF16 保留模块:', len(config['quantization_config']['ignore']))
PY
```

期望 `quant_method` 为 `w8a8_int8`。实测结构：497 个 Linear 中 446 个 INT8、51 个 BF16（25 个 packed group），INT8 覆盖率 89.74%。

### 步骤 4：启动 SGLang 服务

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

要点：

- 量化模型**必须**显式加 `--quantization w8a8_int8`，否则可能按 BF16 加载而报错或变慢。
- 端口被占用时先停掉上一个服务；BF16 与 W8A8 做公平对比时要使用相同的端口和 TP 配置。
- SGLang 会提示 `python -m sglang.launch_server` 已推荐改用 `sglang serve`，两者参数一致。
- 想让 API 里的模型名更好看，可以加 `--served-model-name`（该参数只影响命名，不影响性能）。

### 步骤 5：验证推理

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "你好，请做一个自我介绍。"}],
    "max_tokens": 128
  }'
```

`/v1/chat/completions` 不会校验 `model` 字段，随便填即可；但 `/v1/models` 返回的名字等于 `--served-model-name`（默认是模型路径），脚本里若要做一致性判断请用同一个名字。

更完整的评估和测速命令见 [USAGE.md](USAGE.md)，结果口径见 [RESULTS.md](RESULTS.md)。

## 七、环境变量速查

| 变量 | 用途 | 本项目取值 |
|------|------|------------|
| `HIP_VISIBLE_DEVICES` | 选择可见的 Hygon GPU | 量化 `4`；推理 `4,5,6,7` |
| `HF_HOME` / `HF_HUB_CACHE` | HuggingFace 缓存位置 | 可选，离线环境建议指向本地盘 |
| `PYTHONPATH` | 一般不需要设置 | 脚本通过相对/绝对路径导入 |

不需要设置额外的 HIP / SGLang 特殊变量；实测环境直接使用默认值即可。

## 八、常见问题

### Q1: `torch.cuda.is_available()` 返回 False

1. 确认驱动正常：`rocm-smi`（DTK 自带，在 `/opt/dtk/bin`）或 `hy-smi`（在 `/opt/hyhal/bin`）。
2. 确认用的是项目环境里的解释器：`python -c "import torch; print(torch.__version__, torch.version.hip)"`，`torch.version.hip` 应该有值。
3. 如果 `torch.version.hip` 是 `None`，说明 torch 被换成了 CUDA 或 CPU 版本，需要恢复 DTK 提供的版本。

### Q2: 量化时显存不足

- 校准只用单卡，先减少样本：`--num-calib-samples 64 --max-calib-length 512`。
- 校准阶段结束、统计完激活值后模型会搬到 CPU，显存峰值主要出现在加载和首次 forward。
- 确认没有别的进程占用同一张卡：`rocm-smi --showmeminfo vram`。

### Q3: 端口 30000 被占用

```bash
ss -ltnp | grep 30000
```

换端口即可，但 BF16 和 W8A8 对比时两边要保持一致。

### Q4: 换成 NVIDIA GPU 怎么改

1. 安装对应 CUDA 版本的 PyTorch 与 SGLang（这部分需要自己准备，本项目未验证）。
2. 把命令里的 `HIP_VISIBLE_DEVICES` 换成 `CUDA_VISIBLE_DEVICES`。
3. SGLang 的 attention backend、量化 kernel 支持范围可能不同，需要按官方文档调整参数。

### Q5: 输出乱码或重复

按顺序排查：

1. `config.json` 里 `quantization_config.quant_method` 是否为 `w8a8_int8`；
2. 启动时是否加了 `--quantization w8a8_int8`；
3. 是否误用了早期的 SmoothQuant checkpoint（当前 runtime 不消费静态 per-channel 激活缩放，强行使用会输出乱码，详见 [QUANTIZATION_METHOD.md](QUANTIZATION_METHOD.md)）；
4. `weight_scale` 是否存在且数量与 INT8 权重匹配。

### Q6: SGLang 装不上 / 想升级 SGLang

不建议在能跑通的机器上升级。先确认问题是否真的来自 SGLang；如果确实需要从源码安装，请在另一套环境里验证后再替换，不要直接覆盖当前可用的版本。

### Q7: 模型太大，磁盘放不下

BF16 模型 52 GB + 量化产物 30 GB 是最小占用，量化过程中还会产生临时分片。磁盘不足时先清理 `logs/` 里的旧日志，或把输出目录指到其它盘（`--output-path`）。

## 九、全新机器从零搭建

> 本项目所有实测结果都不是在全新机器上得到的。这一节只是给需要从零开始的机器一个参考流程，Hygon 相关的驱动和 DTK 请以官方文档为准。

### 9.1 系统依赖

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y build-essential cmake git curl python3.10 python3.10-dev python3-pip

# CentOS / RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git curl python3.10 python3.10-devel
```

### 9.2 驱动与 DTK

按 Hygon 官方文档安装 DCU 驱动和 DTK（本项目实测机器上 DTK 位于 `/opt/dtk`）。安装后应能看到：

```bash
/opt/dtk/bin/rocm-smi     # 应列出全部 DCU
/opt/hyhal/bin/hy-smi
```

### 9.3 PyTorch 与 SGLang

```bash
# torch 必须使用 DTK 附带的构建，不要用官网 CUDA/ROCm 轮子
python3 -c "import torch; print(torch.__version__, torch.version.hip)"

# SGLang
pip install "sglang>=0.5.12"
```

如果 DTK 没有自带 torch，请按 Hygon 官方提供的安装包安装，安装完再重新执行上面的验证命令。

### 9.4 其余依赖

```bash
# 注意：>= 号必须加引号，否则 shell 会把它当成重定向
pip install -r requirements.txt
```

等价的手动写法：

```bash
pip install "transformers>=5.6.0" "safetensors>=0.8.0" "accelerate>=1.14.0" \
            "datasets>=4.8.4" "compressed-tensors>=0.15.0" requests numpy
```

### 9.5 验证

```bash
python scripts/check_env.py
```
