# 量化方法详细说明

## 1. 目标和约束

目标模型是 Qwen3.8-27B BF16，本地推理环境是 Hygon K100AI + SGLang TP4。

量化目标：

- 最终模型必须是 SGLang 可原生加载的 `w8a8_int8`
- 权重是 INT8
- 激活由运行时动态量化
- 尽量降低精度损失
- 尽量提升 decode 速度
- 不依赖额外的 GPTQ runtime 或不可部署的 SmoothQuant runtime

## 2. 最终算法组成

最终算法为：

```text
Activation-aware GPTQ Learned Rounding
+ Per-output-channel INT8 Weight Quantization
+ Selective BF16 Packed Modules
```

推理时使用 SGLang 自带的动态 per-token 激活量化。

## 3. 为什么移除了旧 SmoothQuant 缩放方案

早期实现做过：

```text
W_scaled = W * s
```

但 SmoothQuant 要数值等价，推理输入也必须变成：

```text
X_scaled = X / s
```

当前 SGLang `w8a8_int8` 路径中的激活量化是动态 per-token，不接受静态 per-channel 激活缩放。早期 checkpoint 只改了权重，没有把 `1/s` 迁移到模型其它部分，所以虽然速度正常，实际输出是乱码。

最终版本删除了不可部署的缩放，直接对原始 BF16 权重做 activation-aware GPTQ W8A8。

## 4. 校准数据

校准数据格式为 JSONL，每一行包含文本：

```json
{"text": "..."}
```

本次使用：

```text
128 条 calibration samples
每条最多 1024 tokens
```

数据混合：

- 中文自然语言
- 英文自然语言
- 代码
- 数学
- 逻辑推理

校准 forward 过程中，对每个 Linear 层收集：

```text
act_abs_max[in_features]
act_sq_sum[in_features]
act_count
```

这些统计数据用于后续 weight rounding 和敏感层排序。

## 5. 每输出通道 INT8 权重量化

对任意 Linear 权重：

```text
W in R^(out_features x in_features)
```

按输出通道计算对称量化 scale：

```text
scale[o] = max_j(abs(W[o, j])) / 127
```

初始量化：

```text
Q0[o, j] = round(W[o, j] / scale[o])
Q0[o, j] = clamp(Q0[o, j], -128, 127)
```

最终保存：

```text
weight      : int8
weight_scale: float32, shape [out_features, 1]
```

推理时反量化：

```text
W_hat = Q * scale
```

## 6. Activation-aware GPTQ Learned Rounding

普通 round 只按最近整数取整：

```text
Q = round(W / scale)
```

GPTQ learned rounding 会同时考虑 calibration activation 对不同输入通道的敏感度。

用二阶矩近似 Hessian 对角：

```text
H_diag[k] = mean(X[:, k]^2)
```

加入 damping：

```text
H_diag = H_diag + damp * mean(H_diag)
```

本次：

```text
damp = 0.01
```

对每个权重同时计算 floor 和 ceil 两种候选：

```text
Q_floor = floor(W / scale)
Q_ceil  = ceil(W / scale)
```

分别计算 Hessian 加权重建误差：

```text
err_floor = (W - Q_floor * scale)^2 * H_diag
err_ceil  = (W - Q_ceil  * scale)^2 * H_diag
```

选择误差更小的候选：

```text
Q = Q_ceil if err_ceil < err_floor else Q_floor
```

这一步只改变 round 方向，不改变 runtime layout。最终仍然是标准 INT8 weight 和 per-output-channel scale。

## 7. Selective BF16

Qwen3.8-27B 的 SGLang 模型会把多个 checkpoint 权重融合成一个运行时 Linear：

```text
q_proj + k_proj + v_proj -> qkv_proj
gate_proj + up_proj      -> gate_up_proj
in_proj_qkv + in_proj_z  -> in_proj_qkvz
in_proj_a + in_proj_b    -> in_proj_ba
```

所以 selective BF16 必须按 packed group 做，不能只把组内一个分量保留 BF16。否则 fused group 会出现部分 INT8、部分 BF16，SGLang 无法合法加载。

选择流程：

1. 对每个 Linear 计算权重重建相对误差：

```text
relative_error = ||W - Q*scale|| / ||W||
```

2. 把 Linear 映射到 SGLang packed group。
3. 对每个 group 取组内最大相对误差。
4. 按误差从大到小排序。
5. 根据 `--bf16-ratio` 选择最敏感的 group 保留 BF16。
6. `lm_head` 固定保留 BF16。

本次结果：

```text
Linear 总数        : 497
INT8 Linear        : 446
BF16 Linear        : 51
BF16 packed groups : 25
INT8 coverage      : 约 89.74%
```

## 8. 导出格式

输出为标准 SGLang `w8a8_int8` checkpoint：

```text
config.json
model.safetensors.index.json
model-*.safetensors
quantization_report.json
tokenizer / chat template 等
```

`config.json` 写入：

```json
{
  "quantization_config": {
    "quant_method": "w8a8_int8",
    "ignore": ["需要保持 BF16 的 SGLang packed module"],
    "quant_description": {
      "某个.weight": "INT8_PER_CHANNEL",
      "某个.weight": "FLOAT"
    }
  }
}
```

其中：

- INT8 层保存 `weight` 和 `weight_scale`
- BF16 层只保存原始 BF16 `weight`
- `ignore` 列表告诉 SGLang 哪些 packed module 不量化

## 9. 为什么这样能加速

推理时，SGLang `w8a8_int8` 路径执行：

```text
activation -> per-token INT8 dynamic quantization
weight     -> per-channel INT8
GEMM       -> INT8 x INT8
epilogue   -> dequant / scale / bias
```

与 BF16 GEMM 相比，权重显存和 GEMM 带宽压力明显下降，因此 decode 阶段速度提升。

本次 benchmark 中：

```text
TPOT 从 49.5 ms 降到 25.1 ms
模型大小从 51.75 GB 降到 29.09 GB
```

## 10. 验证方法

量化后需要做四类验证：

1. 文本输出检查：确认不是乱码，可以正常回答简单问题。
2. PPL：BF16 与 W8A8 在固定文本子集上的 perplexity。
3. Logits alignment：KLD、Top-1 agreement、Top-5 agreement、cosine similarity。
4. 能力子集：GSM8K、MMLU、HumanEval 小型子集。

本项目已经保存了这些验证结果。

