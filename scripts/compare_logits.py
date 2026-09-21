#!/usr/bin/env python3
"""
Compare BF16 and Enhanced-W8A8 logits without relying on a running SGLang
server.

The quantized checkpoint stores INT8 weights plus per-output-channel
``weight_scale`` tensors.  Hugging Face Transformers can load the weights, but
it treats them as raw BF16 values and ignores the scales.  This script therefore
loads the model and dequantizes every scaled Linear in-place before comparing
logits.

Memory strategy:
  * Run the BF16 model first and cache a small slice of per-token logits.
  * Unload the BF16 model.
  * Load/dequantize the quantized model and compare against the cached slice.

Only the last ``--positions`` token positions are cached by default.  This keeps
the cache a few hundred MB instead of tens of GB while still measuring the
distribution alignment that matters for generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_test_data(path: Path, tokenizer, num_samples: int, max_length: int):
    samples = []
    if path.exists():
        for line in path.read_text().splitlines():
            if len(samples) >= num_samples:
                break
            item = json.loads(line)
            text = item.get("text", item.get("content", ""))
            if not text:
                continue
            toks = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False,
            )["input_ids"]
            if toks.numel() >= 2:
                samples.append(toks)
    else:
        for _ in range(num_samples):
            samples.append(torch.randint(0, tokenizer.vocab_size, (1, max_length)))
    return samples


def dequantize_quantized_model(model, quant_path: Path):
    index = json.loads((quant_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    scale_files = {}
    scales = {}
    for key, shard in weight_map.items():
        if not key.endswith(".weight_scale"):
            continue
        if shard not in scale_files:
            scale_files[shard] = safe_open(quant_path / shard, framework="pt")
        scales[key] = scale_files[shard].get_tensor(key)

    def hf_to_ckpt(name: str) -> str:
        if name.startswith("model."):
            return "model.language_model." + name[len("model.") :]
        return name

    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith(".weight"):
                continue
            scale_key = hf_to_ckpt(name) + "_scale"
            if scale_key not in scales:
                continue
            scale = scales[scale_key].to(param.device, dtype=torch.float32)
            param.data = (param.data.float() * scale).to(param.dtype)
            count += 1

    for f in scale_files.values():
        try:
            f.__exit__(None, None, None)
        except Exception:
            pass
    return count


def collect_logits(model, samples, positions: int, cache_path: Path):
    cache = {}
    model.eval()
    with torch.no_grad():
        for i, input_ids in enumerate(samples):
            input_ids = input_ids.to(model.device)
            out = model(input_ids).logits.float().cpu()
            # Keep at most the last `positions` token positions.
            keep = min(positions, out.shape[1])
            cache[i] = out[:, -keep:, :]
            del out
            if (i + 1) % 16 == 0:
                print(f"    {i + 1}/{len(samples)} samples")
    torch.save(cache, cache_path)
    return cache


def compute_metrics(bf16_logits: torch.Tensor, quant_logits: torch.Tensor) -> dict:
    bf16_logits = bf16_logits.float().reshape(-1, bf16_logits.shape[-1])
    quant_logits = quant_logits.float().reshape(-1, quant_logits.shape[-1])

    p = F.softmax(bf16_logits, dim=-1)
    q = F.softmax(quant_logits, dim=-1)
    kld = F.kl_div(q.log(), p, reduction="batchmean").item()

    bf16_top1 = bf16_logits.argmax(dim=-1)
    quant_top1 = quant_logits.argmax(dim=-1)
    top1 = (bf16_top1 == quant_top1).float().mean().item()

    bf16_top5 = bf16_logits.topk(5, dim=-1).indices
    quant_top5 = quant_logits.topk(5, dim=-1).indices
    top5 = (bf16_top5 == quant_top5).all(dim=-1).float().mean().item()

    cosine = F.cosine_similarity(bf16_logits, quant_logits, dim=-1).mean().item()
    rel = (
        (bf16_logits - quant_logits).norm()
        / (bf16_logits.norm() + 1e-8)
    ).item()
    return {
        "kld": float(kld),
        "top1_agreement": float(top1),
        "top5_agreement": float(top5),
        "cosine_similarity": float(cosine),
        "relative_error": float(rel),
    }


def load_model(path: Path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-model", default="/home/weight/Qwen3.8-27B")
    parser.add_argument(
        "--quant-model",
        default="/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8",
    )
    parser.add_argument(
        "--test-data",
        default="/home/acceleration/quantization/calibration/test_data.jsonl",
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument(
        "--output",
        default="/home/acceleration/quantization/benchmarks/logits_comparison.json",
    )
    args = parser.parse_args()

    test_path = Path(args.test_data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output_path.with_suffix(".bf16_logits.pt")

    tokenizer = AutoTokenizer.from_pretrained(args.bf16_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = load_test_data(
        test_path, tokenizer, args.num_samples, args.max_length
    )
    print(f"Loaded {len(samples)} test samples")

    print("Running BF16 model...")
    bf16_model = load_model(Path(args.bf16_model))
    collect_logits(bf16_model, samples, args.positions, cache_path)
    del bf16_model
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading and dequantizing quantized model...")
    quant_model = load_model(Path(args.quant_model))
    n_scaled = dequantize_quantized_model(quant_model, Path(args.quant_model))
    print(f"  Dequantized {n_scaled} scaled Linear weights")

    cached = torch.load(cache_path, map_location="cpu")
    all_metrics = []
    with torch.no_grad():
        for i, input_ids in enumerate(samples):
            input_ids = input_ids.to(quant_model.device)
            out = quant_model(input_ids).logits.float().cpu()
            keep = min(args.positions, out.shape[1])
            quant_slice = out[:, -keep:, :]
            metrics = compute_metrics(cached[i], quant_slice)
            all_metrics.append(metrics)
            del out, quant_slice
            if (i + 1) % 16 == 0:
                print(f"    {i + 1}/{len(samples)} samples")

    summary = {
        "bf16_model": args.bf16_model,
        "quant_model": args.quant_model,
        "num_samples": len(all_metrics),
        "positions_per_sample": args.positions,
        "metrics": {
            key: {
                "mean": float(np.mean([m[key] for m in all_metrics])),
                "std": float(np.std([m[key] for m in all_metrics])),
            }
            for key in all_metrics[0]
        },
        "per_sample": all_metrics,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    cache_path.unlink(missing_ok=True)

    print("=" * 60)
    print("Logits alignment (BF16 vs Enhanced-W8A8)")
    print("=" * 60)
    for key, value in summary["metrics"].items():
        print(f"  {key}: {value['mean']:.6f} +/- {value['std']:.6f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
