#!/usr/bin/env python3
"""
Small perplexity comparison between BF16 and the Enhanced-W8A8 checkpoint.

The quantized checkpoint is dequantized in-place (INT8 weight * per-channel
scale) before the Hugging Face forward pass, which lets us compute an offline
approximation of the actual SGLang W8A8 weights without needing the server.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(path: Path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )


def dequantize_model(model, quant_path: Path):
    index = json.loads((quant_path / "model.safetensors.index.json").read_text())
    files = {}
    scales = {}
    for key, shard in index["weight_map"].items():
        if not key.endswith(".weight_scale"):
            continue
        if shard not in files:
            files[shard] = safe_open(quant_path / shard, framework="pt")
        scales[key] = files[shard].get_tensor(key)

    def hf_to_ckpt(name):
        return "model.language_model." + name[len("model.") :] if name.startswith("model.") else name

    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            scale_key = hf_to_ckpt(name) + "_scale"
            if name.endswith(".weight") and scale_key in scales:
                scale = scales[scale_key].to(param.device, dtype=torch.float32)
                param.data = (param.data.float() * scale).to(param.dtype)
                count += 1
    for f in files.values():
        try:
            f.__exit__(None, None, None)
        except Exception:
            pass
    return count


def compute_ppl(model, tokenizer, samples, max_length):
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(samples):
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.numel() < 2:
                continue
            out = model(input_ids, labels=input_ids)
            loss = out.loss
            if loss is not None and torch.isfinite(loss):
                total_nll += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
            del out, input_ids
            if (i + 1) % 8 == 0:
                print(f"    {i + 1}/{len(samples)} samples")
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


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
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--output",
        default="/home/acceleration/quantization/benchmarks/ppl_comparison.json",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.bf16_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []
    for line in Path(args.test_data).read_text().splitlines():
        if len(samples) >= args.num_samples:
            break
        item = json.loads(line)
        text = item.get("text", "")
        if text:
            samples.append(text)
    print(f"Loaded {len(samples)} text samples")

    print("Computing BF16 PPL...")
    bf16_model = load_model(Path(args.bf16_model))
    bf16_ppl = compute_ppl(bf16_model, tokenizer, samples, args.max_length)
    del bf16_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  BF16 PPL = {bf16_ppl:.4f}")

    print("Computing Enhanced-W8A8 PPL...")
    quant_model = load_model(Path(args.quant_model))
    n = dequantize_model(quant_model, Path(args.quant_model))
    print(f"  Dequantized {n} scaled weights")
    quant_ppl = compute_ppl(quant_model, tokenizer, samples, args.max_length)
    print(f"  W8A8 PPL = {quant_ppl:.4f}")

    result = {
        "bf16_model": args.bf16_model,
        "quant_model": args.quant_model,
        "num_samples": len(samples),
        "max_length": args.max_length,
        "bf16_ppl": bf16_ppl,
        "w8a8_ppl": quant_ppl,
        "ppl_degradation": quant_ppl - bf16_ppl,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
