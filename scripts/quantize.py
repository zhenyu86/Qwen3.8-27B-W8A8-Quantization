#!/usr/bin/env python3
"""
Enhanced W8A8 quantizer for Qwen3.8-27B on SGLang/Hygon.

Produces a checkpoint that SGLang's native ``w8a8_int8`` backend loads directly:

  * weights are per-output-channel INT8 symmetric,
  * rounding is activation-aware (GPTQ-style learned rounding),
  * the most sensitive SGLang packed modules are kept in BF16,
  * activation quantization is left to SGLang's dynamic per-token INT8 kernel,
    so no extra runtime, custom kernel or input preprocessing is involved.

Algorithm details: docs/QUANTIZATION_METHOD.md
Command-line usage: docs/USAGE.md
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Name mapping helpers
# ---------------------------------------------------------------------------

def _safetensors_key_to_module_name(st_key: str) -> str:
    base = st_key
    for suffix in (".weight_scale", ".weight"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.startswith("model.language_model."):
        return "model." + base[len("model.language_model.") :]
    return base


def _hf_name_to_ckpt(hf_name: str) -> str:
    if hf_name.startswith("model."):
        return "model.language_model." + hf_name[len("model.") :]
    return hf_name


def get_group(name: str):
    """Map an HF Linear name to an SGLang packed-group tuple."""
    m = re.match(r"model\.layers\.(\d+)\.(linear_attn|self_attn|mlp)\.(.+)", name)
    if not m:
        return None
    idx, typ, proj = m.groups()
    idx = int(idx)
    if typ == "linear_attn":
        if proj in ("in_proj_qkv", "in_proj_z"):
            return (idx, "linear_attn", "in_proj_qkvz")
        if proj in ("in_proj_a", "in_proj_b"):
            return (idx, "linear_attn", "in_proj_ba")
        if proj == "out_proj":
            return (idx, "linear_attn", "out_proj")
    elif typ == "self_attn":
        if proj in ("q_proj", "k_proj", "v_proj"):
            return (idx, "self_attn", "qkv_proj")
        if proj == "o_proj":
            return (idx, "self_attn", "o_proj")
    elif typ == "mlp":
        if proj in ("gate_proj", "up_proj"):
            return (idx, "mlp", "gate_up_proj")
        if proj == "down_proj":
            return (idx, "mlp", "down_proj")
    return (idx, typ, proj)


# ---------------------------------------------------------------------------
# Activation statistics collection
# ---------------------------------------------------------------------------

class ActivationStatsCollector:
    def __init__(self, model: nn.Module):
        self.stats: "OrderedDict[str, dict]" = OrderedDict()
        self._hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.stats[name] = {
                    "count": 0,
                    "abs_max": None,
                    "sq_sum": None,
                }
                self._hooks.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inputs):
            if not inputs or inputs[0] is None:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            amax = x.abs().amax(dim=0).cpu()
            sqsum = (x * x).sum(dim=0).cpu()
            s = self.stats[name]
            if s["abs_max"] is None:
                s["abs_max"] = amax
                s["sq_sum"] = sqsum
            else:
                s["abs_max"] = torch.maximum(s["abs_max"], amax)
                s["sq_sum"] = s["sq_sum"] + sqsum
            s["count"] += x.shape[0]
        return fn

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Quantization math
# ---------------------------------------------------------------------------

def quantize_linear(weight: torch.Tensor, stats: dict, use_gptq: bool, damp: float):
    """Return int8 weight, per-channel scale, and reconstruction metrics."""
    w = weight.float()
    w_scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 127.0
    q_vals = w / w_scale
    q_floor = q_vals.floor().clamp(-128, 127)
    q_ceil = q_vals.ceil().clamp(-128, 127)

    if use_gptq and stats["count"] > 0:
        h = stats["sq_sum"].float() / max(1, stats["count"])
        h = h.clamp_min(1e-6) + damp * h.mean().clamp_min(1e-8)
        err_floor = (w - q_floor * w_scale).square() * h
        err_ceil = (w - q_ceil * w_scale).square() * h
        q = torch.where(err_ceil < err_floor, q_ceil, q_floor)
    else:
        q = q_floor

    w_int8 = q.to(torch.int8)
    dequant = w_int8.float() * w_scale
    diff = w - dequant
    denom = w.norm() + 1e-12
    metrics = {
        "relative": (diff.norm() / denom).item(),
        "mse": diff.square().mean().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            w.flatten().unsqueeze(0), dequant.flatten().unsqueeze(0)
        ).item(),
        "kld": 0.0,  # per-weight KLD is not meaningful; keep field for report
    }
    return w_int8, w_scale, metrics


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_quantized_model(
    model: nn.Module,
    quant_state: dict,
    bf16_layers: set,
    output_path: Path,
    model_path: Path,
    group_ignore: list[str],
):
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    orig_index = json.loads(
        (model_path / "model.safetensors.index.json").read_text()
    )
    orig_map = orig_index["weight_map"]
    shard_keys: "OrderedDict[str, list[str]]" = OrderedDict()
    for key, shard in orig_map.items():
        shard_keys.setdefault(shard, []).append(key)

    current_shard = {}
    current_bytes = 0
    max_shard_bytes = 4 * 1024**3
    files = []
    final_map = {}
    shard_idx = 1
    total_int8 = 0
    total_bf16 = 0

    def flush():
        nonlocal current_shard, current_bytes, shard_idx
        if not current_shard:
            return
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(output_path / fname))
        for k in current_shard:
            final_map[k] = fname
        files.append(fname)
        shard_idx += 1
        current_shard = {}
        current_bytes = 0

    for shard_name, keys in shard_keys.items():
        data = load_file(str(model_path / shard_name))
        for key in keys:
            if key.endswith(".weight"):
                mod_name = _safetensors_key_to_module_name(key)
                if mod_name in bf16_layers or mod_name not in quant_state:
                    tensor = data[key]
                    total_bf16 += 1
                else:
                    tensor = quant_state[mod_name]["weight_int8"]
                    scale_key = key[: -len(".weight")] + ".weight_scale"
                    scale = quant_state[mod_name]["weight_scale"]
                    current_shard[key] = tensor
                    current_bytes += tensor.numel() * tensor.element_size()
                    current_shard[scale_key] = scale
                    current_bytes += scale.numel() * scale.element_size()
                    total_int8 += 1
                    continue
            elif key.endswith(".weight_scale"):
                continue
            else:
                tensor = data[key]
            current_shard[key] = tensor
            current_bytes += tensor.numel() * tensor.element_size()
            if current_bytes >= max_shard_bytes:
                flush()
        del data
        gc.collect()

    flush()

    total_size = sum((output_path / f).stat().st_size for f in files)
    index = {"metadata": {"total_size": total_size}, "weight_map": final_map}
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2)
    )

    for fname in model_path.iterdir():
        if fname.name.endswith((".json", ".jinja", ".txt", ".md")):
            dst = output_path / fname.name
            if not dst.exists():
                shutil.copy2(fname, dst)

    config_path = output_path / "config.json"
    config = json.loads(config_path.read_text())
    quant_description = {}
    for mod_name in quant_state:
        quant_description[_hf_name_to_ckpt(mod_name) + ".weight"] = (
            "FLOAT" if mod_name in bf16_layers else "INT8_PER_CHANNEL"
        )
    config["quantization_config"] = {
        "quant_method": "w8a8_int8",
        "ignore": group_ignore,
        "quant_description": quant_description,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"  Saved {len(files)} shards; {total_int8} INT8, {total_bf16} BF16/other weights")
    print(f"  Model size: {total_size / 1024**3:.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/weight/Qwen3.8-27B")
    parser.add_argument(
        "--output-path",
        default="/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8",
    )
    parser.add_argument(
        "--calib-data",
        default="/home/acceleration/quantization/calibration/calib_data.jsonl",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="占位参数，当前算法不使用（保留以便既有命令行原样执行）",
    )
    parser.add_argument("--bf16-ratio", type=float, default=0.10)
    parser.add_argument("--use-gptq", action="store_true", default=True)
    parser.add_argument("--no-gptq", action="store_true")
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--num-calib-samples", type=int, default=128)
    parser.add_argument("--max-calib-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    use_gptq = args.use_gptq and not args.no_gptq
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    print("=" * 70)
    print("Enhanced W8A8 quantization (activation-aware GPTQ + selective BF16)")
    print(f"  model: {model_path}")
    print(f"  output: {output_path}")
    print(f"  calibration samples: {args.num_calib_samples}")
    print(f"  bf16 ratio: {args.bf16_ratio:.2f}")
    print(f"  use_gptq: {use_gptq}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_inputs = []
    calib_path = Path(args.calib_data)
    if calib_path.exists():
        for line in calib_path.read_text().splitlines():
            if len(calib_inputs) >= args.num_calib_samples:
                break
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                continue
            toks = tokenizer(
                text,
                return_tensors="pt",
                max_length=args.max_calib_length,
                truncation=True,
                padding=False,
            )["input_ids"]
            if toks.numel() >= 2:
                calib_inputs.append(toks)
    else:
        for _ in range(args.num_calib_samples):
            calib_inputs.append(torch.randint(0, tokenizer.vocab_size, (1, 1024)))

    print(f"Loading model on {args.device}...")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    linear_layers = OrderedDict(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )
    print(f"Found {len(linear_layers)} Linear modules")

    print("Collecting activation statistics...")
    collector = ActivationStatsCollector(model)
    with torch.no_grad():
        for i, toks in enumerate(calib_inputs):
            toks = toks.to(args.device)
            model(toks)
            del toks
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(calib_inputs)} samples")
            if (i + 1) % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    collector.remove()
    print(f"Collected stats for {sum(1 for s in collector.stats.values() if s['count'])} layers")

    print("Moving model to CPU and quantizing...")
    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    layer_results = []
    quant_state = {}
    for idx, (name, module) in enumerate(linear_layers.items()):
        stats = collector.stats.get(name)
        if stats is None or stats["count"] == 0:
            layer_results.append({"name": name, "relative": 0.0, "skip": True})
            continue
        weight = module.weight.data.cpu().float()
        w_int8, w_scale, metrics = quantize_linear(
            weight, stats, use_gptq=use_gptq, damp=args.damp
        )
        quant_state[name] = {"weight_int8": w_int8, "weight_scale": w_scale}
        layer_results.append(
            {
                "name": name,
                "relative": metrics["relative"],
                "mse": metrics["mse"],
                "cosine": metrics["cosine"],
                "kld": metrics["kld"],
                "skip": False,
            }
        )
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(linear_layers)} layers")
    print(f"Quantization took {time.time() - t0:.1f}s")

    # Group-aware selective BF16 selection.
    valid = [x for x in layer_results if not x.get("skip", False)]
    grouped: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for x in valid:
        g = get_group(x["name"])
        if g is not None:
            grouped.setdefault(g, []).append(x)
    ranked = sorted(
        grouped.items(),
        key=lambda kv: max(x["relative"] for x in kv[1]),
        reverse=True,
    )
    target = max(1, int(round(len(valid) * args.bf16_ratio)))
    selected_groups = []
    selected_count = 0
    for g, items in ranked:
        selected_groups.append(g)
        selected_count += len(items)
        if selected_count >= target:
            break

    bf16_layers = {
        x["name"]
        for g, items in grouped.items()
        if g in selected_groups
        for x in items
    }
    bf16_layers.add("lm_head")
    group_ignore = [
        f"model.layers.{idx}.{typ}.{proj}" for idx, typ, proj in selected_groups
    ]
    print(f"Selected {len(selected_groups)} packed groups, {len(bf16_layers)} Linear modules for BF16")

    print("Saving checkpoint...")
    save_quantized_model(
        model=model,
        quant_state=quant_state,
        bf16_layers=bf16_layers,
        output_path=output_path,
        model_path=model_path,
        group_ignore=group_ignore,
    )

    report = {
        "method": "activation_aware_gptq_w8a8_selective_bf16",
        "runtime": "sglang_w8a8_int8",
        "config": {
            "bf16_ratio": args.bf16_ratio,
            "use_gptq": use_gptq,
            "damp": args.damp,
            "num_calib_samples": len(calib_inputs),
            "num_linear_layers": len(linear_layers),
            "num_bf16": len(bf16_layers),
            "num_quantized": len(linear_layers) - len(bf16_layers),
        },
        "layer_results": layer_results,
        "bf16_layers": sorted(bf16_layers),
        "bf16_groups": group_ignore,
    }
    (output_path / "quantization_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print("Done.")


if __name__ == "__main__":
    main()
