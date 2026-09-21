#!/usr/bin/env python3
"""
DEPRECATED LEGACY SCRIPT.

This file contains the earlier SmoothQuant implementation that produced an
invalid SGLang W8A8 checkpoint (fast but garbage output).  Use ``quantize.py``
instead; it is the corrected deployment-ready pipeline.
"""
"""
Enhanced W8A8 Quantization for Qwen3.5-27B.
Memory-efficient: GPU only for calibration forward pass, quantization on CPU.
"""

import json
import os
import sys
import time
import gc
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ============================================================
# SmoothQuant + Quantization (all CPU-friendly)
# ============================================================

def smoothquant_quantize(
    weight_bf16_cpu: torch.Tensor,  # [out, in] CPU
    act_cpu: torch.Tensor,          # [n, in] CPU
    alpha: float = 0.5,
    use_gptq: bool = True,
    damp: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Quantize a single layer: SmoothQuant + optional GPTQ rounding.
    All computation on CPU to avoid GPU OOM.
    Returns: (weight_int8, weight_scale, metrics)
    """
    out_feat, in_feat = weight_bf16_cpu.shape

    # SmoothQuant scaling
    w_abs_max = weight_bf16_cpu.abs().max(dim=0, keepdim=True).values.clamp(min=1e-5)
    a_abs_max = act_cpu.abs().max(dim=0, keepdim=True).values.clamp(min=1e-5)
    s = torch.pow(a_abs_max, alpha) / torch.pow(w_abs_max, 1.0 - alpha)
    s = s.clamp(min=1e-5)

    # Scale weight and activation
    w_scaled = weight_bf16_cpu * s
    act_scaled = act_cpu / s

    # Per-output-channel quantization scale
    w_scale = w_scaled.abs().max(dim=1, keepdim=True).values / 127.0
    w_scale = w_scale.clamp(min=1e-5)

    # Initial quantization
    w_int8 = (w_scaled / w_scale).round().clamp(-128, 127).to(torch.int8)

    # GPTQ optimization (vectorized, no Python loop over output channels)
    if use_gptq and out_feat > 0 and in_feat > 0:
        X = act_cpu.float()
        n = X.shape[0]
        # Input covariance diagonal
        H_diag = (X * X).mean(dim=0)  # [in]
        H_diag += damp * H_diag.mean() + 1e-6

        # Vectorized rounding optimization
        # w_scaled: [out, in], w_scale: [out, 1]
        w_float = w_scaled.float()                      # [out, in]
        q_vals = w_float / w_scale                       # [out, in]
        q_floor = q_vals.floor().clamp(-128, 127)
        q_ceil = q_vals.ceil().clamp(-128, 127)

        # Reconstruction error for each choice (broadcast over output dim)
        err_floor = (w_float - q_floor * w_scale) ** 2  # [out, in]
        err_ceil = (w_float - q_ceil * w_scale) ** 2    # [out, in]

        # Hessian-weighted cost (H_diag broadcast over output dim)
        cost_floor = err_floor * H_diag                  # [out, in]
        cost_ceil = err_ceil * H_diag                    # [out, in]

        # Choose optimal rounding (vectorized over all output channels)
        choose_ceil = cost_ceil < cost_floor
        q_opt = torch.where(choose_ceil, q_ceil, q_floor)
        w_int8 = q_opt.to(torch.int8)

        del H_diag, X, q_vals, q_floor, q_ceil, err_floor, err_ceil, cost_floor, cost_ceil, choose_ceil, q_opt
        gc.collect()

    # Compute error metrics on CPU
    X = act_cpu.float()
    out_bf16 = X @ w_scaled.float().T
    out_int8 = X @ (w_int8.float() * w_scale).T
    diff = out_bf16 - out_int8

    mse = (diff ** 2).mean().item()
    cos = F.cosine_similarity(out_bf16.flatten().unsqueeze(0),
                               out_int8.flatten().unsqueeze(0)).item()
    rel = (diff.norm() / (out_bf16.norm() + 1e-8)).item()

    # KLD (safe)
    try:
        p = F.softmax(out_bf16, dim=-1)
        q = F.softmax(out_int8, dim=-1)
        kld = F.kl_div(q.log(), p, reduction='batchmean').item()
        if math.isnan(kld) or math.isinf(kld):
            kld = 0.0
    except Exception:
        kld = 0.0

    del out_bf16, out_int8, diff, X
    gc.collect()

    return w_int8, w_scale, {'mse': mse, 'cosine': cos, 'relative': rel, 'kld': kld}


# ============================================================
# Hook collector
# ============================================================

class HookCollector:
    def __init__(self, model):
        self.model = model
        self.inputs = {}
        self._hooks = []

    def register(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                h = mod.register_forward_pre_hook(self._make(name))
                self._hooks.append(h)

    def _make(self, name):
        def fn(mod, inp):
            if inp[0] is not None:
                # Detach and move to CPU immediately to save GPU memory
                self.inputs[name] = inp[0].detach().cpu()
        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def pop(self, name):
        return self.inputs.pop(name, None)


# ============================================================
# Save quantized model
# ============================================================

def _safetensors_key_to_module_name(st_key: str) -> str:
    """Convert a safetensors key to the named_modules() name.

    Examples:
        model.language_model.layers.0.linear_attn.in_proj_qkv.weight
            -> model.layers.0.linear_attn.in_proj_qkv
        model.language_model.layers.11.self_attn.q_proj.weight
            -> model.layers.11.self_attn.q_proj
        lm_head.weight -> lm_head
        model.language_model.embed_tokens.weight -> model.embed_tokens
    """
    base = st_key
    for suffix in ('.weight_scale', '.weight'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if base.startswith("model.language_model."):
        return "model." + base[len("model.language_model."):]
    return base


def save_quantized_model(
    model: nn.Module,
    quant_state: Dict,
    bf16_layers: set,
    output_path: str,
    model_path: str,
):
    """Save quantized model in SGLang w8a8_int8 format.

    Key mapping between named_modules() and safetensors:
      named_modules: model.layers.X.linear_attn.in_proj_qkv
      safetensors:   model.language_model.layers.X.linear_attn.in_proj_qkv.weight
    The safetensors keys have an extra "language_model." prefix.

    SGLang load_weights() strips that prefix and uses stacked_params_mapping
    to pack individual weights (q_proj+k_proj+v_proj -> qkv_proj, etc).
    We must save individual weight names so SGLang can pack them.
    We must also save weight_scale alongside each quantized weight.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file

    print("  Building state dict (fixed key mapping)...")

    # Load original model's index to get safetensors keys and shard mapping
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        orig_index = json.load(f)
    weight_map = orig_index.get('weight_map', {})

    # Group keys by shard
    shard_keys = {}
    for key, shard in weight_map.items():
        if shard not in shard_keys:
            shard_keys[shard] = []
        shard_keys[shard].append(key)

    # Process each shard: replace .weight with INT8, add .weight_scale
    shard_idx = 1
    current_shard = {}
    current_size = 0
    MAX_SHARD = 4 * 1024**3
    files = []
    final_weight_map = {}
    total_int8 = 0
    total_bf16 = 0

    for shard_name in sorted(shard_keys.keys()):
        shard_path = os.path.join(model_path, shard_name)
        print(f"    Loading {shard_name}...")
        shard_data = load_file(shard_path)

        for key in shard_keys[shard_name]:
            # Convert safetensors key to named_modules() name
            mod_name = _safetensors_key_to_module_name(key)

            if key.endswith('.weight'):
                if mod_name in bf16_layers:
                    # Selective BF16: copy the original weight unchanged and do
                    # not add a weight_scale tensor.
                    current_shard[key] = shard_data[key]
                    current_size += shard_data[key].nelement() * shard_data[key].element_size()
                    total_bf16 += 1
                elif mod_name in quant_state:
                    # Quantized layer -> save INT8 weight
                    current_shard[key] = quant_state[mod_name]['weight_int8']
                    current_size += quant_state[mod_name]['weight_int8'].nelement()
                    total_int8 += 1

                    # Add weight_scale for this layer
                    scale_key = key[:-len('.weight')] + '.weight_scale'
                    current_shard[scale_key] = quant_state[mod_name]['weight_scale']
                    current_size += quant_state[mod_name]['weight_scale'].nelement() * 4
                else:
                    # Non-linear or lm_head -> keep original BF16
                    current_shard[key] = shard_data[key]
                    current_size += shard_data[key].nelement() * shard_data[key].element_size()
                    total_bf16 += 1
            elif key.endswith('.weight_scale'):
                # Original model has no weight_scale.  If the layer was selected
                # for BF16 we must not leave a stale scale behind.
                continue
            else:
                # Non-weight params (layernorm, embed, conv1d, etc.) -> copy as-is
                current_shard[key] = shard_data[key]
                current_size += shard_data[key].nelement() * shard_data[key].element_size()

        del shard_data
        gc.collect()

        # Flush shard if large enough
        if current_size >= MAX_SHARD and current_shard:
            fname = f"model-{shard_idx:05d}.safetensors"
            save_file(current_shard, str(out_dir / fname))
            for k in current_shard:
                final_weight_map[k] = fname
            files.append(fname)
            shard_idx += 1
            current_shard = {}
            current_size = 0

    # Write remaining keys as final shard(s)
    if current_shard:
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(out_dir / fname))
        for k in current_shard:
            final_weight_map[k] = fname
        files.append(fname)

    # Write index
    total_size = sum(os.path.getsize(str(out_dir / f)) for f in files)
    index = {"metadata": {"total_size": total_size}, "weight_map": final_weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Copy config.json (without quantization_config that might confuse SGLang)
    config_src = os.path.join(model_path, "config.json")
    config_dst = out_dir / "config.json"
    shutil.copy2(config_src, config_dst)

    # Copy other required files
    for fname in os.listdir(model_path):
        if fname.endswith(('.json', '.jinja', '.txt', '.md', '.py')):
            src = os.path.join(model_path, fname)
            dst = out_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)

    print(f"  Saved {len(files)} shards: {total_int8} INT8 weights, "
          f"{total_bf16} BF16/other weights")


# ============================================================
# Main
# ============================================================

def quantize_model(
    model_path: str,
    output_path: str,
    calib_data_path: str,
    alpha: float = 0.5,
    bf16_ratio: float = 0.10,
    use_gptq: bool = True,
    damp: float = 0.01,
    max_calib_batches: int = 4,
    num_calib_samples: int = 128,
    device: str = 'cuda:0',
):
    print("=" * 60)
    print("Enhanced W8A8 Quantization (Memory-Efficient)")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Alpha: {alpha}, BF16 ratio: {bf16_ratio}, GPTQ: {use_gptq}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load calibration data
    print("\nLoading calibration data...")
    calib_tensors = []
    if os.path.exists(calib_data_path):
        with open(calib_data_path) as f:
            for line in f:
                if len(calib_tensors) >= num_calib_samples:
                    break
                item = json.loads(line)
                text = item.get('text', '')
                if text:
                    toks = tokenizer(text, return_tensors="pt", max_length=2048,
                                     truncation=True, padding=False)
                    calib_tensors.append(toks['input_ids'])
    else:
        print(f"  WARNING: {calib_data_path} not found")
        for _ in range(num_calib_samples):
            calib_tensors.append(torch.randint(0, tokenizer.vocab_size, (1, 1024)))
    print(f"  {len(calib_tensors)} calibration samples")

    # Load model
    print("\nLoading model in BF16...")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on {device}")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB allocated")

    # Get linear layers
    linear_layers = OrderedDict()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_layers[name] = mod
    print(f"  {len(linear_layers)} linear layers")

    # Step 1: Collect activations
    print("\nStep 1: Collecting activations (GPU forward pass)...")
    collector = HookCollector(model)
    collector.register()

    with torch.no_grad():
        for i, batch in enumerate(calib_tensors[:max_calib_batches]):
            batch = batch.to(device)
            try:
                model(batch)
            except Exception as e:
                print(f"  Batch {i} failed: {e}")
            del batch
            if i % 2 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"  Collected {len(collector.inputs)} layer activations")
    print(f"  GPU memory after calibration: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Free model from GPU (no longer needed for GPU computation)
    # We'll keep it for saving later, but move to CPU
    print("  Moving model to CPU to free GPU for quantization...")
    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory after CPU move: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Step 2: Quantize on CPU
    print("\nStep 2: Quantizing layers (CPU)...")
    t_start = time.time()
    layer_results = []
    quant_state = {}

    for idx, name in enumerate(linear_layers):
        act = collector.pop(name)
        if act is None:
            print(f"  [{idx+1}/{len(linear_layers)}] {name}: NO ACT, skip")
            layer_results.append({'name': name, 'kld': 0, 'cosine': 1.0, 'skip': True})
            continue

        # Flatten activation
        if act.dim() > 2:
            act = act.reshape(-1, act.shape[-1])

        weight = linear_layers[name].weight.data.cpu().float()

        # Quantize on CPU
        w_int8, w_scale, metrics = smoothquant_quantize(
            weight, act, alpha=alpha, use_gptq=use_gptq, damp=damp
        )

        quant_state[name] = {
            'weight_int8': w_int8,
            'weight_scale': w_scale,
        }
        layer_results.append({
            'name': name,
            'kld': metrics['kld'],
            'cosine': metrics['cosine'],
            'relative': metrics['relative'],
            'mse': metrics['mse'],
            'skip': False,
        })

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(linear_layers) - idx - 1) / rate
            print(f"  [{idx+1}/{len(linear_layers)}] {name}: "
                  f"KLD={metrics['kld']:.6f} cos={metrics['cosine']:.4f} "
                  f"[{rate:.1f} layers/s, ETA {eta:.0f}s]")

        # Free memory
        del act, weight, w_int8, w_scale
        if idx % 20 == 0:
            gc.collect()

    elapsed = time.time() - t_start
    print(f"  Quantization done in {elapsed:.1f}s ({len(linear_layers)/elapsed:.1f} layers/s)")

    # Step 3: Select BF16 layers.  SGLang fuses several checkpoint weights into
    # one runtime Linear (q/k/v, gate/up, in_proj_qkv/z, in_proj_a/b), so the
    # selection must be made at the packed-group level.
    print("\nStep 3: Selecting BF16 layers...")
    quantized_results = [r for r in layer_results if not r.get('skip', False)]
    group_of_name = None

    def get_group(name: str):
        """Return a stable SGLang packed-group key for an HF Linear name."""
        import re

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

    grouped: Dict[Tuple, List[dict]] = {}
    for r in quantized_results:
        g = get_group(r["name"])
        if g is not None:
            grouped.setdefault(g, []).append(r)

    ranked_groups = sorted(
        grouped.items(),
        key=lambda kv: max(item.get("relative", 0.0) for item in kv[1]),
        reverse=True,
    )
    total_components = sum(len(items) for _, items in grouped.items())
    target_components = max(1, int(round(total_components * bf16_ratio)))

    selected_groups = []
    selected_components = 0
    for g, items in ranked_groups:
        selected_groups.append(g)
        selected_components += len(items)
        if selected_components >= target_components:
            break

    bf16_layers = {
        item["name"]
        for g, items in grouped.items()
        if g in selected_groups
        for item in items
    }
    # SGLang's W8A8 runtime keeps lm_head in BF16.
    bf16_layers.add("lm_head")

    print(f"  Keeping {len(bf16_layers)}/{len(quantized_results) + 1} layers in BF16")
    for name in sorted(bf16_layers):
        if name != "lm_head":
            print(f"    BF16: {name}")

    # Print statistics
    all_klds = [r['kld'] for r in quantized_results if r['kld'] > 0]
    if all_klds:
        print(f"\n  KLD stats: min={min(all_klds):.8f} max={max(all_klds):.6f} "
              f"mean={sum(all_klds)/len(all_klds):.6f}")

    # Step 4: Save
    print("\nStep 4: Saving quantized model...")
    save_quantized_model(
        model=model,
        quant_state=quant_state,
        bf16_layers=bf16_layers,
        output_path=output_path,
        model_path=model_path,
    )

    # Save report
    report = {
        'config': {
            'alpha': alpha, 'bf16_ratio': bf16_ratio,
            'use_gptq': use_gptq, 'damp': damp,
            'num_linear_layers': len(linear_layers),
            'num_quantized': len(quantized_results) - len(bf16_layers & {r['name'] for r in quantized_results}),
            'num_bf16': len(bf16_layers),
        },
        'layer_results': layer_results,
        'bf16_layers': list(bf16_layers),
    }
    report_path = os.path.join(output_path, "quantization_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")

    print("\nDone!")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/weight/Qwen3.8-27B")
    parser.add_argument("--output-path",
                        default="/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8")
    parser.add_argument("--calib-data",
                        default="/home/acceleration/quantization/calibration/calib_data.jsonl")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--bf16-ratio", type=float, default=0.10)
    parser.add_argument("--use-gptq", action="store_true", default=True)
    parser.add_argument("--no-gptq", action="store_true")
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--max-calib-batches", type=int, default=4)
    parser.add_argument("--num-calib-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.no_gptq:
        args.use_gptq = False

    quantize_model(
        model_path=args.model_path,
        output_path=args.output_path,
        calib_data_path=args.calib_data,
        alpha=args.alpha,
        bf16_ratio=args.bf16_ratio,
        use_gptq=args.use_gptq,
        damp=args.damp,
        max_calib_batches=args.max_calib_batches,
        num_calib_samples=args.num_calib_samples,
        device=args.device,
    )
