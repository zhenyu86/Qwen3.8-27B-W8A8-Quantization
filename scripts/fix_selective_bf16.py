#!/usr/bin/env python3
"""
Repair the already-quantized checkpoint so that the selective-BF16 policy is
actually applied.

The previous run saved every quantizable Linear as INT8, even layers that the
report selected for BF16.  This script:

  1. Reads the existing quantization report.
  2. Builds SGLang-compatible packed groups (qkv/gate-up/in_proj_qkvz/in_proj_ba).
  3. Re-ranks groups by activation-aware relative error and selects a target
     fraction of individual Linear modules.
  4. Copies the original BF16 weights back for selected layers, removes their
     INT8 weight_scale tensors, and updates the checkpoint index.
  5. Writes a proper ``quantization_config`` into config.json so SGLang's
     ``w8a8_int8`` path skips those BF16 modules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

from safetensors.torch import load_file, save_file


def group_of_name(name: str):
    """Map an HF-style individual Linear name to an SGLang packed group."""
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
        return (idx, typ, proj)

    if typ == "self_attn":
        if proj in ("q_proj", "k_proj", "v_proj"):
            return (idx, "self_attn", "qkv_proj")
        if proj == "o_proj":
            return (idx, "self_attn", "o_proj")
        return (idx, typ, proj)

    if typ == "mlp":
        if proj in ("gate_proj", "up_proj"):
            return (idx, "mlp", "gate_up_proj")
        if proj == "down_proj":
            return (idx, "mlp", "down_proj")
        return (idx, typ, proj)

    return (idx, typ, proj)


def hf_name_to_ckpt(name: str) -> str:
    if name.startswith("model."):
        return "model.language_model." + name[len("model.") :]
    return name


def select_bf16_layers(report: dict, bf16_ratio: float) -> tuple[set[str], list[dict]]:
    """Select sensitive individual Linear modules with packed-group consistency."""
    entries = [
        x
        for x in report.get("layer_results", [])
        if not x.get("skip", False)
        and re.match(r"model\.layers\.\d+", x.get("name", ""))
    ]
    groups: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for x in entries:
        g = group_of_name(x["name"])
        if g is None:
            continue
        groups.setdefault(g, []).append(x)

    ranked_groups = sorted(
        groups.items(),
        key=lambda kv: max(item.get("relative", 0.0) for item in kv[1]),
        reverse=True,
    )

    total = sum(len(items) for _, items in groups.items())
    target_components = max(1, int(round(total * bf16_ratio)))
    selected_groups: list[tuple] = []
    selected_components = 0
    for g, items in ranked_groups:
        selected_groups.append(g)
        selected_components += len(items)
        if selected_components >= target_components:
            break

    bf16_layers = {
        x["name"]
        for g, items in groups.items()
        if g in selected_groups
        for x in items
    }
    # SGLang's w8a8 path does not quantize lm_head. Keep it BF16 in both the
    # checkpoint and the report.
    bf16_layers.add("lm_head")

    group_report = [
        {
            "group": f"model.layers.{idx}.{typ}.{proj}",
            "layers": sorted(x["name"] for x in groups[g]),
            "max_relative_error": max(x.get("relative", 0.0) for x in groups[g]),
            "max_mse": max(x.get("mse", 0.0) for x in groups[g]),
        }
        for idx, typ, proj in sorted(selected_groups)
    ]
    return bf16_layers, group_report


def sglang_ignore_prefixes(group_report: list[dict]) -> list[str]:
    """Return exact SGLang module prefixes to keep in BF16."""
    ignore = []
    for g in group_report:
        # Example group string: model.layers.7.mlp.gate_up_proj
        ignore.append(g["group"])
    return ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-model",
        default="/home/weight/Qwen3.8-27B",
    )
    parser.add_argument(
        "--quant-model",
        default="/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8",
    )
    parser.add_argument("--bf16-ratio", type=float, default=0.10)
    args = parser.parse_args()

    orig_dir = Path(args.original_model)
    quant_dir = Path(args.quant_model)
    report_path = quant_dir / "quantization_report.json"
    index_path = quant_dir / "model.safetensors.index.json"
    config_path = quant_dir / "config.json"

    report = json.loads(report_path.read_text())
    bf16_layers, group_report = select_bf16_layers(report, args.bf16_ratio)

    print(f"Selected {len(group_report)} SGLang packed groups, "
          f"{len(bf16_layers)} individual HF Linear modules for BF16.")

    # Determine selected checkpoint weight keys.
    selected_weights = {hf_name_to_ckpt(name) + ".weight" for name in bf16_layers}
    selected_scales = {key + "_scale" for key in selected_weights}

    orig_index = json.loads(
        (orig_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]

    # Load only the original tensors that will be restored.
    needed_shards = sorted({orig_index[k] for k in selected_weights if k in orig_index})
    original_tensors = {}
    for shard in needed_shards:
        data = load_file(str(orig_dir / shard))
        for key in selected_weights:
            if orig_index.get(key) == shard and key in data:
                original_tensors[key] = data[key]

    missing = selected_weights - set(original_tensors)
    if missing:
        raise RuntimeError(f"Missing original weights: {sorted(missing)}")

    index = json.loads(index_path.read_text())
    old_weight_map = index["weight_map"]
    current_shards = sorted(set(old_weight_map.values()))

    new_weight_map = dict(old_weight_map)
    for key in selected_scales:
        new_weight_map.pop(key, None)

    for shard in current_shards:
        shard_path = quant_dir / shard
        data = load_file(str(shard_path))
        changed = False

        for key in list(data.keys()):
            if key in selected_scales:
                del data[key]
                changed = True
            elif key in original_tensors:
                data[key] = original_tensors[key]
                changed = True

        if changed:
            print(f"  Updating {shard}")
            save_file(data, str(shard_path))

    # Recalculate index metadata from on-disk shard sizes.
    total_size = sum((quant_dir / shard).stat().st_size for shard in current_shards)
    index["metadata"] = {"total_size": total_size}
    index["weight_map"] = new_weight_map
    index_path.write_text(json.dumps(index, indent=2))

    # Write quantization_config understood by SGLang W8A8Int8Config.
    config = json.loads(config_path.read_text())
    quant_description = {}
    all_linear = {x["name"] for x in report.get("layer_results", [])}
    all_linear.add("lm_head")
    for name in all_linear:
        quant_description[hf_name_to_ckpt(name) + ".weight"] = (
            "FLOAT" if name in bf16_layers else "INT8_PER_CHANNEL"
        )
    config["quantization_config"] = {
        "quant_method": "w8a8_int8",
        "ignore": sglang_ignore_prefixes(group_report),
        "quant_description": quant_description,
    }
    config_path.write_text(json.dumps(config, indent=2))

    # Update report with the corrected policy.
    report["bf16_layers"] = sorted(bf16_layers)
    report["bf16_groups"] = group_report
    report["config"]["bf16_ratio"] = args.bf16_ratio
    report["config"]["num_linear_layers"] = len(all_linear)
    report["config"]["num_bf16"] = len(bf16_layers)
    report["config"]["num_quantized"] = len(all_linear) - len(bf16_layers)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote config with {len(sglang_ignore_prefixes(group_report))} ignored modules.")
    print(f"Updated report: {len(bf16_layers)} BF16, "
          f"{len(all_linear) - len(bf16_layers)} INT8 linear modules.")


if __name__ == "__main__":
    main()
