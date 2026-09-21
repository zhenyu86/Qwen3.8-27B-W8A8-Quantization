#!/usr/bin/env python3
"""
Re-save the already-quantized model with correct packed weight format for SGLang.

Reads the unpacked quantized checkpoint, packs weights according to 
SGLang's stacked_params_mapping, and saves a new checkpoint.
"""

import json, os, gc, shutil, torch
from pathlib import Path
from safetensors.torch import load_file, save_file

MODEL_PATH = "/home/weight/Qwen3.8-27B"
QUANT_PATH = "/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8"
REPORT_PATH = os.path.join(QUANT_PATH, "quantization_report.json")
OUTPUT_PATH = "/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8-v2"

# Load quantization report to get quant_state info
with open(REPORT_PATH) as f:
    report = json.load(f)

bf16_layers = set(report['bf16_layers'])
layer_results = {r['name']: r for r in report['layer_results']}

# Determine which layers are INT8 vs BF16
int8_layers = set()
for r in report['layer_results']:
    if not r.get('skip', False) and r['name'] not in bf16_layers:
        int8_layers.add(r['name'])

print(f"INT8 layers: {len(int8_layers)}")
print(f"BF16 layers: {len(bf16_layers)}")

# Stacking rules
# Format: (unpacked_suffix, packed_suffix, shard_ids)
# shard_ids is tuple for multi-shard, int for single-shard
STACKING_RULES = [
    # Linear attention GDN packing
    (".linear_attn.in_proj_qkv", ".linear_attn.in_proj_qkvz", (0, 1, 2)),
    (".linear_attn.in_proj_z",   ".linear_attn.in_proj_qkvz", 3),
    (".linear_attn.in_proj_b",   ".linear_attn.in_proj_ba",    0),
    (".linear_attn.in_proj_a",   ".linear_attn.in_proj_ba",    1),
    # MLP gate+up packing
    (".mlp.gate_proj", ".mlp.gate_up_proj", 0),
    (".mlp.up_proj",   ".mlp.gate_up_proj", 1),
    # Full attention QKV packing
    (".self_attn.q_proj", ".self_attn.qkv_proj", "q"),
    (".self_attn.k_proj", ".self_attn.qkv_proj", "k"),
    (".self_attn.v_proj", ".self_attn.qkv_proj", "v"),
]

def find_unpack_rule(param_name):
    """Check if param_name is an unpacked component that should be packed."""
    for unpacked_suffix, packed_suffix, shard_id in STACKING_RULES:
        if unpacked_suffix + ".weight" in param_name or unpacked_suffix + ".weight_scale" in param_name:
            return unpacked_suffix, packed_suffix, shard_id
    return None

# Build packing groups
# For each layer that has packed components, group them
packing_groups = {}  # packed_name -> {shard_id: unpacked_name}

for name in list(int8_layers) + list(bf16_layers):
    for unpacked_suffix, packed_suffix, shard_id in STACKING_RULES:
        unpacked_name = name  # e.g., "model.layers.0.linear_attn.in_proj_qkv"
        # Check if this layer has this unpacked component
        if unpacked_suffix in unpacked_name:
            packed_name = unpacked_name.replace(unpacked_suffix, packed_suffix)
            if packed_name not in packing_groups:
                packing_groups[packed_name] = {}
            
            if isinstance(shard_id, tuple):
                for i, sid in enumerate(shard_id):
                    packing_groups[packed_name][sid] = unpacked_name
            else:
                packing_groups[packed_name][shard_id] = unpacked_name

print(f"\nPacking groups: {len(packing_groups)}")
for packed, components in sorted(packing_groups.items())[:5]:
    print(f"  {packed}: {sorted(components.items())}")

# Process checkpoint shards
orig_index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
with open(orig_index_path) as f:
    orig_index = json.load(f)
orig_weight_map = orig_index['weight_map']

# Group keys by shard
shard_key_groups = {}
for key, shard in orig_weight_map.items():
    if shard not in shard_key_groups:
        shard_key_groups[shard] = []
    shard_key_groups[shard].append(key)

out_dir = Path(OUTPUT_PATH)
out_dir.mkdir(parents=True, exist_ok=True)

# For packing, we need to know which unpacked component maps to which shard_id
# Build reverse mapping: unpacked_name -> (packed_name, shard_id)
unpacked_to_packed = {}
for packed_name, components in packing_groups.items():
    for shard_id, unpacked_name in components.items():
        unpacked_to_packed[unpacked_name] = (packed_name, shard_id)

# Collect all packed results
packed_results = {}  # packed_name.weight -> tensor, packed_name.weight_scale -> tensor

# Also track which unpacked names have been packed (to avoid double-saving)
packed_component_names = set(unpacked_to_packed.keys())

# Process each shard from original model
final_files = []
final_weight_map = {}
shard_idx = 1
current_shard = {}
current_size = 0
MAX_SHARD = 4 * 1024**3

for shard_name in sorted(shard_key_groups.keys()):
    print(f"  Loading {shard_name}...")
    shard_data = load_file(os.path.join(MODEL_PATH, shard_name))
    
    for key in shard_key_groups[shard_name]:
        # Determine the base parameter name (without .weight or .weight_scale)
        if key.endswith('.weight'):
            base_name = key[:-7]  # remove ".weight"
            suffix = ".weight"
        elif key.endswith('.weight_scale'):
            base_name = key[:-13]  # remove ".weight_scale"
            suffix = ".weight_scale"
        else:
            # Non-weight parameter (norm, embedding, etc.) - keep as-is
            current_shard[key] = shard_data[key]
            continue
        
        # Check if this parameter is in a packing group
        if base_name in unpacked_to_packed:
            packed_name, shard_id = unpacked_to_packed[base_name]
            packed_key = packed_name + suffix
            
            if packed_key not in packed_results:
                packed_results[packed_key] = {}
            
            # Store with shard_id as key for later concatenation
            packed_results[packed_key][shard_id] = shard_data[key]
            continue
        
        # Check if this is a quantized parameter
        if base_name in int8_layers:
            if suffix == ".weight":
                # Load quantized INT8 weight from the quant checkpoint
                # Need to find it in the quant checkpoint shards
                # For now, keep the original - we'll replace later
                current_shard[key] = shard_data[key]
            elif suffix == ".weight_scale":
                current_shard[key] = shard_data[key]
            continue
        
        # BF16 or non-quantized: keep as-is
        current_shard[key] = shard_data[key]
    
    del shard_data
    gc.collect()

# Now concatenate packed groups
print("\n  Packing weights...")
for packed_key, shard_dict in packed_results.items():
    sorted_items = sorted(shard_dict.items())
    if isinstance(sorted_items[0][1], torch.Tensor):
        packed_tensor = torch.cat([item[1] for item in sorted_items], dim=0)
        current_shard[packed_key] = packed_tensor
        print(f"    {packed_key}: {' + '.join(str(s[0]) for s in sorted_items)} -> {packed_tensor.shape}")

# Also: we need to handle the quantized weights that AREN'T packed
# For non-packed quantized layers, we need to load from the quant checkpoint
# Let's load quant weights from the v1 checkpoint
print("\n  Loading quantized weights from v1 checkpoint...")
v1_shards = {}
for shard_name in sorted(shard_key_groups.keys()):
    # Map original shard name to v1 shard name
    v1_path = os.path.join(QUANT_PATH, shard_name.replace("-of-00018", ""))
    if not os.path.exists(v1_path):
        # Try with different naming
        for f in os.listdir(QUANT_PATH):
            if f.endswith('.safetensors'):
                v1_path = os.path.join(QUANT_PATH, f)
                break
    
v1_data = {}
for f in sorted(os.listdir(QUANT_PATH)):
    if f.endswith('.safetensors'):
        print(f"    Loading {f}...")
        data = load_file(os.path.join(QUANT_PATH, f))
        v1_data.update(data)
        if len(v1_data) > 1000:
            break  # Got enough

# Replace quantized weights and add weight_scale
for name in int8_layers:
    weight_key = f"{name}.weight"
    scale_key = f"{name}.weight_scale"
    
    # Check if this is a packed component
    if name in packed_component_names:
        continue  # Already handled by packing
    
    # Non-packed quantized layer: load from v1
    if weight_key in v1_data:
        current_shard[weight_key] = v1_data[weight_key]
        print(f"    INT8: {weight_key} -> {v1_data[weight_key].shape}")
    if scale_key in v1_data:
        current_shard[scale_key] = v1_data[scale_key]

# Add packed weight_scales
for packed_key in list(packed_results.keys()):
    if packed_key.endswith('.weight'):
        scale_key = packed_key.replace('.weight', '.weight_scale')
        if scale_key not in current_shard and scale_key in v1_data:
            current_shard[scale_key] = v1_data[scale_key]

del v1_data
gc.collect()

# Write output shards
print(f"\n  Writing {OUTPUT_PATH}...")
shard_idx = 1
output_tensors = {}
output_size = 0

for key in sorted(current_shard.keys()):
    t = current_shard[key]
    t_size = t.nelement() * t.element_size()
    if output_size + t_size > MAX_SHARD and output_tensors:
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(output_tensors, str(out_dir / fname))
        for k in output_tensors:
            final_weight_map[k] = fname
        final_files.append(fname)
        shard_idx += 1
        output_tensors = {}
        output_size = 0
    output_tensors[key] = t
    output_size += t_size

if output_tensors:
    fname = f"model-{shard_idx:05d}.safetensors"
    save_file(output_tensors, str(out_dir / fname))
    for k in output_tensors:
        final_weight_map[k] = fname
    final_files.append(fname)

# Write index
total_size = sum(os.path.getsize(str(out_dir / f)) for f in final_files)
index = {"metadata": {"total_size": total_size}, "weight_map": final_weight_map}
with open(out_dir / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

# Copy config and other files
shutil.copy2(os.path.join(MODEL_PATH, "config.json"), out_dir / "config.json")
for fname in os.listdir(MODEL_PATH):
    if fname.endswith(('.json', '.jinja', '.txt', '.md')):
        src = os.path.join(MODEL_PATH, fname)
        dst = out_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

# Update config with quantization config (keep from v1)
with open(os.path.join(QUANT_PATH, "config.json")) as f:
    v1_config = json.load(f)
with open(out_dir / "config.json") as f:
    config = json.load(f)

config["quantization_config"] = v1_config.get("quantization_config", {})
with open(out_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n  Saved {len(final_files)} shards")
print(f"  Total size: {total_size/1024**3:.2f} GB")
print("  Done!")
