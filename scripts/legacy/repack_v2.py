#!/usr/bin/env python3
"""
Repack quantized checkpoint to match SGLang's weight loading format.

SGLang's Qwen3_5ForCausalLM.load_weights() uses stacked_params_mapping to pack
individual weights into fused parameters BEFORE calling weight_loader. Our v1
checkpoint saved weights with individual names, so the loaders never found them.

This script:
1. Reads v1 checkpoint (unpacked quantized weights)
2. Packs them according to stacked_params_mapping
3. Saves v2 checkpoint with correct packed names
"""

import json, os, gc, shutil, torch
from pathlib import Path
from safetensors.torch import load_file, save_file

V1_PATH = "/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8"
ORIG_PATH = "/home/weight/Qwen3.8-27B"
V2_PATH = "/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8-v2"

# Load report to know which layers are INT8 vs BF16
with open(os.path.join(V1_PATH, "quantization_report.json")) as f:
    report = json.load(f)
bf16_layers = set(report['bf16_layers'])
int8_layers = set()
for r in report['layer_results']:
    if not r.get('skip', False) and r['name'] not in bf16_layers:
        int8_layers.add(r['name'])
print(f"INT8: {len(int8_layers)}, BF16: {len(bf16_layers)}")

# Load v1 checkpoint
print("Loading v1 checkpoint...")
v1_tensors = {}
for f in sorted(os.listdir(V1_PATH)):
    if f.endswith('.safetensors'):
        print(f"  {f}")
        v1_tensors.update(load_file(os.path.join(V1_PATH, f)))
print(f"  {len(v1_tensors)} tensors loaded")

# Load original model index to know the key structure
with open(os.path.join(ORIG_PATH, "model.safetensors.index.json")) as f:
    orig_index = json.load(f)
orig_weight_map = orig_index['weight_map']

# Stacking rules from Qwen3_5ForCausalLM.load_weights():
#   checkpoint_name -> packed_param_name, shard_id
# After prefix stripping, checkpoint uses "model.layers.X" format
STACKING = [
    # GatedDeltaNet
    (".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_qkvz.weight", (0,1,2)),
    (".linear_attn.in_proj_z.weight",   ".linear_attn.in_proj_qkvz.weight", 3),
    (".linear_attn.in_proj_b.weight",   ".linear_attn.in_proj_ba.weight",    0),
    (".linear_attn.in_proj_a.weight",   ".linear_attn.in_proj_ba.weight",    1),
    (".linear_attn.in_proj_qkv.weight_scale", ".linear_attn.in_proj_qkvz.weight_scale", (0,1,2)),
    (".linear_attn.in_proj_z.weight_scale",   ".linear_attn.in_proj_qkvz.weight_scale", 3),
    (".linear_attn.in_proj_b.weight_scale",   ".linear_attn.in_proj_ba.weight_scale",    0),
    (".linear_attn.in_proj_a.weight_scale",   ".linear_attn.in_proj_ba.weight_scale",    1),
    # MLP
    (".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", 0),
    (".mlp.up_proj.weight",   ".mlp.gate_up_proj.weight", 1),
    (".mlp.gate_proj.weight_scale", ".mlp.gate_up_proj.weight_scale", 0),
    (".mlp.up_proj.weight_scale",   ".mlp.gate_up_proj.weight_scale", 1),
    # Full attention
    (".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", "q"),
    (".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight", "k"),
    (".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight", "v"),
    (".self_attn.q_proj.weight_scale", ".self_attn.qkv_proj.weight_scale", "q"),
    (".self_attn.k_proj.weight_scale", ".self_attn.qkv_proj.weight_scale", "k"),
    (".self_attn.v_proj.weight_scale", ".self_attn.qkv_proj.weight_scale", "v"),
]

# Build v2 tensors
v2_tensors = {}

# First, copy all non-weight parameters from v1
for key, tensor in v1_tensors.items():
    if not key.endswith('.weight') and not key.endswith('.weight_scale'):
        v2_tensors[key] = tensor

# Identify packed components
packed_components = set()  # keys that are packing inputs (should NOT appear in v2)
packing_groups = {}  # packed_key -> {shard_id: tensor}

for unpacked_suffix, packed_suffix, shard_id in STACKING:
    for key in list(v1_tensors.keys()):
        if key.endswith(unpacked_suffix):
            # This key is a packing input
            packed_key = key.replace(unpacked_suffix, packed_suffix)
            
            if packed_key not in packing_groups:
                packing_groups[packed_key] = {}
            
            if isinstance(shard_id, tuple):
                for i, sid in enumerate(shard_id):
                    packing_groups[packed_key][sid] = v1_tensors[key]
            else:
                packing_groups[packed_key][shard_id] = v1_tensors[key]
            
            packed_components.add(key)

# Concatenate packed groups
for packed_key, shards in packing_groups.items():
    sorted_shards = sorted(shards.items())
    tensors = [s[1] for s in sorted_shards]
    v2_tensors[packed_key] = torch.cat(tensors, dim=0)
    print(f"  Packed: {packed_key} <- {len(tensors)} parts -> {v2_tensors[packed_key].shape}")

# Add remaining weight/scale keys that weren't packed
for key, tensor in v1_tensors.items():
    if key not in packed_components and key not in v2_tensors:
        v2_tensors[key] = tensor

print(f"\nv2 has {len(v2_tensors)} tensors (v1 had {len(v1_tensors)})")

# Save v2
out_dir = Path(V2_PATH)
out_dir.mkdir(parents=True, exist_ok=True)

# Split into shards
sorted_keys = sorted(v2_tensors.keys())
files = []
current_shard = {}
current_size = 0
shard_idx = 1
MAX_SHARD = 4 * 1024**3
weight_map = {}

for key in sorted_keys:
    t = v2_tensors[key]
    t_size = t.nelement() * t.element_size()
    if current_size + t_size > MAX_SHARD and current_shard:
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(out_dir / fname))
        for k in current_shard:
            weight_map[k] = fname
        files.append(fname)
        shard_idx += 1
        current_shard = {}
        current_size = 0
    current_shard[key] = t
    current_size += t_size

if current_shard:
    fname = f"model-{shard_idx:05d}.safetensors"
    save_file(current_shard, str(out_dir / fname))
    for k in current_shard:
        weight_map[k] = fname
    files.append(fname)

# Write index
total_size = sum(t.nelement() * t.element_size() for t in v2_tensors.values())
index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
with open(out_dir / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

# Copy config files
shutil.copy2(os.path.join(ORIG_PATH, "config.json"), out_dir / "config.json")
for fname in os.listdir(ORIG_PATH):
    if fname.endswith(('.json', '.jinja', '.txt', '.md')) and fname != "config.json":
        src = os.path.join(ORIG_PATH, fname)
        dst = out_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

# Update config with quantization config from v1
with open(out_dir / "config.json") as f:
    config = json.load(f)
with open(os.path.join(V1_PATH, "config.json")) as f:
    v1_config = json.load(f)
config["quantization_config"] = v1_config.get("quantization_config", {})
with open(out_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# Also copy quantization report
shutil.copy2(os.path.join(V1_PATH, "quantization_report.json"), out_dir / "quantization_report.json")

print(f"\nSaved {len(files)} shards, {total_size/1024**3:.2f} GB to {V2_PATH}")
print("Done!")
