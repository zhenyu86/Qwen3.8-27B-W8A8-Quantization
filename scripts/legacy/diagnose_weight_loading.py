"""
Diagnostic: Check how SGLang loads the quantized checkpoint.
We'll simulate the Qwen3_5ForConditionalGeneration.load_weights logic
and verify weight_scale loading for stacked params.
"""
import json
import torch
from safetensors.torch import load_file
from pathlib import Path

CKPT_DIR = Path("/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8")

stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_qkvz.", "in_proj_z.", 3),
    ("in_proj_ba.", "in_proj_b.", 0),
    ("in_proj_ba.", "in_proj_a.", 1),
]

with open(CKPT_DIR / "model.safetensors.index.json") as f:
    index = json.load(f)

# Load all tensors
all_tensors = {}
for shard_file in set(index["weight_map"].values()):
    tensors = load_file(str(CKPT_DIR / shard_file))
    all_tensors.update(tensors)

print(f"Total tensors in checkpoint: {len(all_tensors)}")

# Simulate the load_weights for a few layers
loaded_count = 0
scale_not_loaded = 0
weight_loaded_no_scale = 0

for name in sorted(all_tensors.keys()):
    if "weight_scale" in name or "weight" not in name or "scale" in name:
        continue
    
    # Simulate the name transforms
    if "language_model" in name:
        sim_name = name.replace("model.language_model.", "model.")
    else:
        sim_name = name
    if ".self_attn." in sim_name:
        sim_name = sim_name.replace(".self_attn", "")
    
    # Check stacked params
    matched = False
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if "mlp.experts" in sim_name:
            continue
        if weight_name not in sim_name:
            continue
        fused_name = sim_name.replace(weight_name, param_name)
        scale_name = fused_name.replace(".weight", ".weight_scale")
        matched = True
        
        # Check if the corresponding scale exists
        # Need to convert back to original checkpoint key format for checking
        if scale_name in all_tensors:
            loaded_count += 1
        else:
            # The scale key in checkpoint uses different format
            # Let's check directly
            orig_scale_name = name.replace(".weight", ".weight_scale")
            if orig_scale_name in all_tensors:
                loaded_count += 1
            else:
                scale_not_loaded += 1
                if scale_not_loaded <= 5:
                    print(f"  SCALE MISMATCH for {name}")
                    print(f"    sim_name: {sim_name}")
                    print(f"    fused_name: {fused_name}")
                    print(f"    scale_name: {scale_name}")
        break
    
    if not matched:
        # Non-stacked weight
        scale_name = sim_name.replace(".weight", ".weight_scale")
        if scale_name in all_tensors:
            loaded_count += 1
        else:
            orig_scale_name = name.replace(".weight", ".weight_scale")
            if orig_scale_name in all_tensors:
                loaded_count += 1
            else:
                weight_loaded_no_scale += 1

print(f"\nWeights with matching scales: {loaded_count}")
print(f"Scales potentially misloaded: {scale_not_loaded}")
print(f"Weights without scales (expected for layernorm/embed/etc): {weight_loaded_no_scale}")

# Check specifically for the key issue: do weight_scale keys exist for ALL quantized weights?
print("\n=== Checking weight_scale coverage ===")
weight_keys = set()
scale_keys = set()
for k in all_tensors.keys():
    if k.endswith(".weight") and "embed" not in k and "lm_head" not in k and "visual" not in k and "norm" not in k and "conv1d" not in k and "mtp" not in k:
        weight_keys.add(k)
    elif k.endswith(".weight_scale"):
        scale_keys.add(k)

print(f"Quantized weight keys: {len(weight_keys)}")
print(f"Weight scale keys: {len(scale_keys)}")

# For each weight, check if there's a matching scale
missing_scales = []
for wk in sorted(weight_keys):
    sk = wk.replace(".weight", ".weight_scale")
    if sk not in scale_keys:
        missing_scales.append(wk)

print(f"Weights missing scales: {len(missing_scales)}")
if missing_scales:
    for k in missing_scales[:5]:
        print(f"  {k}")
