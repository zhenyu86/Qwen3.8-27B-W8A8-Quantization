"""Quick diagnostic to check quantized checkpoint vs what SGLang expects."""
import json, torch
from safetensors import safe_open
from pathlib import Path

ckpt_dir = Path("/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8")

# Read index
with open(ckpt_dir / "model.safetensors.index.json") as f:
    idx = json.load(f)

# Check weight types and shapes
weight_count = 0
scale_count = 0
missing_scale = []
embed_quantized = False

for shard_file in set(idx["weight_map"].values()):
    with safe_open(ckpt_dir / shard_file, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if "weight_scale" in key:
                scale_count += 1
            elif "weight" in key and "scale" not in key:
                weight_count += 1
                if tensor.dtype == torch.int8:
                    pass  # OK
                elif "embed" in key or "lm_head" in key:
                    pass  # Expected to be float
                else:
                    print(f"WARNING: {key} is {tensor.dtype} (expected int8)")

print(f"\nWeight count: {weight_count}")
print(f"Scale count: {scale_count}")

# Check if weight_scale exists for each non-embedding weight
for shard_file in set(idx["weight_map"].values()):
    with safe_open(ckpt_dir / shard_file, framework="pt") as f:
        for key in sorted(f.keys()):
            if "weight" in key and "scale" not in key and "bias" not in key:
                scale_key = key.replace(".weight", ".weight_scale")
                if scale_key not in idx["weight_map"]:
                    missing_scale.append(key)

if missing_scale:
    print(f"\nMissing weight_scale for {len(missing_scale)} weights:")
    for k in missing_scale[:10]:
        print(f"  {k}")
    if len(missing_scale) > 10:
        print(f"  ... and {len(missing_scale) - 10} more")
else:
    print("\nAll weights have matching weight_scale keys")

# Check first few weight+scale pairs
print("\nSample weight+scale pairs:")
count = 0
for shard_file in set(idx["weight_map"].values()):
    with safe_open(ckpt_dir / shard_file, framework="pt") as f:
        for key in sorted(f.keys()):
            if "weight_scale" in key and count < 5:
                weight_key = key.replace(".weight_scale", ".weight")
                if weight_key in idx["weight_map"]:
                    w = f.get_tensor(key)
                    with safe_open(ckpt_dir / idx["weight_map"][weight_key], framework="pt") as wf:
                        wt = wf.get_tensor(weight_key)
                    print(f"  {weight_key}: {wt.shape} {wt.dtype} + {key}: {w.shape} {w.dtype} min={w.min():.4f} max={w.max():.4f}")
                    count += 1
