"""
Fix the quantized checkpoint:
1. Dequantize lm_head.weight back to BF16 (SGLang doesn't quantize lm_head)
2. Remove lm_head.weight_scale from checkpoint
3. Save new checkpoint files

This fixes the garbage output issue where lm_head was loaded as raw INT8
without dequantization.
"""
import json
import torch
import os
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
from safetensors import safe_open

CKPT_DIR = Path("/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8")

def fix_checkpoint():
    # Read index
    with open(CKPT_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find which shard contains lm_head.weight and lm_head.weight_scale
    lm_head_shard = weight_map.get("lm_head.weight")
    lm_head_scale_shard = weight_map.get("lm_head.weight_scale")
    
    print(f"lm_head.weight shard: {lm_head_shard}")
    print(f"lm_head.weight_scale shard: {lm_head_scale_shard}")
    
    if lm_head_shard is None:
        print("lm_head.weight not found in checkpoint!")
        return
    
    # Load the shard containing lm_head
    shard_path = CKPT_DIR / lm_head_shard
    tensors = load_file(str(shard_path))
    
    print(f"\nLoaded {lm_head_shard}:")
    for k, v in tensors.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    
    # Dequantize lm_head.weight
    weight = tensors["lm_head.weight"]
    scale = tensors.get("lm_head.weight_scale")
    
    if weight.dtype == torch.int8 and scale is not None:
        print(f"\nDequantizing lm_head.weight: INT8 -> BF16")
        print(f"  weight shape: {weight.shape}, dtype: {weight.dtype}")
        print(f"  scale shape: {scale.shape}, dtype: {scale.dtype}")
        print(f"  scale range: [{scale.min():.6f}, {scale.max():.6f}]")
        
        # Dequantize: weight_bf16 = weight_int8 * scale
        # weight is [out, in], scale is [out, 1]
        weight_bf16 = weight.to(torch.bfloat16) * scale.to(torch.bfloat16)
        tensors["lm_head.weight"] = weight_bf16
        
        # Remove weight_scale
        if "lm_head.weight_scale" in tensors:
            del tensors["lm_head.weight_scale"]
            print(f"  Removed lm_head.weight_scale")
        
        print(f"  Dequantized weight range: [{weight_bf16.min():.6f}, {weight_bf16.max():.6f}]")
        print(f"  Dequantized weight dtype: {weight_bf16.dtype}")
    else:
        print(f"lm_head.weight dtype is {weight.dtype}, not INT8. Skipping.")
    
    # Save the fixed shard
    print(f"\nSaving fixed {lm_head_shard}...")
    save_file(tensors, str(shard_path))
    
    # Update the index - remove lm_head.weight_scale entry
    if "lm_head.weight_scale" in weight_map:
        del weight_map["lm_head.weight_scale"]
        print(f"Removed lm_head.weight_scale from index")
    
    # Save updated index
    with open(CKPT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Updated model.safetensors.index.json")
    
    # Verify
    print(f"\nVerification:")
    with open(CKPT_DIR / "model.safetensors.index.json") as f:
        new_index = json.load(f)
    print(f"  lm_head.weight in index: {'lm_head.weight' in new_index['weight_map']}")
    print(f"  lm_head.weight_scale in index: {'lm_head.weight_scale' in new_index['weight_map']}")
    
    # Re-load and verify
    tensors2 = load_file(str(shard_path))
    print(f"  lm_head.weight dtype: {tensors2['lm_head.weight'].dtype}")
    print(f"  lm_head.weight_scale exists: {'lm_head.weight_scale' in tensors2}")
    
    # Also check total parameter count
    total_keys = len(new_index["weight_map"])
    total_size = sum(os.path.getsize(CKPT_DIR / f) for f in set(new_index["weight_map"].values()))
    print(f"\nTotal keys: {total_keys}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    fix_checkpoint()
