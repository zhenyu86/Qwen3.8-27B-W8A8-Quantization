"""
Minimal diagnostic: load one layer's weights and check if scales are loaded correctly.
Run with: python3 scripts/check_scales.py
"""
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from safetensors import safe_open
import glob

ckpt_dir = '/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8'
files = sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors')))

# Check layer 0's weight and weight_scale
layer0_w = None
layer0_s = None
layer0_zw = None
layer0_zs = None

for f in files:
    with safe_open(f, framework='pt') as sf:
        for k in sf.keys():
            if 'layers.0.linear_attn.in_proj_qkv.weight' == k.split('model.language_model.')[-1]:
                layer0_w = sf.get_tensor(k)
                print(f"  {k}: shape={layer0_w.shape}, dtype={layer0_w.dtype}, "
                      f"min={layer0_w.float().min():.4f}, max={layer0_w.float().max():.4f}")
            if 'layers.0.linear_attn.in_proj_qkv.weight_scale' == k.split('model.language_model.')[-1]:
                layer0_s = sf.get_tensor(k)
                print(f"  {k}: shape={layer0_s.shape}, dtype={layer0_s.dtype}, "
                      f"min={layer0_s.min():.6f}, max={layer0_s.max():.6f}, mean={layer0_s.mean():.6f}")
            if 'layers.0.linear_attn.in_proj_z.weight' == k.split('model.language_model.')[-1]:
                layer0_zw = sf.get_tensor(k)
                print(f"  {k}: shape={layer0_zw.shape}, dtype={layer0_zw.dtype}, "
                      f"min={layer0_zw.float().min():.4f}, max={layer0_zw.float().max():.4f}")
            if 'layers.0.linear_attn.in_proj_z.weight_scale' == k.split('model.language_model.')[-1]:
                layer0_zs = sf.get_tensor(k)
                print(f"  {k}: shape={layer0_zs.shape}, dtype={layer0_zs.dtype}, "
                      f"min={layer0_zs.min():.6f}, max={layer0_zs.max():.6f}, mean={layer0_zs.mean():.6f}")

# Now check what the model parameters look like after loading
# Simulate the W8A8 loading path
print("\n=== Simulating W8A8 loading for in_proj_qkvz ===")
# in_proj_qkvz.weight should be [q_size+k_size+v_size+z_size, input_size]
# The checkpoint has in_proj_qkv.weight [10240, 5120] and in_proj_z.weight [5120, 5120]
# Fused: [15360, 5120]
if layer0_w is not None and layer0_zw is not None:
    # Concatenate along output dim
    fused_weight = torch.cat([layer0_w, layer0_zw], dim=0)
    print(f"Fused weight shape: {fused_weight.shape}")
    print(f"  dtype: {fused_weight.dtype}, min: {fused_weight.float().min():.4f}, max: {fused_weight.float().max():.4f}")

if layer0_s is not None and layer0_zs is not None:
    fused_scale = torch.cat([layer0_s, layer0_zs], dim=0)
    print(f"Fused scale shape: {fused_scale.shape}")
    print(f"  dtype: {fused_scale.dtype}, min: {fused_scale.min():.6f}, max: {fused_scale.max():.6f}")

# Check the reference BF16 model's first layer for comparison
print("\n=== Reference BF16 model layer 0 ===")
ref_dir = '/home/weight/Qwen3.8-27B/'
ref_files = sorted(glob.glob(os.path.join(ref_dir, '*.safetensors')))
for f in ref_files:
    with safe_open(f, framework='pt') as sf:
        for k in sf.keys():
            if 'layers.0.linear_attn.in_proj_qkv.weight' in k:
                ref_w = sf.get_tensor(k)
                print(f"  {k}: shape={ref_w.shape}, dtype={ref_w.dtype}, "
                      f"min={ref_w.float().min():.4f}, max={ref_w.float().max():.4f}")
            if 'layers.0.linear_attn.in_proj_z.weight' in k:
                ref_w = sf.get_tensor(k)
                print(f"  {k}: shape={ref_w.shape}, dtype={ref_w.dtype}, "
                      f"min={ref_w.float().min():.4f}, max={ref_w.float().max():.4f}")

print("\nDone.")
