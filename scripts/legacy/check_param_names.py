"""
Minimal: just check what parameter names exist in the model after init.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.load_config import LoadConfig

args = ServerArgs(
    model_path='/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8',
    tp_size=1,
    quantization='w8a8_int8',
    dtype='bfloat16',
)
config = ModelConfig.from_server_args(args)

# Get quant_config like the loader does
from sglang.srt.model_loader.loader import get_quant_config
from sglang.srt.model_loader.loader import get_model_architecture

model_class, _ = get_model_architecture(config)
packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})
load_config = LoadConfig()
quant_config = get_quant_config(config, load_config, packed_modules_mapping, None)

print(f"Quant config type: {type(quant_config)}")

# Build model
model = model_class(
    config=config.hf_text_config,
    quant_config=quant_config,
    prefix="",
)

# Check params
params_dict = dict(model.named_parameters(remove_duplicate=False))
scale_params = {k: v for k, v in params_dict.items() if 'weight_scale' in k}

print(f"\nTotal params: {len(params_dict)}")
print(f"Weight_scale params: {len(scale_params)}")

# Show first few scale param names
for i, (k, v) in enumerate(scale_params.items()):
    if i < 10:
        print(f"  {k}: shape={v.shape}")

# Now simulate name mapping
print("\n=== Simulating load_weights name mapping ===")
stacked_params_mapping = [
    ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_z.", "in_proj_z.", 3),
    ("in_proj_ba.", "in_proj_b.", 0),
    ("in_proj_ba.", "in_proj_a.", 1),
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]

test_keys = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
    "model.language_model.layers.0.linear_attn.in_proj_z.weight_scale",
    "model.language_model.layers.0.linear_attn.in_proj_a.weight_scale",
    "model.language_model.layers.0.linear_attn.in_proj_b.weight_scale",
    "model.language_model.layers.0.linear_attn.out_proj.weight_scale",
    "model.language_model.layers.0.mlp.gate_proj.weight_scale",
    "model.language_model.layers.0.mlp.up_proj.weight_scale",
    "model.language_model.layers.0.mlp.down_proj.weight_scale",
]

for ckpt_key in test_keys:
    name = ckpt_key
    if "language_model" in name:
        name = name.replace(r"model.language_model.", r"model.")
    
    shard_id = None
    for param_name, weight_name, sid in stacked_params_mapping:
        if weight_name in name:
            name = name.replace(weight_name, param_name)
            shard_id = sid
            break
    
    in_dict = name in params_dict
    status = "FOUND" if in_dict else "MISSING!"
    print(f"  {ckpt_key.split('layers.0.')[-1]}")
    print(f"    -> {name.split('layers.0.')[-1]} shard_id={shard_id} -> {status}")
