"""
Monkey-patch W8A8Int8LinearMethod.process_weights_after_loading to check
if weight_scale values are loaded correctly (non-zero) before transposition.
"""
import torch
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TP_SIZE"] = "1"  # Use TP=1 for simpler debugging

# Monkey-patch to capture weight/scale values after loading
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod
from torch.nn import Parameter

_original_pwal = W8A8Int8LinearMethod.process_weights_after_loading
_layer_info = []

def _patched_pwal(self, layer):
    """Log weight and scale before transposition."""
    w = layer.weight.data
    s = layer.weight_scale.data
    
    w_min = w.float().min().item()
    w_max = w.float().max().item()
    w_mean = w.float().abs().mean().item()
    w_nonzero = (w != 0).sum().item()
    
    s_min = s.min().item()
    s_max = s.max().item()
    s_mean = s.mean().item()
    s_nonzero = (s != 0).sum().item()
    
    info = {
        'w_shape': list(w.shape),
        'w_dtype': str(w.dtype),
        'w_min': w_min,
        'w_max': w_max,
        'w_mean': w_mean,
        'w_nonzero_pct': w_nonzero / w.numel() * 100,
        's_shape': list(s.shape),
        's_dtype': str(s.dtype),
        's_min': s_min,
        's_max': s_max,
        's_mean': s_mean,
        's_nonzero_pct': s_nonzero / s.numel() * 100,
    }
    _layer_info.append(info)
    
    # Call original
    _original_pwal(self, layer)

W8A8Int8LinearMethod.process_weights_after_loading = _patched_pwal

# Now load the model
print("Loading model with monkey-patch diagnostic...")
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs

args = ServerArgs(
    model_path='/home/acceleration/quantization/Qwen3.8-27B-Enhanced-W8A8',
    tp_size=1,
    quantization='w8a8_int8',
    dtype='bfloat16',
)

config = ModelConfig.from_server_args(args)

from sglang.srt.model_loader.loader import get_model_loader
loader = get_model_loader(args)
model = loader.load_model(config)

# Analyze results
print(f"\n=== Collected info from {len(_layer_info)} layers ===")

# Show first 10 layers
for i, info in enumerate(_layer_info[:10]):
    print(f"\n--- Layer {i} ---")
    for k, v in info.items():
        print(f"  {k}: {v}")

# Summary statistics
zero_scale = sum(1 for i in _layer_info if i['s_max'] == 0)
tiny_scale = sum(1 for i in _layer_info if 0 < i['s_max'] < 1e-6)
normal_scale = sum(1 for i in _layer_info if i['s_max'] >= 1e-6)
zero_weight = sum(1 for i in _layer_info if i['w_max'] == 0)

print(f"\n=== Summary ===")
print(f"Total layers checked: {len(_layer_info)}")
print(f"Zero weight scales (BROKEN): {zero_scale}")
print(f"Tiny weight scales (<1e-6): {tiny_scale}")
print(f"Normal weight scales (>=1e-6): {normal_scale}")
print(f"Zero weights: {zero_weight}")

# Check if all weights are int8
int8_count = sum(1 for i in _layer_info if 'int8' in i['w_dtype'])
print(f"INT8 weights: {int8_count}/{len(_layer_info)}")

# Save full log
with open('/home/acceleration/quantization/logs/weight_load_debug.json', 'w') as f:
    json.dump(_layer_info, f, indent=2)
print(f"\nFull log saved to logs/weight_load_debug.json")
