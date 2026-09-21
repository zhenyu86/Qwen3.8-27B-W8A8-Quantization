#!/usr/bin/env python3
"""
Fix: Re-save the quantized model with weights packed to match SGLang's
stacked_params_mapping format.

The issue: SGLang's Qwen3_5ForCausalLM.load_weights() packs individual 
checkpoint weights into fused parameters using stacked_params_mapping:
  in_proj_qkv + in_proj_z -> in_proj_qkvz (shard_ids 0,1,2 / 3)
  in_proj_b + in_proj_a   -> in_proj_ba   (shard_ids 0 / 1)
  gate_proj + up_proj     -> gate_up_proj (shard_ids 0 / 1)
  q_proj + k_proj + v_proj -> qkv_proj    (shard_ids q/k/v)

But our quantization script saved them with the ORIGINAL (unpacked) names.
SGLang's weight loaders then can't find the packed names -> garbage output.

Fix: Load quantized weights, pack them, and re-save with packed names.
"""

import json
import os
import gc
import shutil
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file


# ============================================================
# Stacking rules from Qwen3_5ForCausalLM.load_weights()
# ============================================================

# For full_attention layers: q_proj/k_proj/v_proj -> qkv_proj
# For all layers: gate_proj/up_proj -> gate_up_proj
# For linear_attention layers:
#   in_proj_qkv + in_proj_z -> in_proj_qkvz (0,1,2 / 3)
#   in_proj_b + in_proj_a   -> in_proj_ba   (0 / 1)

# The stacking mapping: (packed_name, unpacked_name, shard_id)
STACKED_MAPPINGS = [
    # Attention QKV packing
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    # MLP gate+up packing
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    # GatedDeltaNet packing
    ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
    ("in_proj_qkvz", "in_proj_z", 3),
    ("in_proj_ba", "in_proj_b", 0),
    ("in_proj_ba", "in_proj_a", 1),
]


def find_stacking(unpacked_name):
    """Given an unpacked weight name like 'model.layers.0.linear_attn.in_proj_qkv.weight',
    return (packed_full_name, shard_id) or None.
    
    Example: 'model.layers.0.linear_attn.in_proj_qkv.weight'
      -> ('model.layers.0.linear_attn.in_proj_qkvz.weight', (0,1,2))
    """
    for packed_part, unpacked_part, shard_id in STACKED_MAPPINGS:
        # Check if this weight matches the unpacked pattern
        # e.g., ".in_proj_qkv.weight" in the name
        pattern = f".{unpacked_part}.weight"
        scale_pattern = f".{unpacked_part}.weight_scale"
        
        if pattern in unpacked_name:
            packed_name = unpacked_name.replace(f".{unpacked_part}.", f".{packed_part}.")
            return packed_name, shard_id
        if scale_pattern in unpacked_name:
            packed_name = unpacked_name.replace(f".{unpacked_part}.", f".{packed_part}.")
            return packed_name, shard_id
    
    return None


def pack_quantized_weights(quant_state):
    """
    Pack quantized weights from unpacked names to SGLang's packed format.
    
    quant_state: dict of {name: {'weight_int8': tensor, 'weight_scale': tensor}}
    where name is like 'model.layers.0.linear_attn.in_proj_qkv'
    
    Returns: dict with packed names and concatenation logic info
    """
    # Group weights by their packed target
    pack_groups = {}  # packed_name -> {shard_id: (name, weight, scale)}
    standalone = {}   # name -> (weight, scale) for non-packed weights
    
    for name, state in quant_state.items():
        weight_key = f"{name}.weight"
        result = find_stacking(weight_key)
        
        if result is not None:
            packed_name, shard_id = result
            # Get the base packed name (without .weight suffix)
            packed_base = packed_name.replace(".weight", "").replace(".weight_scale", "")
            suffix = ".weight" if ".weight" in packed_name and ".weight_scale" not in packed_name else ".weight_scale"
            
            if packed_base not in pack_groups:
                pack_groups[packed_base] = {}
            
            # shard_id can be int or tuple
            if isinstance(shard_id, tuple):
                for i, sid in enumerate(shard_id):
                    pack_groups[packed_base][sid] = (name, suffix)
            else:
                pack_groups[packed_base][shard_id] = (name, suffix)
        else:
            standalone[name] = True
    
    return pack_groups, standalone


def repack_and_save(model_path, output_path, quant_state, bf16_layers):
    """Re-save model with packed weight format for SGLang."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine packed groups
    pack_groups, standalone = pack_quantized_weights(quant_state)
    
    print(f"  Pack groups: {len(pack_groups)}")
    print(f"  Standalone layers: {len(standalone)}")
    
    # Build the complete state_dict
    # Strategy: load each original shard, replace quantized weights with packed INT8 versions
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        orig_index = json.load(f)
    orig_weight_map = orig_index['weight_map']
    
    # Group keys by original shard
    shard_keys = {}
    for key, shard in orig_weight_map.items():
        if shard not in shard_keys:
            shard_keys[shard] = []
        shard_keys[shard].append(key)
    
    # For each packed group, compute the packed weight
    packed_tensors = {}
    
    for packed_base, shards in pack_groups.items():
        # Sort by shard_id
        sorted_shards = sorted(shards.items())
        
        # Determine which component names map to which shard_ids
        # We need to figure out: for shard_id (0,1,2) -> in_proj_qkv, shard_id 3 -> in_proj_z
        # The actual shard_id -> name mapping is in our quant_state
        
        # Collect weights in shard order
        weight_parts = []
        scale_parts = []
        
        for shard_id, (orig_name, suffix) in sorted_shards:
            if orig_name in quant_state:
                qs = quant_state[orig_name]
                weight_parts.append(qs['weight_int8'])
                scale_parts.append(qs['weight_scale'])
            else:
                print(f"  WARNING: {orig_name} not in quant_state")
                continue
        
        if weight_parts:
            # Concatenate along output dimension (dim=0)
            packed_weight = torch.cat(weight_parts, dim=0)
            packed_scale = torch.cat(scale_parts, dim=0)
            
            packed_tensors[f"{packed_base}.weight"] = packed_weight
            packed_tensors[f"{packed_base}.weight_scale"] = packed_scale
    
    # Now build final state_dict by processing each original shard
    print("  Building packed state_dict...")
    final_files = []
    final_weight_map = {}
    shard_idx = 1
    current_shard = {}
    current_size = 0
    MAX_SHARD = 4 * 1024**3
    
    for shard_name in sorted(shard_keys.keys()):
        print(f"    Processing {shard_name}...")
        shard_data = load_file(os.path.join(model_path, shard_name))
        
        for key in shard_keys[shard_name]:
            # Check if this key has been packed
            if key in packed_tensors:
                # Replace with packed version
                current_shard[key] = packed_tensors[key]
            else:
                # Check if this is a quantized individual weight that should NOT be saved
                # (because it's been packed into a different key)
                is_packed_component = False
                for packed_key in packed_tensors:
                    if key.replace(".weight", "") in packed_key:
                        # This might be a component that was packed
                        # Only skip if the packed version exists
                        base = key.rsplit(".", 1)[0]
                        for packed_base in pack_groups:
                            if base in pack_groups:
                                is_packed_component = True
                                break
                
                if is_packed_component and key not in packed_tensors:
                    # Skip - this component was packed into another key
                    # But we need to check: is this a weight or scale of a packed component?
                    # If so, skip it because the packed version is already saved
                    continue
                
                # Check if this is a quantized weight that should stay as INT8
                base_name = key.rsplit(".", 1)[0] if "." in key else ""
                if base_name in quant_state and key.endswith(".weight"):
                    # This is a non-packed quantized weight
                    current_shard[key] = quant_state[base_name]['weight_int8']
                elif base_name in quant_state and key.endswith(".weight_scale"):
                    current_shard[key] = quant_state[base_name]['weight_scale']
                else:
                    # Non-quantized parameter, keep as-is
                    current_shard[key] = shard_data[key]
        
        del shard_data
        gc.collect()
        
        # Write shard if large enough
        for key in list(current_shard.keys()):
            t = current_shard[key]
            t_size = t.nelement() * t.element_size()
            if current_size + t_size > MAX_SHARD and current_shard:
                fname = f"model-{shard_idx:05d}.safetensors"
                save_file({k: current_shard[k] for k in list(current_shard.keys())}, 
                         str(out_dir / fname))
                for k in current_shard:
                    final_weight_map[k] = fname
                final_files.append(fname)
                shard_idx += 1
                current_shard = {}
                current_size = 0
            current_shard[key] = t
            current_size += t_size
            del current_shard[key]  # Remove from dict after counting
    
    if current_shard:
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(out_dir / fname))
        for k in current_shard:
            final_weight_map[k] = fname
        final_files.append(fname)
    
    # ... rest of save logic
    print(f"  Saved {len(final_files)} shards")
