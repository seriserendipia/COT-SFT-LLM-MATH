#!/bin/bash
# ============================================================================
# Tunix 0.1.6 Monkey-Patches
# 修复 3 个已知问题，使 Tunix 能正确运行 Qwen2.5-Coder-1.5B + LoRA
#
# 用法: bash TPU-project/patch_tunix.sh
# 前置: source ~/tpu-env-311/bin/activate
# ============================================================================

set -e

TUNIX_DIR=$(python3 -c "import tunix; import os; print(os.path.dirname(tunix.__file__))")
echo "  Tunix location: $TUNIX_DIR"

# ── Patch 1: Qwen2.5-Coder-1.5B classmethod ─────────────────────────
# 问题: Tunix 不原生支持 Coder 变体
# 修复: 添加 qwen2p5_coder_1p5b classmethod（架构与基座 1.5B 完全一致）
MODEL_PY="$TUNIX_DIR/models/qwen2/model.py"

if grep -q "qwen2p5_coder_1p5b" "$MODEL_PY" 2>/dev/null; then
    echo "  Patch 1 (Coder classmethod): already applied"
else
    echo "  Patch 1 (Coder classmethod): applying..."
    # Insert after qwen2p5_1p5b method (find the closing paren and add after)
    python3 << 'PYEOF'
import re

model_py = "$TUNIX_DIR/models/qwen2/model.py"
with open(model_py) as f:
    content = f.read()

# Find the end of qwen2p5_1p5b method and insert coder variant after it
patch = '''
  @classmethod
  def qwen2p5_coder_1p5b(cls):  # qwen2.5-coder-1.5B (same arch as base)
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )
'''

# Insert after the qwen2p5_1p5b method block
# Find pattern: the end of qwen2p5_1p5b (use_tied_embedding=True followed by closing paren and newline)
# We look for the qwen2p5_1p5b definition and its closing
pattern = r'(def qwen2p5_1p5b\(cls\):.*?use_tied_embedding=True,\s*\))'
match = re.search(pattern, content, re.DOTALL)
if match:
    insert_pos = match.end()
    content = content[:insert_pos] + '\n' + patch + content[insert_pos:]
    with open(model_py, 'w') as f:
        f.write(content)
    print("    Inserted after qwen2p5_1p5b")
else:
    print("    WARNING: Could not find insertion point. Manual patch needed.")
PYEOF
    # Fix the shell variable in the python script
    python3 -c "
import re, os
model_py = os.path.join('$TUNIX_DIR', 'models', 'qwen2', 'model.py')
with open(model_py) as f:
    content = f.read()
if 'qwen2p5_coder_1p5b' in content:
    print('    Already patched')
else:
    patch = '''
  @classmethod
  def qwen2p5_coder_1p5b(cls):  # qwen2.5-coder-1.5B (same arch as base)
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )
'''
    pattern = r'(def qwen2p5_1p5b\(cls\):.*?use_tied_embedding=True,\s*\))'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        pos = match.end()
        content = content[:pos] + '\n' + patch + content[pos:]
        with open(model_py, 'w') as f:
            f.write(content)
        print('    Applied successfully')
    else:
        print('    WARNING: insertion point not found')
"
fi

# ── Patch 2: LoRA dtype cast for KV cache ────────────────────────────
# 问题: LoRA 层输出 float32，但 KV cache 是 bfloat16，导致 dynamic_update_slice 报错
# 修复: 在写入 cache 前 cast 到 cache 的 dtype
if grep -q "astype(cache\['v'\]\.dtype)" "$MODEL_PY" 2>/dev/null; then
    echo "  Patch 2 (dtype cast): already applied"
else
    echo "  Patch 2 (dtype cast): applying..."
    python3 -c "
import os
model_py = os.path.join('$TUNIX_DIR', 'models', 'qwen2', 'model.py')
with open(model_py) as f:
    content = f.read()

# Find the pattern: setting slice_indices then immediately dynamic_update_slice
old = '''      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice('''

new = '''      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      # Cast to cache dtype (LoRA may output float32 while cache is bfloat16)
      value_proj = value_proj.astype(cache['v'].dtype)
      key_proj = key_proj.astype(cache['k'].dtype)
      value_proj = jax.lax.dynamic_update_slice('''

if old in content:
    content = content.replace(old, new)
    with open(model_py, 'w') as f:
        f.write(content)
    print('    Applied successfully')
else:
    print('    Pattern not found (may already be patched or code changed)')
"
fi

# ── Patch 3: _shard_optimizer rank mismatch fix ──────────────────────
# 问题: Qwen2 的 3D attention weights + LoRA 2D params 导致 PartitionSpec rank 不匹配
# 修复: 在 sharding 前检查 rank 并截断 spec
TRAINER_PY="$TUNIX_DIR/sft/peft_trainer.py"

if grep -q "_fix_sharding" "$TRAINER_PY" 2>/dev/null; then
    echo "  Patch 3 (shard_optimizer): already applied"
else
    echo "  Patch 3 (shard_optimizer): applying..."
    python3 -c "
import os
trainer_py = os.path.join('$TUNIX_DIR', 'sft', 'peft_trainer.py')
with open(trainer_py) as f:
    content = f.read()

old_block = '''    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_pspecs
    )'''

new_block = '''    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    # Fix rank mismatch: LoRA params are rank-2 but may inherit rank-3+
    # partition specs from base model (e.g., Qwen2 3D attention weights).
    def _fix_sharding(leaf, sharding):
      if not hasattr(leaf, 'shape'):
        return sharding
      rank = len(leaf.shape)
      if isinstance(sharding, jax.sharding.NamedSharding):
        spec = sharding.spec
        spec_len = len(spec)
        if spec_len > rank:
          new_spec = jax.sharding.PartitionSpec(*spec[:rank])
          return jax.sharding.NamedSharding(sharding.mesh, new_spec)
      return sharding

    optimizer_pspecs = jax.tree.map(
        _fix_sharding, optimizer_state, optimizer_pspecs
    )

    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_pspecs
    )'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(trainer_py, 'w') as f:
        f.write(content)
    print('    Applied successfully')
else:
    print('    Pattern not found (may already be patched or code changed)')
"
fi

# ── Patch 4: base_config.yaml ────────────────────────────────────────
# 问题: Tunix CLI 需要 base_config.yaml 但 pip 安装时可能不包含
CLI_DIR="$TUNIX_DIR/cli"
CONFIG_DIR="$CLI_DIR/configs"
if [ -f "$CLI_DIR/base_config.yaml" ] || [ -f "$CONFIG_DIR/base_config.yaml" ]; then
    echo "  Patch 4 (base_config.yaml): already present"
else
    echo "  Patch 4 (base_config.yaml): copying..."
    if [ -f ~/project/TPU-project/archived/base_config.yaml ]; then
        cp ~/project/TPU-project/archived/base_config.yaml "$CLI_DIR/base_config.yaml"
        mkdir -p "$CONFIG_DIR" && cp ~/project/TPU-project/archived/base_config.yaml "$CONFIG_DIR/base_config.yaml"
        echo "    Copied from archived/"
    else
        echo "    WARNING: base_config.yaml not found in archived/, download manually from GitHub"
    fi
fi

echo ""
echo "  All patches applied!"
