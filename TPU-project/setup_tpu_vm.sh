#!/bin/bash
# ============================================================================
# TPU VM 一键环境恢复脚本
# 用法: 新建 TPU VM 后 SSH 进去运行:
#   git clone https://github.com/seriserendipia/COT-SFT-LLM-MATH.git ~/project
#   cd ~/project && bash TPU-project/setup_tpu_vm.sh
#
# 前置条件:
#   - Ubuntu 22.04 TPU VM (TRC v4-8 或其他)
#   - 已有 Python 3.11 (sudo apt install python3.11 python3.11-venv)
#   - 已 clone 代码到 ~/project
# ============================================================================

set -e  # 遇到错误立即退出

echo "============================================"
echo "  TPU VM Environment Setup"
echo "============================================"

# ── 1. 系统依赖 ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system dependencies..."
sudo apt update -qq && sudo apt install -y -qq git python3-pip python3.11-venv

# ── 2. Python 虚拟环境 ───────────────────────────────────────────────
echo ""
echo "[2/6] Creating Python 3.11 venv..."
if [ -d ~/tpu-env-311 ]; then
    echo "  venv already exists, skipping creation"
else
    python3.11 -m venv ~/tpu-env-311
fi
source ~/tpu-env-311/bin/activate
pip install --upgrade pip -q

# ── 3. Python 包安装 ─────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python packages..."
# ⚠️ JAX 版本: Tunix 0.1.6 测试在 0.8.x/0.9.x，不要随意升级
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
pip install "google-tunix[prod]" datasets transformers importlib_resources matplotlib -q

# ── 4. Tunix monkey-patches ──────────────────────────────────────────
echo ""
echo "[4/6] Applying Tunix monkey-patches..."
bash ~/project/TPU-project/patch_tunix.sh

# ── 5. 下载模型权重 ──────────────────────────────────────────────────
# 需要 HF_TOKEN 环境变量（Qwen2.5-Coder-1.5B 需要认证）
# 运行前设置: export HF_TOKEN=hf_xxx
echo ""
echo "[5/6] Downloading model weights..."
MODEL_DIR=~/models/qwen2.5-coder-1.5b
if [ -d "$MODEL_DIR" ] && [ "$(ls -1 $MODEL_DIR/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Model already cached at $MODEL_DIR, skipping"
else
    echo "  Downloading Qwen2.5-Coder-1.5B to $MODEL_DIR..."
    python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-Coder-1.5B',
    local_dir='$MODEL_DIR',
    token=os.environ.get('HF_TOKEN'),
)
print('  Download complete!')
"
fi

# ── 6. 验证 ──────────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying environment..."
python3 -c "
import jax
print(f'  JAX {jax.__version__}, devices: {[str(d) for d in jax.devices()]}')
import tunix
print(f'  Tunix {tunix.__version__}')
from tunix.models.qwen2.model import ModelConfig
assert hasattr(ModelConfig, 'qwen2p5_coder_1p5b'), 'Coder classmethod missing!'
print('  Coder classmethod: OK')
from transformers import AutoTokenizer
print('  Transformers: OK')
import matplotlib
print(f'  Matplotlib {matplotlib.__version__}')
print()
print('  All checks passed!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  "
echo "  每次 SSH 登录后别忘了:"
echo "    source ~/tpu-env-311/bin/activate"
echo "  "
echo "  运行实验:"
echo "    cd ~/project && python3 TPU-project/tpu_sft_pipeline.py"
echo "  "
echo "  生成图表:"
echo "    python3 TPU-project/generate_figures.py"
echo "============================================"
