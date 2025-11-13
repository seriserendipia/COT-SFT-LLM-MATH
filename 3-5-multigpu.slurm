#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# 多GPU训练配置（针对 ankner/gsm8k-CoT 数据集，GRPO训练）
# ============================================================================
# 数据集      | GPU数 | 时间      | 内存  | CPU | 说明
# ------------|-------|----------|-------|-----|------------------
# 测试        | 2     | 00:15:00 | 32G   | 12  | 快速测试（10样本）
# 小规模      | 2     | 00:30:00 | 40G   | 12  | 开发调试（100样本）
# 中等规模    | 2     | 01:30:00 | 48G   | 16  | 中等规模（1000样本）
# 完整训练    | 2     | 06:00:00 | 64G   | 16  | 完整训练（7470样本，当前配置）
# ============================================================================
# 
# 📊 计算过程（完整 GRPO 训练，2×A100）：
#   - 有效批次大小: 2 (per_device) × 2 (grad_accum) × 2 (GPUs) = 8
#   - 总步数: 7470 / 8 = 934 steps
#   - 估计时间: 934 × 8s = 7472s ≈ 2.08小时 (GRPO需生成多个候选响应)
#   - 加载+保存开销: +30分钟
#   - DDP同步开销: +15%
#   - 安全余量 (×2): 6小时
# 
# 💾 显存使用（每个 A100 40GB）：
#   - 模型（4-bit）: 4GB
#   - LoRA + 优化器 + 激活值: 8-10GB
#   - GRPO生成缓存: 3-4GB
#   - 剩余余量: ~23GB ✅ 充足
# 
# 🚀 多GPU加速效果：
#   - 理论加速: 2倍
#   - 实际加速: ~1.7-1.8倍 (通信开销)
# ============================================================================

# 当前配置：完整训练（7470样本，2×GPU）
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="v100|a40|a100"
#SBATCH --mail-user=yihelu@usc.edu
#SBATCH --mail-type=END,FAIL


# ============================================================================
# 多GPU策略说明：
# 1. PyTorch DDP (DistributedDataParallel)：自动启用，无需手动配置
# 2. accelerate 库会自动检测 SLURM 环境变量并初始化多GPU
# 3. 每个GPU运行相同代码，但处理不同批次数据
# 4. 梯度在GPU间自动同步
# ============================================================================

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"

# Move to the correct directory
cd /home1/yihelu/csci566/
echo "Current working directory is `pwd`"

# Load conda environment
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

# ============================================================================
# 多GPU DDP 训练配置
# ============================================================================
# 
# ⚠️ 重要：DDP + gradient_checkpointing + LoRA 存在已知冲突
# 错误：RuntimeError: Expected to mark a variable ready only once
# 解决：必须关闭 gradient_checkpointing=False（已在代码中设置）
# 
# DDP 模式特点：
#   ✅ 配置简单，无需额外依赖
#   ✅ 训练速度快（1.8× 单GPU）
#   ✅ 与 4-bit 量化 + LoRA 完美兼容
#   ❌ 不能使用 gradient_checkpointing
#   📊 显存占用：~14GB/GPU（Qwen2.5-1.5B）
# 
# 如果显存不足，可以：
#   - 减小 per_device_train_batch_size（代码中改为 1）
#   - 增加 gradient_accumulation_steps（代码中改为 4）
#   - 减小 max_completion_length（代码中改为 128）
# 
echo "🚀 Launching 2-GPU DDP training..."
accelerate launch --config_file accelerate_config_2gpu.yaml 3-5-qwen-grpo-train-multigpu.py
# 
# ============================================================================

echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
