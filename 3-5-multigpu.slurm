#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# 多GPU训练配置（适合大数据集）
# ============================================================================
# 数据集大小 | GPU数 | 时间       | 内存  | CPU | 说明
# -----------|-------|-----------|-------|-----|------------------
# 100 样本   | 2     | 00:20:00  | 48G   | 16  | 2倍加速
# 1000 样本  | 2     | 01:00:00  | 64G   | 16  | 中等规模
# 全数据集   | 4     | 04:00:00  | 96G   | 32  | 完整训练加速
# ============================================================================

# 当前配置：2×GPU 测试配置
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:05:00
#SBATCH --gpus-per-task=2  # 申请2个GPU
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
