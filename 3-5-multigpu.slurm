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
#SBATCH --mem=48G
#SBATCH --time=00:05:00
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

# 检查 accelerate 配置（如果未配置，使用默认多GPU设置）
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "Creating accelerate default config for multi-GPU..."
    accelerate config default
fi

# 使用 accelerate 启动多GPU训练
# accelerate 会自动检测 SLURM 环境并配置分布式训练
accelerate launch --multi_gpu --num_processes=2 3-5-qwen-grpo-train-multigpu.py

# 或者使用 torchrun（PyTorch原生多GPU）
# torchrun --nproc_per_node=2 3-qwen-grpo-train-multigpu.py

echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
