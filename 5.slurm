#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="v100|a40|a100"
#SBATCH --output=slurm-%j-sft-grpo-train.out
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=END,FAIL

# ============================================================================
# SFT + GRPO 训练任务 (全量数据: 7470 �?
# 🚀 �?GPU 模式 (2×GPU, gradient_checkpointing=False)
# 预估时间: ~3-4 小时 | 安全边际: 12�?| 配置: 48 小时（GPU 分区最大限制）
# ============================================================================

source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

echo "🚀 Starting SFT+GRPO Multi-GPU Training Job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Date: `date`"

# 使用 accelerate 启动�?GPU 训练
accelerate launch --config_file accelerate_config_2gpu.yaml 5-qwen-sft-grpo-train.py

echo "Done at `date`"
