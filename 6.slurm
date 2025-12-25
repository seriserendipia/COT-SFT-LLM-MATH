#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"
#SBATCH --output=slurm-%j-sft-grpo-eval.out
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=END,FAIL

# ============================================================================
# SFT+GRPO 完整评估 (全量测试�? 1319 �?
# 预估时间: ~3 小时 | 安全边际: 8�?| 配置: 24 小时
# ============================================================================

source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

echo "🚀 Starting SFT+GRPO Evaluation..."
python 6-qwen-sft-grpo-eval.py
