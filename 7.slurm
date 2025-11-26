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
#SBATCH --output=slurm-%j-grpo-only-eval.out
#SBATCH --mail-user=yihelu@usc.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# GRPO-only 消融评估 (全量测试集: 1319 条)
# 预估时间: ~3 小时 | 安全边际: 8倍 | 配置: 24 小时
# ============================================================================

source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

echo "🚀 Starting GRPO-only Evaluation (Ablation Study)..."
echo "⚠️  Testing GRPO LoRA without SFT foundation"
python 7-qwen-sftbased-grpo-only-eval.py
