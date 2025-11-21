#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Baseline 推理任务配置（针对 ankner/gsm8k-CoT 数据集）
# ============================================================================
# 数据集      | 样本数 | 时间      | 内存  | CPU | GPU    | 说明
# ------------|--------|----------|-------|-----|--------|------------------
# 快速测试    | 10     | 00:15:00 | 12G   | 4   | A100×1 | 验证推理流程
# 完整推理    | 1320   | 03:00:00 | 20G   | 8   | A100×1 | 完整测试集（当前配置）
# ============================================================================
# 
# 📊 计算过程（完整推理）：
#   - 测试集样本: 1320
#   - 每样本时间: ~5秒 (生成256 tokens @ 50 tokens/s on A100)
#   - 纯推理时间: 1320 × 5s = 6600s ≈ 1.83小时
#   - 加载+保存开销: +20分钟
#   - 安全余量 (×1.5): 3小时
# 
# 💾 显存使用（A100 40GB）：
#   - 模型（4-bit）: 4GB
#   - KV缓存: 2-3GB
#   - 剩余余量: ~33GB ✅ 非常充足
# ============================================================================

# ============================================================================
# GPU 策略：指定 A100 获得最佳性能
# ============================================================================
# 可用 GPU 型号（按性能排序）：
#   l40s (CUDA 8.9) > a100 (CUDA 8.0) > a40 (CUDA 8.6) > v100 (CUDA 7.0)
#   ⚠️ p100 (CUDA 6.0) 不兼容 PyTorch 2.8+
# 
# 推理任务特点：
#   - 对GPU性能要求相对训练更低
#   - A100: 推理速度最快（~50 tokens/s）
#   - A40/V100: 性能也足够（~30-40 tokens/s）
# 
# 灵活策略：如果排队过长，可改为 --gpus-per-task=1 接受任意GPU
# ============================================================================

# 当前配置：完整推理（1320样本 Baseline 推理）
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"

# ============================================================================
# 邮件通知配置
# ============================================================================
#SBATCH --mail-user=yihelu@usc.edu
#SBATCH --mail-type=END,FAIL

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
# 运行 Baseline 推理
# ============================================================================
# 
# 📊 任务说明：
#   - 使用原始 Qwen2.5-Coder-1.5B 模型（无微调）
#   - 在测试集 train[924:1319] (395样本) 上评估
#   - 作为 SFT 和 GRPO 模型的对比基线
# 
# 💾 输出目录：inference_results_baseline/
#   - inference_results.json: 详细推理结果
#   - evaluation_metrics.json: 评估指标
#   - wrong_cases.json: 错误案例分析
# 
echo "🚀 Starting Baseline Model Inference..."
python 0-qwen-baseline-inference.py
# 
# ============================================================================

# End of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo "
"
