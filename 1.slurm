#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# 资源配置推荐（针对 ankner/gsm8k-CoT 数据集）
# ============================================================================
# 数据集      | 样本数 | 时间      | 内存  | CPU | GPU    | 说明
# ------------|--------|----------|-------|-----|--------|------------------
# 测试        | 10     | 00:15:00 | 16G   | 4   | A100×1 | 快速测试
# 小规模      | 100    | 00:20:00 | 20G   | 8   | A100×1 | 开发调试
# 中等规模    | 1000   | 00:45:00 | 24G   | 8   | A100×1 | 中等规模
# 完整训练    | 7470   | 03:00:00 | 32G   | 12  | A100×1 | 完整训练（当前配置）
# ============================================================================
# 
# 📊 计算过程（完整训练）：
#   - 有效批次大小: 1 × 4 = 4
#   - 总步数: 7470 / 4 = 1868 steps
#   - 估计时间: 1868 × 2s = 3736s ≈ 1小时 + 开销20分钟 = 1.5小时
#   - 安全余量 (×2): 3小时
# 
# 💾 显存使用（A100 40GB）：
#   - 模型（4-bit）: 4GB
#   - LoRA + 优化器 + 激活值: 8-10GB
#   - 剩余余量: ~26GB ✅ 充足
# ============================================================================

# ============================================================================
# GPU 策略：指定 A100 获得最佳性能
# ============================================================================
# 可用 GPU 型号（按性能排序）：
#   l40s (CUDA 8.9) > a100 (CUDA 8.0) > a40 (CUDA 8.6) > v100 (CUDA 7.0)
#   ⚠️ p100 (CUDA 6.0) 不兼容 PyTorch 2.8+
# 
# A100 优势：
#   ✅ 40GB/80GB 显存充足
#   ✅ 训练速度快（约 V100 的 2倍）
#   ✅ 支持 TF32/BF16 自动加速
#   ⚠️ 排队可能较长（USC CARC 热门GPU）
# 
# 灵活策略：如果排队过长，可改为 --gpus-per-task=1 接受任意GPU
# ============================================================================



## 完整训练配置（7470样本）
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
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

# Move to the correct directory
cd /home1/yihelu/csci566/
echo "Current working directory is `pwd`"

# Load Python module if required
# module load python  # Adjust based on your system if necessary
# module load conda

# conda init
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh

conda activate torch-env

python 1-qwen-cot-sft.py

# Run Jupyter Notebook with papermill
# papermill HW2_microsoft_malware_prediction__yihelu.ipynb output_notebook.ipynb

# End of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
