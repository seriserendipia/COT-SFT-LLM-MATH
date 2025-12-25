#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# 资源配置推荐（针�?ankner/gsm8k-CoT 数据集）
# ============================================================================
# 数据�?     | 样本�?| 时间      | 内存  | CPU | GPU    | 说明
# ------------|--------|----------|-------|-----|--------|------------------
# 测试        | 10     | 00:15:00 | 12G   | 4   | A100×1 | 快速验�?# 完整推理    | 1320   | 03:00:00 | 20G   | 8   | A100×1 | 完整测试集（当前配置�?# ============================================================================
# 
# 📊 计算过程（完整推理）�?#   - 测试集样�? 1320
#   - 每样本时�? ~5�?(生成256 tokens @ 50 tokens/s)
#   - 纯推理时�? 1320 × 5s = 6600s �?1.83小时
#   - 加载+保存开销: +20分钟
#   - 安全余量 (×1.5): 3小时
# 
# 💾 显存使用（A100 40GB）：
#   - 模型�?-bit�? LoRA: 4.5GB
#   - KV缓存: 2-3GB
#   - 剩余余量: ~33GB �?非常充足
# ============================================================================

# ============================================================================
# GPU 策略：指�?A100 获得最佳性能
# ============================================================================
# 可用 GPU 型号（按性能排序）：
#   l40s (CUDA 8.9) > a100 (CUDA 8.0) > a40 (CUDA 8.6) > v100 (CUDA 7.0)
#   ⚠️ p100 (CUDA 6.0) 不兼�?PyTorch 2.8+
# 
# 推理任务特点�?#   - 对GPU性能要求相对训练更低
#   - A100: 推理速度最快（~50 tokens/s�?#   - A40/V100: 性能也足够（~30-40 tokens/s�?# 
# 灵活策略：如果排队过长，可改�?--gpus-per-task=1 接受任意GPU
# ============================================================================



## 完整推理配置�?320样本推理�?#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"

# ============================================================================
# 邮件通知配置
# ============================================================================
#SBATCH --mail-user=your_email@example.com
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


python 2-qwen-cot-inference.py

# Run Jupyter Notebook with papermill
# papermill HW2_microsoft_malware_prediction__yihelu.ipynb output_notebook.ipynb

# End of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
