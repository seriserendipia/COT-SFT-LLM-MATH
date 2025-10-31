#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# 资源配置推荐（根据数据集大小选择）
# ============================================================================
# 数据集大小 | 时间       | 内存  | CPU | GPU | 说明
# -----------|-----------|-------|-----|-----|------------------
# 10 样本    | 00:15:00  | 16G   | 4   | 1   | 快速测试
# 100 样本   | 00:30:00  | 32G   | 8   | 1   | 小规模训练（当前配置）
# 1000 样本  | 02:00:00  | 32G   | 8   | 1   | 中等规模
# 全数据集   | 08:00:00  | 48G   | 16  | 1   | 完整训练 (7.5k样本)
# ============================================================================

# ============================================================================
# GPU 策略：不指定型号 = 接受所有可用 GPU（L40S/A100/A40/V100）
# 优点：大大提高获得资源速度，减少排队时间
# 注意：系统可能分配 P100（CUDA 6.0），代码会自动检测并报警告
# ============================================================================
# 可用 GPU 型号（按性能排序）：
#   l40s (CUDA 8.9) > a100 (CUDA 8.0) > a40 (CUDA 8.6) > v100 (CUDA 7.0)
#   ⚠️ p100 (CUDA 6.0) 不兼容 PyTorch 2.8+
# ============================================================================




# ============================================================================
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:30:00
#SBATCH --gpus-per-task=1

# ============================================================================
# 邮件通知配置
# ============================================================================
#SBATCH --mail-user=yihelu@usc.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# 如果排队太久，可取消以下注释指定特定GPU型号：
# #SBATCH --gpus-per-task=v100:1    # 推荐：性能够用，资源充足
# #SBATCH --gpus-per-task=a40:1 

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
