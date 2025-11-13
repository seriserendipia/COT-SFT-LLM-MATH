#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# 测试任务配置说明
# ============================================================================
# 用途：测试 answer_utils.py 模块的功能
# 数据量：仅加载 5 条样本用于验证
# 预计运行时间：< 5 分钟
# 
# 任务说明：
# 1. 加载 GSM8K-CoT 数据集前 5 条
# 2. 展示原始数据结构
# 3. 演示答案提取的每一步过程
# 4. 验证比较逻辑是否正确
# 5. 模拟准确率计算流程
# ============================================================================

# ============================================================================
# 资源配置（测试任务 - 最小配置）
# ============================================================================
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"

# ============================================================================
# 邮件通知配置
# ============================================================================
#SBATCH --mail-user=yihelu@usc.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# GPU 配置（测试任务不需要特定 GPU，接受任何可用的）
# ============================================================================
# 测试任务主要验证数据处理逻辑，对 GPU 性能要求不高
# 不指定 GPU 型号可以更快获得资源
# 
# 如果需要指定特定 GPU（不推荐用于测试）：
# #SBATCH --gpus-per-task=v100:1
# #SBATCH --gpus-per-task=a40:1
# ============================================================================

echo ""
echo "=========================================="
echo "  Answer Utils 模块测试任务"
echo "=========================================="
echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo ""

# Move to the correct directory
cd /home1/yihelu/csci566/
echo "Current working directory is `pwd`"
echo ""

# ============================================================================
# 环境配置
# ============================================================================
echo "🔧 Configuring environment..."
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh

echo "📦 Activating torch-env conda environment..."
conda activate torch-env

# 验证环境
echo ""
echo "🔍 Environment verification:"
python --version
echo "Python path: $(which python)"
echo ""

# ============================================================================
# 运行测试
# ============================================================================
echo "=========================================="
echo "  开始运行测试脚本"
echo "=========================================="
echo ""

python test_answer_utils.py

# 保存退出码
TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "  测试完成"
echo "=========================================="
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ 测试成功完成！"
else
    echo "❌ 测试失败，退出码: $TEST_EXIT_CODE"
fi

echo ""
echo "Program finished at: `date`"
echo ""

# 退出并返回测试脚本的退出码
exit $TEST_EXIT_CODE
