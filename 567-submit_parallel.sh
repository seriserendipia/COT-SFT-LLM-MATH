#!/bin/bash
# ============================================================================
# 并行任务提交脚本
# 用途：支持同时提交多个训练+评估任务，输出目录自动使用 Job ID 隔离
# ============================================================================

echo "🚀 提交 SFT+GRPO 训练和评估任务..."
echo ""

# 1. 提交训练任务，获取 Job ID
JOB5=$(sbatch --parsable 5.slurm)

if [ -z "$JOB5" ]; then
    echo "❌ 错误：训练任务提交失败"
    exit 1
fi

echo "✅ 训练任务已提交"
echo "   Job ID: $JOB5"
echo "   输出目录: qwen_sft_grpo_lora_${JOB5}/"
echo ""

# 2. 提交完整评估任务（依赖训练完成）
echo "📝 提交完整模型评估任务 (6.slurm)..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 --export=ALL,TRAINING_JOB_ID=$JOB5 6.slurm)

if [ -z "$JOB6" ]; then
    echo "❌ 错误：评估任务 6 提交失败"
else
    echo "✅ 完整评估任务已提交"
    echo "   Job ID: $JOB6"
    echo "   依赖: 训练任务 $JOB5 完成后执行"
    echo "   输出目录: inference_results_sft_grpo_${JOB5}/"
fi
echo ""

# 3. 提交消融评估任务（依赖训练完成）
echo "📝 提交消融评估任务 (7.slurm)..."
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5 --export=ALL,TRAINING_JOB_ID=$JOB5 7.slurm)

if [ -z "$JOB7" ]; then
    echo "❌ 错误：评估任务 7 提交失败"
else
    echo "✅ 消融评估任务已提交"
    echo "   Job ID: $JOB7"
    echo "   依赖: 训练任务 $JOB5 完成后执行"
    echo "   输出目录: inference_results_grpo_only_${JOB5}/"
fi
echo ""

# 4. 总结
echo "=========================================="
echo "📊 任务提交总结"
echo "=========================================="
echo "训练任务: $JOB5 (qwen_sft_grpo_lora_${JOB5}/)"
echo "完整评估: $JOB6 (等待 $JOB5)"
echo "消融评估: $JOB7 (等待 $JOB5)"
echo "=========================================="
echo ""
echo "💡 监控命令:"
echo "   查看队列: squeue -u \$USER"
echo "   查看日志: tail -f slurm-${JOB5}-*.out"
echo ""
echo "💡 要提交更多并行任务，直接再次运行此脚本即可"
