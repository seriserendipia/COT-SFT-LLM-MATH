# COT-SFT-LLM-MATH


Chain-of-Thought 推理的数学问题求解：结合监督学习 (SFT) 和强化学习 (GRPO) 的多阶段训练方案。

---

## 📋 实验设计

### 消融实验矩阵

| 编号 | 方法 | 训练脚本 | 评估脚本 | 模型架构 | 输出目录 | 实验目的 |
|------|------|---------|---------|---------|---------|---------|
| 0 | **Baseline** | - | `0-qwen-baseline-inference.py` | Base | `inference_results_baseline/` | 零样本基线 |
| 1-2 | **SFT** | `1-qwen-cot-sft.py` | `2-qwen-cot-inference.py` | Base + SFT LoRA | `qwen_peft_sft_lora/`<br>`inference_results_sft/` | 监督学习效果 |
| 3-4 | **GRPO (Base)** | `3-5-qwen-grpo-train-multigpu.py` | `4-qwen-grpo-inference.py` | Base + GRPO LoRA | `qwen_grpo_lora_multigpu/`<br>`inference_results_grpo/` | 纯强化学习效果 |
| 5-6 | **SFT+GRPO** | `5-qwen-sft-grpo-train.py` | `6-qwen-sft-grpo-eval.py` | Base + SFT + GRPO LoRA | `qwen_sft_grpo_lora/`<br>`inference_results_sft_grpo_full/` | 联合训练协同效应 |
| 7 | **GRPO-only (ablation)** | - | `7-qwen-sftbased-grpo-only-eval.py` | Base + GRPO LoRA (no SFT) | `inference_results_grpo_only/` | GRPO 独立能力测试 |

### 训练策略说明

- **方案 A（脚本 3-4）**：从 Base 模型直接用 GRPO 训练
- **方案 B（脚本 5-7）**：分层 LoRA
  - 加载 SFT LoRA → `merge_and_unload()` → 在新 base 上训练 GRPO LoRA
  - 优点：SFT 和 GRPO 权重分离，可独立评估各自贡献

---

## 🚀 快速开始

### 环境配置
```bash
conda activate torch-env
```

### 训练流程

#### 单独提交
```bash
# 1. SFT 训练
sbatch 1.slurm

# 2. 评估 SFT
sbatch 2.slurm

# 3. GRPO 训练（从 Base）
sbatch 3-5-multigpu.slurm

# 4. 评估 GRPO
sbatch 4.slurm

# 5. SFT+GRPO 联合训练
sbatch 5.slurm

# 6. 评估完整模型
sbatch 6.slurm

# 7. 评估 GRPO 独立能力（消融）
sbatch 7.slurm
```

#### 批量提交（自动等待前一步完成）

**方法 1：使用 `--dependency` 参数**
```bash
JOB5=$(sbatch --parsable 5.slurm)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 6.slurm)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5 7.slurm)
```

**方法 2：一键脚本（创建 `submit_567.sh`）**
```bash
#!/bin/bash
# 文件名: submit_567.sh
# 用途: 按顺序提交脚本 5、6、7

echo "🚀 Starting SFT+GRPO training and evaluation pipeline..."

# 1. 提交训练任务（脚本 5）
echo "📝 Submitting training job (5.slurm)..."
JOB5=$(sbatch --parsable 5.slurm)

if [ -z "$JOB5" ]; then
    echo "❌ Error: Failed to submit job 5"
    exit 1
fi
echo "✅ Training job submitted: Job ID = $JOB5"

# 2. 提交评估任务（脚本 6，依赖脚本 5）
echo "📝 Submitting full evaluation job (6.slurm)..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 6.slurm)

if [ -z "$JOB6" ]; then
    echo "❌ Error: Failed to submit job 6"
    exit 1
fi
echo "✅ Full evaluation job submitted: Job ID = $JOB6 (depends on $JOB5)"

# 3. 提交消融评估任务（脚本 7，依赖脚本 5）
echo "📝 Submitting GRPO-only evaluation job (7.slurm)..."
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5 7.slurm)

if [ -z "$JOB7" ]; then
    echo "❌ Error: Failed to submit job 7"
    exit 1
fi
echo "✅ GRPO-only evaluation job submitted: Job ID = $JOB7 (depends on $JOB5)"

# 4. 总结
echo ""
echo "=========================================="
echo "📊 Job Dependency Summary"
echo "=========================================="
echo "Job $JOB5: Training (5.slurm) - Running now"
echo "Job $JOB6: Full eval (6.slurm) - Waits for $JOB5"
echo "Job $JOB7: GRPO-only eval (7.slurm) - Waits for $JOB5"
echo "=========================================="
echo ""
echo "💡 Monitor jobs with: squeue -u $USER"
echo "💡 Check logs: tail -f slurm-*.out"
```

使用方法：
```bash
# 给脚本添加执行权限
chmod +x submit_567.sh

# 执行
./submit_567.sh
```

**方法 3：PowerShell 版本（Windows 本地测试用）**
```powershell
# 文件名: submit_567.ps1
Write-Host "🚀 Starting SFT+GRPO training and evaluation pipeline..." -ForegroundColor Green

# 1. 提交训练任务
Write-Host "📝 Submitting training job (5.slurm)..." -ForegroundColor Cyan
$job5Output = sbatch --parsable 5.slurm
$JOB5 = $job5Output.Trim()

if ([string]::IsNullOrEmpty($JOB5)) {
    Write-Host "❌ Error: Failed to submit job 5" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Training job submitted: Job ID = $JOB5" -ForegroundColor Green

# 2. 提交完整评估任务
Write-Host "📝 Submitting full evaluation job (6.slurm)..." -ForegroundColor Cyan
$job6Output = sbatch --parsable --dependency=afterok:$JOB5 6.slurm
$JOB6 = $job6Output.Trim()

if ([string]::IsNullOrEmpty($JOB6)) {
    Write-Host "❌ Error: Failed to submit job 6" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Full evaluation job submitted: Job ID = $JOB6 (depends on $JOB5)" -ForegroundColor Green

# 3. 提交 GRPO-only 评估任务
Write-Host "📝 Submitting GRPO-only evaluation job (7.slurm)..." -ForegroundColor Cyan
$job7Output = sbatch --parsable --dependency=afterok:$JOB5 7.slurm
$JOB7 = $job7Output.Trim()

if ([string]::IsNullOrEmpty($JOB7)) {
    Write-Host "❌ Error: Failed to submit job 7" -ForegroundColor Red
    exit 1
}
Write-Host "✅ GRPO-only evaluation job submitted: Job ID = $JOB7 (depends on $JOB5)" -ForegroundColor Green

# 4. 总结
Write-Host ""
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "📊 Job Dependency Summary" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "Job $JOB5: Training (5.slurm) - Running now"
Write-Host "Job $JOB6: Full eval (6.slurm) - Waits for $JOB5"
Write-Host "Job $JOB7: GRPO-only eval (7.slurm) - Waits for $JOB5"
Write-Host "==========================================" -ForegroundColor Yellow
```

---

## 📊 关键指标追踪

### 训练阶段
- **脚本 5 输出**：
  - `reward_statistics.json` - 奖励函数详细统计
  - `training_history.json` - Loss、学习率曲线
  - `training_summary.json` - 训练配置和最终指标

### 评估阶段
所有评估脚本输出：
- `results.json` - 每个样本的详细结果
- `evaluation_summary.json` - 汇总统计
- `error_cases.json` - 错误案例分析
- `visualization_data.json` - 可视化数据（用于绘图）

---

## 🔧 待改进策略

### 1. Checkpoint 管理与评估策略

#### 当前策略（实用方案）

**训练时保存**：
```python
# 脚本 5 配置
save_strategy="steps"
save_steps=100              # 每 100 步保存一次
save_total_limit=5          # 保留最新 5 个 checkpoint
```

**评估时加载**：
- 脚本 6、7 自动加载**最新的 checkpoint**（参考脚本 2、4 的逻辑）
- 不硬编码路径，智能查找 `checkpoint-*` 目录

#### 评估成本分析

**单次全量评估成本**：
- 测试集：1,319 条数据
- 时间：~3 小时（基于实际运行数据）
- 平均：~8 秒/条

**评估所有 checkpoint 的成本**：
- 如果保留 10 个 checkpoint → 10 × 3 小时 = **30 小时**
- GPU 时间和计算资源成本非常高

**当前方案（默认使用最新）**：
- ✅ 最新 checkpoint 通常接近最佳（训练收敛）
- ✅ 评估成本：1 次 × 3 小时 = **3 小时**
- ✅ 保留 5 个 checkpoint 以备未来分析（无需重新训练）

#### 可选的深度评估方案

如果最新 checkpoint 结果不理想，可以手动评估其他 checkpoint：

**方案 1：基于训练日志选择**
```python
# 查看训练时的 reward 统计
reward_logs = json.load("qwen_sft_grpo_lora/reward_statistics.json")
accuracies = [log['accuracy'] for log in reward_logs]

# 找出训练时 accuracy 峰值对应的 checkpoint
peak_steps = find_peaks(accuracies)
# 手动评估这些峰值 checkpoint
```

**方案 2：快速筛选 + 完整评估**
```bash
# 步骤 1：用 100 条数据快速筛选所有 checkpoint（每次 2 分钟）
for cp in checkpoint-*; do
    python eval.py --checkpoint=$cp --test_size=100
done

# 步骤 2：只对 Top 3 跑完整测试集（3 × 3 小时）
```

**方案 3：不评估，直接使用**
- 训练参数合理的情况下，最新 checkpoint 就是最好的选择
- 节省 27 小时评估时间（90% 成本）

#### 为什么 GRPO 不能像 SFT 一样自动选最佳？

**SFT（脚本 1）可以**：
```python
load_best_model_at_end=True        # ✅ 自动保存最佳
dataset_kwargs={"val_size": 0.1}   # ✅ 自动划分验证集
```
- 监督学习简单，验证只需计算 loss

**GRPO（脚本 3-5、5）不行**：
- 强化学习需要生成多个候选（`num_generations=4-8`）
- 需要运行奖励函数对比
- TRL 库的 `GRPOConfig` 不支持 `dataset_kwargs`
- 验证成本高（每次验证 = 跑完整生成流程）

**结论**：保留多个 checkpoint + 默认使用最新 = 平衡效率和灵活性

### 2. 评估策略增强
**当前问题**：
- 只用贪心解码（`temperature=0.0`）
- GRPO 训练时生成多个候选（`num_generations=4`），但评估时只生成 1 次

**改进方案**：
- [ ] **Self-Consistency**：生成 5 次取多数投票答案
  - Google 论文证明在 CoT 任务上有效
  - 实现：`model.generate(..., do_sample=True, temperature=0.7, num_return_sequences=5)`
  - 对 5 个答案提取结果，取众数作为最终答案
- [ ] **采样评估**：`temperature=0.7` 测试生成多样性
- [ ] **Beam Search**：`num_beams=3` 对比贪心解码

**注意**：GRPO 训练时的多次生成是为了计算奖励对比，评估时可以用不同策略。

### 3. 奖励函数优化
**当前问题**：
- 二值奖励（0/1）稀疏信号
- 无格式奖励（格式错误和答案错误无法区分）

**改进方案**：
- [ ] **分层奖励**：
  ```python
  if 格式正确:
      if 答案正确: reward = 1.0
      else: reward = 0.3  # 格式对但答案错
  else:
      reward = 0.0  # 格式错
  ```
- [ ] **部分正确奖励**：数量级正确给 0.5

### 4. 训练数据增强
**当前问题**：
- 只用数学任务数据
- 可能存在灾难性遗忘（其他能力退化）

**改进方案**：
- [ ] 检查其他任务性能（代码生成、常识推理等）
- [ ] 混合其他来源训练数据（如 FLAN、OpenOrca）
- [ ] 多任务联合训练

### 5. 验证集评估
**当前缺失**：
- 训练时没有在验证集上评估
- 无法监控过拟合

**改进方案**：
- [ ] 每 N 步在验证集（50 条）上评估准确率
- [ ] 记录验证曲线用于 Early Stopping
- [ ] 对比训练集/验证集准确率监控泛化

### 6. 错误分析深化
**当前分析**：
- 只分了"格式错误"和"总错误"

**改进方案**：
- [ ] **问题类型分类**：百分比、分数、几何、多步推理等
- [ ] **错误模式识别**：计算错误、理解错误、单位转换错误
- [ ] **可视化错误分布**：按问题长度、复杂度分组

---

## 📈 可视化示例

```python
import json
import matplotlib.pyplot as plt

# 1. 训练奖励曲线
with open('qwen_sft_grpo_lora/reward_statistics.json') as f:
    rewards = json.load(f)
accuracies = [r['accuracy'] for r in rewards]
plt.plot(accuracies)
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('GRPO Training Reward Curve')

# 2. 评估指标对比
with open('inference_results_sft_grpo_full/visualization_data.json') as f:
    data = json.load(f)
plt.scatter(data['generation_lengths'], data['correctness'])
plt.xlabel('Generation Length')
plt.ylabel('Correct (1) or Wrong (0)')
```

---

## 🎯 消融实验预期结果

| 模型 | 预期准确率 | 说明 |
|------|-----------|------|
| Baseline | ~10% | 零样本基线 |
| SFT | 50-70% | 监督学习效果 |
| GRPO (Base) | 15-25% | 纯强化学习（无 SFT 基础） |
| GRPO-only (脚本7) | ~12% | SFT-based GRPO 脱离 SFT 后性能差 → 证明依赖 SFT |
| **SFT+GRPO** | **60-75%** | **最优：联合训练协同效应** |

---

## 📚 参考资料

- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **TRL Library**: [Transformers Reinforcement Learning](https://github.com/huggingface/trl)
- **Self-Consistency**: [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

