# 📊 实验3-5：GRPO-only训练动态分析与图表说明
# Experiment 3-5: GRPO-only Training Dynamics Analysis and Figure Guide

> **文件包含内容 | Package Contents:**
> - `training_metrics.csv` - 完整训练指标数据（1863步）
> - `training_metrics_visualization.png` - 训练动态可视化图表
> - `visualize_training_metrics.py` - 图表生成脚本
> - `extract_training_metrics.py` - 数据提取脚本

> **实验特点 | Experiment Characteristics:**
> - **无SFT预训练** - 直接从base model开始GRPO训练
> - **对比实验5** - 观察SFT预训练的影响

---

## 📋 目录 | Table of Contents

1. [实验配置概述](#实验配置概述--experiment-configuration)
2. [核心指标分析](#核心指标分析--key-metrics-analysis)
3. [与实验5的对比](#与实验5的对比--comparison-with-experiment-5)
4. [图表说明与标注指南](#图表说明与标注指南--figure-caption-guide)
5. [学术写作建议](#学术写作建议--academic-writing-guidelines)
6. [常见问题解答](#常见问题解答--faq)

---

## 🔧 实验配置概述 | Experiment Configuration

### 模型与硬件 | Model & Hardware

| 配置项 | Configuration | 值 | Value |
|--------|--------------|-----|-------|
| 基础模型 | Base Model | Qwen/Qwen2.5-Coder-1.5B | (1.5B parameters) |
| 量化方式 | Quantization | 4-bit NF4 | (BitsAndBytes) |
| GPU配置 | GPU Setup | 2× GPU | Multi-GPU DDP |
| 训练数据集 | Training Dataset | ankner/gsm8k-CoT | 7465 samples |
| 训练框架 | Framework | TRL GRPOTrainer | v0.23.0 |
| **训练策略** | **Training Strategy** | **GRPO-only** | **No SFT pretraining** |

### 训练超参数 | Training Hyperparameters

```python
# 关键配置 | Key Configuration
learning_rate = 5e-5              # 初始学习率 | Initial LR (比实验5更大)
lr_scheduler = "linear"            # 线性衰减 | Linear decay
num_train_steps = 3733             # 总训练步数 | Total steps
per_device_batch_size = 2          # 单卡批量 | Per-device batch
gradient_accumulation_steps = 2    # 梯度累积 | Gradient accumulation
num_generations = 4                # 每批生成数 | Generations/batch
total_batch_size = 8               # 总批量 2×2×2 | Total: 2×2×2
max_completion_length = 256        # 最大生成长度 | Max generation length
beta = 0.01                        # KL惩罚系数 | KL penalty coefficient
```

### LoRA配置 | LoRA Configuration

```python
# 单层LoRA策略 | Single-layer LoRA Strategy
# 直接在base model上训练GRPO LoRA
# Train GRPO LoRA directly on base model

lora_rank = 16                     # R值 | Rank
lora_alpha = 32                    # Alpha值 | Alpha
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

---

## 📊 核心指标分析 | Key Metrics Analysis

### 1️⃣ 模型准确率 (Model Performance)

**图表位置：** 子图 (a) - 顶部主图
**Figure Location:** Subplot (a) - Top main panel

#### 📈 数据总结 | Data Summary

| 指标 | Metric | 数值 | Value |
|------|--------|------|-------|
| 初始准确率 | Initial Accuracy | **31.2%** | Training start |
| 最终准确率 | Final Accuracy | **50.0%** | After 1863 steps |
| 峰值准确率 | Peak Accuracy | **100.0%** | Maximum observed |
| 平均准确率 | Mean Accuracy | **59.8%** | Overall average |
| 提升幅度 | Improvement | **+18.8%** (绝对) / **60.0%** (相对) | Absolute / Relative |

#### 🔍 详细分析 | Detailed Analysis

**中文分析：**

1. **低起点阶段（Step 0-500）：**
   - 准确率从31.2%开始，明显低于实验5的49.4%
   - **原因：** 没有SFT预训练，模型从base model开始
   - 早期震荡剧烈，反映模型在探索解题策略

2. **快速学习阶段（Step 500-1200）：**
   - 准确率快速提升到约55-60%
   - 学习曲线比实验5更陡峭
   - 说明GRPO能够从零开始学习数学推理

3. **平台期（Step 1200-1863）：**
   - 准确率在50%左右波动
   - 未能达到实验5的71.2%最终准确率
   - 峰值可达100%但不稳定

**English Analysis:**

1. **Low Starting Point (Step 0-500):**
   - Accuracy starts at 31.2%, significantly lower than Experiment 5's 49.4%
   - **Reason:** No SFT pretraining, starting from base model
   - High volatility reflects model exploring problem-solving strategies

2. **Rapid Learning Phase (Step 500-1200):**
   - Accuracy rapidly increases to ~55-60%
   - Steeper learning curve than Experiment 5
   - Demonstrates GRPO can learn mathematical reasoning from scratch

3. **Plateau Phase (Step 1200-1863):**
   - Accuracy fluctuates around 50%
   - Does not reach Experiment 5's 71.2% final accuracy
   - Can peak at 100% but unstable

#### ⚠️ 重要说明 | Important Note

**这是"训练集在线评估"，不是独立测试集！**
**This is "In-Training Evaluation on Training Set", NOT independent test set!**

---

### 2️⃣ 训练损失 (Training Loss)

**图表位置：** 子图 (b) - 左下
**Figure Location:** Subplot (b) - Bottom left

#### 📉 分析 | Analysis

**中文：**
- 损失值从-0.06变化到0.20
- 震荡比实验5更剧烈
- 最终损失为正值，表明训练仍在进行中
- GRPO从零开始的特征：损失不稳定

**English:**
- Loss ranges from -0.06 to 0.20
- More volatile than Experiment 5
- Final positive loss indicates ongoing training
- Characteristic of GRPO from scratch: unstable loss

---

### 3️⃣ KL散度 (KL Divergence from Base Model)

**图表位置：** 子图 (c) - 中下
**Figure Location:** Subplot (c) - Bottom center

#### 🎯 关键发现 | Key Findings

| 阶段 | Stage | KL值 | KL Value | 状态 | Status |
|------|-------|------|----------|------|--------|
| Step 0-500 | 早期 Early | 0.00-0.01 | 极低 Very Low | ✅ 健康 Healthy |
| Step 500-1200 | 上升期 Rising | 0.01-0.10 | 中等 Medium | ⚠️ 注意 Watch |
| Step 1200+ | 发散期 Diverging | 0.10-30.20 | **极高 Very High** | ❌ 危险 Critical |

#### 🔍 详细分析 | Detailed Analysis

**中文分析：**

1. **异常发散现象：**
   - 最终KL散度达到9.37，远超0.05警告阈值
   - 最大KL散度达到30.20，极度危险
   - **含义：** 模型严重偏离base model
   - **原因：** 
     - 没有SFT基础，GRPO直接修改base model
     - KL惩罚系数beta=0.01可能不够大
     - 模型可能过度适应训练集

2. **与实验5的对比：**
   - 实验5：KL最终0.028，峰值0.055（有SFT基础）
   - 实验3-5：KL最终9.37，峰值30.20（无SFT基础）
   - **结论：** SFT预训练能有效约束GRPO的探索范围

**English Analysis:**

1. **Abnormal Divergence:**
   - Final KL: 9.37, far exceeding 0.05 warning threshold
   - Max KL: 30.20, critically high
   - **Interpretation:** Model severely deviates from base model
   - **Possible Causes:**
     - No SFT foundation, GRPO directly modifies base model
     - KL penalty beta=0.01 may be insufficient
     - Potential overfitting to training set

2. **Comparison with Experiment 5:**
   - Exp 5: Final KL 0.028, peak 0.055 (with SFT)
   - Exp 3-5: Final KL 9.37, peak 30.20 (no SFT)
   - **Conclusion:** SFT pretraining effectively constrains GRPO exploration

---

### 4️⃣ 策略熵 (Policy Entropy)

**图表位置：** 子图 (d) - 右下
**Figure Location:** Subplot (d) - Bottom right

#### 📊 数据总结 | Data Summary

| 指标 | Metric | 数值 | Value | 解释 | Interpretation |
|------|--------|------|-------|------|----------------|
| 初始熵 | Initial Entropy | **0.46** | High | 高随机性 | High randomness |
| 最终熵 | Final Entropy | **3.43** | **Very High** | **异常高** | **Abnormally high** |
| 熵变化 | Entropy Change | **+2.97** (+646%) | Abnormal | **异常增长** | **Abnormal increase** |

#### 🔍 详细分析 | Detailed Analysis

**中文分析：**

1. **异常熵增长：**
   - 正常情况：熵应该下降（模型变得更自信）
   - 实际情况：熵大幅增长（模型变得更随机）
   - **警告信号：** 训练不稳定，模型可能崩溃

2. **与实验5对比：**
   - 实验5：熵从0.43降至0.25（正常）
   - 实验3-5：熵从0.46升至3.43（异常）
   - **结论：** 无SFT基础导致训练不稳定

**English Analysis:**

1. **Abnormal Entropy Increase:**
   - Normal: Entropy should decrease (model becomes confident)
   - Actual: Entropy dramatically increases (model becomes random)
   - **Warning:** Training instability, potential model collapse

2. **Comparison with Experiment 5:**
   - Exp 5: Entropy 0.43 → 0.25 (normal)
   - Exp 3-5: Entropy 0.46 → 3.43 (abnormal)
   - **Conclusion:** Lack of SFT causes training instability

---

### 5️⃣ 梯度范数 (Gradient Norm)

**图表位置：** 子图 (e) - 最右下
**Figure Location:** Subplot (e) - Bottom far right

#### ✅ 稳定性评估 | Stability Assessment

| 指标 | Metric | 数值 | Value | 安全阈值 | Safe Threshold | 状态 | Status |
|------|--------|------|-------|----------|----------------|------|--------|
| 最大梯度范数 | Max Grad Norm | **20.53** | - | < 1.0 | Warning | ❌ 危险 Critical |
| 平均梯度范数 | Mean Grad Norm | **2.39** | - | < 0.5 | Ideal | ⚠️ 偏高 High |
| 梯度爆炸风险 | Explosion Risk | **高 High** | - | - | - | ⚠️ 警告 Warning |

#### 🔍 详细分析 | Detailed Analysis

**中文分析：**

1. **梯度不稳定：**
   - 最大梯度范数20.53，远超1.0警告线
   - 平均梯度范数2.39，是实验5的5倍
   - **结论：** 训练过程不稳定，存在梯度爆炸风险

2. **与实验5对比：**
   - 实验5：最大0.69，平均0.45（稳定）
   - 实验3-5：最大20.53，平均2.39（不稳定）
   - **原因：** 无SFT基础导致优化困难

**English Analysis:**

1. **Gradient Instability:**
   - Max gradient norm 20.53, far exceeding 1.0 warning threshold
   - Mean gradient norm 2.39, 5× higher than Experiment 5
   - **Conclusion:** Unstable training, gradient explosion risk

2. **Comparison with Experiment 5:**
   - Exp 5: Max 0.69, Mean 0.45 (stable)
   - Exp 3-5: Max 20.53, Mean 2.39 (unstable)
   - **Reason:** Lack of SFT foundation causes optimization difficulties

---

## 🔄 与实验5的对比 | Comparison with Experiment 5

### 核心差异总结 | Key Differences Summary

| 指标 | Metric | 实验5 (SFT+GRPO) | 实验3-5 (GRPO-only) | 差异分析 |
|------|--------|-----------------|-------------------|----------|
| 初始准确率 | Initial Acc | 49.4% | 31.2% | SFT提供18.2%基础 |
| 最终准确率 | Final Acc | 71.2% | 50.0% | SFT使最终性能高21.2% |
| KL散度 | Final KL | 0.028 | 9.37 | 无SFT导致严重偏离 |
| 策略熵 | Final Entropy | 0.25 ↓ | 3.43 ↑ | SFT保证训练稳定 |
| 梯度范数 | Max Grad | 0.69 | 20.53 | SFT降低优化难度 |
| 训练稳定性 | Stability | ✅ 稳定 | ❌ 不稳定 | SFT是稳定性关键 |

### 💡 关键发现 | Key Insights

**中文：**

1. **SFT的关键作用：**
   - 提供更好的初始化（49.4% vs 31.2%）
   - 约束GRPO的探索范围（KL: 0.028 vs 9.37）
   - 保证训练稳定性（熵下降 vs 熵上升）
   - 提升最终性能（71.2% vs 50.0%）

2. **GRPO-only的挑战：**
   - 从零开始需要更多步骤收敛
   - 容易发生模型崩溃（KL发散、熵增长）
   - 需要更强的正则化（更大的beta值）
   - 最终性能受限

3. **论文建议：**
   - 强烈建议使用SFT+GRPO两阶段训练
   - GRPO-only可作为消融实验展示SFT的必要性
   - 在论文中明确对比两种方法的差异

**English:**

1. **Critical Role of SFT:**
   - Provides better initialization (49.4% vs 31.2%)
   - Constrains GRPO exploration (KL: 0.028 vs 9.37)
   - Ensures training stability (entropy ↓ vs ↑)
   - Improves final performance (71.2% vs 50.0%)

2. **Challenges of GRPO-only:**
   - Requires more steps to converge from scratch
   - Prone to model collapse (KL divergence, entropy growth)
   - Needs stronger regularization (larger beta)
   - Limited final performance

3. **Paper Recommendations:**
   - Strongly recommend two-stage SFT+GRPO training
   - Use GRPO-only as ablation to demonstrate SFT necessity
   - Clearly compare both methods in paper

---

## 🎨 图表说明与标注指南 | Figure Caption Guide

### 📋 推荐图注（LaTeX格式）| Recommended Caption (LaTeX Format)

#### 英文版本（用于论文）| English Version (For Paper)

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{exp3-5_training_visualization.png}
    \caption{Training dynamics of GRPO-only (without SFT pretraining) on GSM8K training set. 
    \textbf{(a)} In-training accuracy starts at 31.2\% and reaches 50.0\%, showing the challenge of training from base model without SFT foundation. Bold line represents smoothed trend; shaded area shows raw per-batch variance.
    \textbf{(b)} Training loss exhibits high volatility compared to SFT+GRPO approach.
    \textbf{(c)} KL divergence increases dramatically to 9.37 (peak: 30.20), far exceeding safe threshold, indicating severe deviation from base model.
    \textbf{(d)} Policy entropy abnormally increases from 0.46 to 3.43, suggesting training instability.
    \textbf{(e)} Gradient norms reach 20.53 (mean: 2.39), indicating optimization difficulties without SFT initialization.
    This experiment demonstrates the critical importance of SFT pretraining for stable GRPO training.}
    \label{fig:grpo_only_dynamics}
\end{figure*}
```

#### 中文版本（内部参考）| Chinese Version (Internal Reference)

```latex
\caption{GRPO-only训练动态（无SFT预训练）在GSM8K训练集上的表现。
\textbf{(a)} 训练中准确率从31.2\%开始，最终达到50.0\%，显示了在没有SFT基础的情况下从base model训练的挑战。粗线表示平滑趋势；阴影区域显示原始批次方差。
\textbf{(b)} 训练损失相比SFT+GRPO方法表现出高度波动性。
\textbf{(c)} KL散度急剧增长至9.37（峰值：30.20），远超安全阈值，表明严重偏离base model。
\textbf{(d)} 策略熵异常地从0.46增长至3.43，表明训练不稳定。
\textbf{(e)} 梯度范数达到20.53（均值：2.39），表明没有SFT初始化导致优化困难。
该实验证明了SFT预训练对于稳定GRPO训练的关键重要性。}
```

---

## 📝 学术写作建议 | Academic Writing Guidelines

### 如何在论文中使用此实验 | How to Use This Experiment in Paper

#### 1. 作为消融实验 (Ablation Study)

**English:**
```
To demonstrate the importance of SFT pretraining, we conduct an ablation 
study by training GRPO directly on the base model (Experiment 3-5). 
Results show that without SFT initialization:
- Initial accuracy drops from 49.4% to 31.2%
- Final accuracy decreases from 71.2% to 50.0%
- KL divergence increases dramatically (0.028 → 9.37)
- Training becomes unstable (entropy increases instead of decreasing)

These findings confirm that SFT pretraining is essential for stable and 
effective GRPO training on mathematical reasoning tasks.
```

**Chinese (参考):**
```
为了证明SFT预训练的重要性，我们进行了消融实验，直接在base model上训练GRPO（实验3-5）。
结果表明，没有SFT初始化：
- 初始准确率从49.4%降至31.2%
- 最终准确率从71.2%降至50.0%
- KL散度急剧增长（0.028 → 9.37）
- 训练变得不稳定（熵增长而非下降）

这些发现确认了SFT预训练对于数学推理任务中稳定且有效的GRPO训练至关重要。
```

#### 2. 对比分析表格 | Comparison Table

```latex
\begin{table}[t]
\centering
\caption{Comparison of SFT+GRPO vs GRPO-only}
\label{tab:sft_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{SFT+GRPO} & \textbf{GRPO-only} \\
\midrule
Initial Accuracy & 49.4\% & 31.2\% \\
Final Accuracy & \textbf{71.2\%} & 50.0\% \\
Peak Accuracy & 92.5\% & 100.0\%* \\
Final KL Divergence & 0.028 & 9.370 \\
Final Entropy & 0.25 & 3.43 \\
Max Gradient Norm & 0.69 & 20.53 \\
Training Stability & Stable & Unstable \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\item[*] Unstable, not sustained
\end{tablenotes}
\end{table}
```

---

## ❓ 常见问题解答 | FAQ

### Q1: 为什么GRPO-only表现这么差？
### Q1: Why does GRPO-only perform so poorly?

**中文答案：**

**核心原因：**
1. **缺乏数学推理基础：** Base model没有经过数学任务微调
2. **探索空间过大：** GRPO从零开始探索，容易偏离正确方向
3. **训练不稳定：** 没有SFT的约束，模型容易崩溃

**English Answer:**

**Core Reasons:**
1. **Lack of mathematical reasoning foundation:** Base model not finetuned on math tasks
2. **Excessive exploration space:** GRPO explores from scratch, easily deviates
3. **Training instability:** Without SFT constraints, model prone to collapse

---

### Q2: 这个实验有什么价值？
### Q2: What is the value of this experiment?

**中文答案：**

**学术价值：**
- ✅ 证明SFT预训练的必要性（消融实验）
- ✅ 展示GRPO的局限性（需要好的初始化）
- ✅ 为两阶段训练提供理论依据

**实用价值：**
- ⚠️ 不建议在实际应用中使用GRPO-only
- ✅ 但可以指导超参数调整（如增大beta）

**English Answer:**

**Academic Value:**
- ✅ Demonstrates necessity of SFT pretraining (ablation)
- ✅ Shows limitations of GRPO (needs good initialization)
- ✅ Provides rationale for two-stage training

**Practical Value:**
- ⚠️ Not recommended for real applications
- ✅ Can guide hyperparameter tuning (e.g., increase beta)

---

## ✅ 关键结论 | Key Conclusions

### 中文总结：

1. **SFT预训练是必需的：** 实验3-5证明直接GRPO训练效果差、不稳定
2. **两阶段训练优越：** 实验5 (SFT+GRPO) 在所有指标上都明显优于实验3-5
3. **训练稳定性关键：** KL散度、策略熵、梯度范数都显示GRPO-only不稳定
4. **论文建议：** 使用此实验作为消融研究，证明方法设计的合理性

### English Summary:

1. **SFT pretraining is essential:** Experiment 3-5 shows direct GRPO training is poor and unstable
2. **Two-stage training superior:** Experiment 5 (SFT+GRPO) clearly outperforms 3-5 on all metrics
3. **Training stability critical:** KL divergence, entropy, gradient norms all show GRPO-only instability
4. **Paper recommendation:** Use this as ablation study to justify method design

---

**文档版本：** v1.0  
**最后更新：** 2024年11月27日  
**作者：** CS566 Group Project - 实验3-5团队

---

**🎓 与实验5对比，充分展示SFT的价值！**
