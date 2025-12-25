# GRPO Training Hyperparameters Summary
# GRPO 训练超参数总结

This document summarizes the hyperparameters used for training the Qwen-2.5-Coder-1.5B model using GRPO (Group Relative Policy Optimization).
本文档总结了使用 GRPO (Group Relative Policy Optimization) 训练 Qwen-2.5-Coder-1.5B 模型时使用的超参数。

## 1. Model Configuration (模型配置)

| Parameter (参数) | Value (值) | Description (说明) |
| :--- | :--- | :--- |
| **Base Model** (基础模型) | `Qwen/Qwen2.5-Coder-1.5B` | The pre-trained model used as a starting point. <br> 用作起点的预训练模型。 |
| **Quantization** (量化) | 4-bit (NF4) | Loaded in 4-bit using NF4 quantization to save memory. <br> 使用 NF4 量化加载 4-bit 模型以节省显存。 |
| **Compute Dtype** (计算精度) | `torch.float16` | Computation is performed in float16. <br> 计算在 float16 下进行。 |
| **Double Quant** (双重量化) | `True` | Enabled for further memory savings. <br> 启用以进一步节省显存。 |

## 2. LoRA Configuration (LoRA 配置)

| Parameter (参数) | Value (值) | Description (说明) |
| :--- | :--- | :--- |
| **Rank (r)** (秩) | 16 | The rank of the low-rank matrices. <br> 低秩矩阵的秩。 |
| **Alpha** | 32 | Scaling factor for LoRA updates (usually 2x rank). <br> LoRA 更新的缩放因子（通常为秩的 2 倍）。 |
| **Target Modules** (目标模块) | All Linear Layers | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. <br> 所有线性层。 |
| **Dropout** | 0.05 | Dropout probability for LoRA layers. <br> LoRA 层的 Dropout 概率。 |
| **Bias** (偏置) | "none" | No bias terms are trained. <br> 不训练偏置项。 |
| **Task Type** (任务类型) | `CAUSAL_LM` | Causal Language Modeling. <br> 因果语言建模。 |

## 3. Training Configuration (训练配置)

| Parameter (参数) | Value (值) | Description (说明) |
| :--- | :--- | :--- |
| **Epochs** (轮数) | 1 | Number of training epochs. <br> 训练轮数。 |
| **Batch Size (Per Device)** (单卡批次) | 2 | Batch size per GPU. <br> 每个 GPU 的批次大小。 |
| **Gradient Accumulation** (梯度累积) | 2 | Number of steps to accumulate gradients before update. <br> 更新前累积梯度的步数。 |
| **Total Batch Size** (总批次) | 8 (for 2 GPUs) | `2 (per_device) * 2 (accum) * 2 (GPUs)`. <br> 总批次大小。 |
| **Gradient Checkpointing** (梯度检查点) | `False` | Disabled due to compatibility issues with DDP + LoRA. <br> 由于与 DDP + LoRA 的兼容性问题而禁用。 |
| **FP16** | `True` | Mixed precision training. <br> 混合精度训练。 |

## 4. Generation Configuration (生成配置)

These parameters control how the model generates responses during the exploration phase of GRPO.
这些参数控制模型在 GRPO 探索阶段如何生成回复。

| Parameter (参数) | Value (值) | Description (说明) |
| :--- | :--- | :--- |
| **Num Generations** (生成数量) | 4 | Number of candidate responses generated per prompt. <br> 每个提示生成的候选回复数量。 |
| **Generation Batch Size** (生成批次) | 4 | Batch size for generation. <br> 生成时的批次大小。 |
| **Max Completion Length** (最大生成长度) | 256 | Maximum tokens for the generated response. <br> 生成回复的最大 token 数。 |
| **Max Prompt Length** (最大提示长度) | 1024 | Maximum tokens for the input prompt. <br> 输入提示的最大 token 数。 |
| **Temperature** (温度) | 0.9 | Controls randomness (higher = more random). <br> 控制随机性（越高越随机）。 |
| **Top P** | 0.9 | Nucleus sampling probability. <br> 核采样概率。 |
| **Top K** | 50 | Top-K sampling. <br> Top-K 采样。 |

## 5. Optimizer & Scheduler (优化器与调度器)

| Parameter (参数) | Value (值) | Description (说明) |
| :--- | :--- | :--- |
| **Optimizer** (优化器) | `paged_adamw_8bit` | Memory-efficient optimizer. <br> 内存高效的优化器。 |
| **Learning Rate** (学习率) | 5e-5 | Initial learning rate. <br> 初始学习率。 |
| **Warmup Ratio** (预热比例) | 0.1 (10%) | Percentage of steps for learning rate warmup. <br> 学习率预热的步数比例。 |
| **Max Grad Norm** (最大梯度范数) | 1.0 | Gradient clipping threshold. <br> 梯度裁剪阈值。 |

## 6. GRPO Specifics (GRPO 特有参数)

| Parameter (参数) | Value (值) | Description (说明) |
| :--- | :--- | :--- |
| **Beta** (KL Penalty) | 0.01 | KL divergence penalty coefficient. Keeps the model close to the reference. <br> KL 散度惩罚系数。保持模型接近参考模型。 |
| **Reward Function** (奖励函数) | Outcome-based | **1.0** for correct answer, **0.0** for incorrect. <br> 基于结果的奖励：正确为 1.0，错误为 0.0。 |

## 7. Hardware & Environment (硬件与环境)

| Parameter (参数) | Value (值) | Description (说明) |
| :--- | :--- | :--- |
| **Multi-GPU Strategy** (多卡策略) | DDP (Distributed Data Parallel) | Managed via `accelerate`. <br> 通过 `accelerate` 管理。 |
| **Device Map** (设备映射) | `{'': LOCAL_RANK}` | Ensures correct model placement on GPUs. <br> 确保模型正确放置在 GPU 上。 |
