# COT-SFT-LLM-MATH

使用 Chain-of-Thought (CoT) 数据对 Qwen2.5-Coder-1.5B 进行监督微调（SFT），用于数学问题求解。

## 数据集格式

### 原始数据集 (Kanan275/GSM8k-CoT)

数据集字段：
- `index`: 样本索引 (int64)
- `instruction`: 问题文本 (string)
- `step_list`: CoT 推理步骤 (string, JSON格式的列表)
- `final_answer`: 最终答案 (string, JSON格式的列表)
- `confidence`: 置信度 (float64)
- `answer_type`: 答案类型 (string, "positive"/"negative"等)

### 数据样例

```json
{
  "index": 0,
  "instruction": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "step_list": "[\"Sure, let's break this down step by step:\", \"1. **April Sales**: Natalia sold clips to 48 friends in April.\", ...]",
  "final_answer": "[\"Final Answer: Natalia sold a total of 72 clips in April and May.\"]",
  "confidence": 0.5194,
  "answer_type": "positive"
}
```

### 训练数据格式

经过 `to_chat()` 函数处理后的格式：

```python
{
  "question": "问题文本 (来自 instruction 字段)",
  "answer": "<think>推理步骤1\n推理步骤2\n...</think>\n最终答案"
}
```

最终转换为文本格式：
```
用户: [问题文本]
助手: <think>[CoT推理过程]</think>
[最终答案]
```

### 数据处理流程

1. **加载原始数据集**
   ```python
   ds = load_dataset("Kanan275/GSM8k-CoT", "default", split="train")
   ```

2. **解析 JSON 字段**
   - `step_list`: JSON字符串 → Python列表 → 合并为文本
   - `final_answer`: JSON字符串 → Python列表 → 取第一个元素

3. **构建训练格式**
   - 将推理步骤放入 `<think>` 标签
   - 最终答案放在标签外

## 文件说明

- `1-qwen-cot-sft.py`: SFT 训练脚本
- `2-qwen-cot-inference.py`: 模型推理和评估脚本
- `3-qwen-grpo-train.py`: GRPO 强化学习训练脚本（未完成）

## 训练配置

- 模型: Qwen2.5-Coder-1.5B
- 量化: 4-bit NF4
- LoRA: r=16, alpha=32
- 优化器: paged_adamw_8bit
- 精度: fp16 (V100 GPU)

## 注意事项

1. **硬件要求**: 
   - 需要支持 CUDA 的 GPU
   - **PyTorch 2.8+ 最低要求 CUDA capability 7.0**
   - ✅ 推荐 GPU：A40、A100、L40S、V100
   - ❌ 不兼容 GPU：P100（CUDA 6.0 太旧）

2. **GPU 选择建议（CARC Discovery）**:
   - `a40:1` - **推荐首选**！兼容性好，性能强（CUDA 8.6）
   - `v100:1` - 最低要求（CUDA 7.0）
   - `a100:1` - 高性能但资源紧张（CUDA 8.0）
   - `l40s:1` - 最强但资源最紧张（CUDA 8.9）

3. **bf16 支持**: 
   - 如果 GPU 不支持 bf16（如 V100），需要改用 fp16
   - 代码已配置为 `fp16=True`

4. **数据格式问题**: 
   - `step_list` 和 `final_answer` 是**类似 JSON 但包含 Python 转义字符**的字符串
   - 不能直接用 `json.loads()` 解析（会报 `JSONDecodeError`）
   - 需要使用 `ast.literal_eval()` 作为备选方案
   - 代码已经实现了多层容错机制：先尝试 JSON 解析，失败后尝试 AST 解析，最后兜底处理