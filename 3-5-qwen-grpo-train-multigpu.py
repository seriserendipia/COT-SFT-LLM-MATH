# mamba activate torch-env
# 多GPU GRPO训练版本（使用 accelerate）

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import torch
import time
import os

# ============================================================================
# 性能分析工具
# ============================================================================
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        print(f"\n⏱️  [{self.name}] 开始...")
        return self
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"✅ [{self.name}] 完成，耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
        return False

# 检查GPU可用性
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查是否有不兼容的 P100
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        if "P100" in gpu_name:
            print("\n" + "="*70)
            print(f"❌ ERROR: GPU {i} is P100 (CUDA 6.0), not compatible with PyTorch 2.8+")
            print("   Minimum requirement: CUDA capability 7.0 (V100 or newer)")
            print("   Please resubmit the job to get different GPUs.")
            print("="*70)
            import sys
            sys.exit(1)
else:
    print("WARNING: CUDA not available, will use CPU (very slow!)")

# 1. 配置 4-bit 量化参数
with Timer("量化配置"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 使用 NF4 量化
        bnb_4bit_compute_dtype=torch.float16,  # 使用 float16（与 fp16 训练一致）
        bnb_4bit_use_double_quant=True,
    )

# 2. 准备数据 - 使用 step_list 作为 CoT 推理过程
with Timer("数据集加载"):
    ds = load_dataset("Kanan275/GSM8k-CoT", "default", split="train[:10]")

def to_chat(e):
    """
    将数据集格式转换为 GRPO 训练格式
    输入字段：
    - instruction: 问题文本（字符串）
    - step_list: CoT推理步骤（可能是字符串或列表）
    - final_answer: 最终答案（可能是字符串或列表）
    
    输出格式：
    - prompt: 问题文本（GRPO 必需字段，不是 query！）
    - ground_truth: 正确答案（用于奖励函数评估）
    """
    import json
    import ast
    
    # 处理 step_list - 可能是字符串、JSON字符串或列表
    steps = e["step_list"]
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except json.JSONDecodeError:
            try:
                steps = ast.literal_eval(steps)
            except (ValueError, SyntaxError):
                steps = [steps]
    
    if isinstance(steps, list):
        think_process = "\n".join(str(s) for s in steps)
    else:
        think_process = str(steps)
    
    # 处理 final_answer - 可能是字符串、JSON字符串或列表
    final_ans = e["final_answer"]
    if isinstance(final_ans, str):
        try:
            final_ans = json.loads(final_ans)
            if isinstance(final_ans, list) and len(final_ans) > 0:
                final_ans = final_ans[0]
        except json.JSONDecodeError:
            try:
                final_ans = ast.literal_eval(final_ans)
                if isinstance(final_ans, list) and len(final_ans) > 0:
                    final_ans = final_ans[0]
            except (ValueError, SyntaxError):
                pass
    elif isinstance(final_ans, list) and len(final_ans) > 0:
        final_ans = final_ans[0]
    
    final_ans = str(final_ans)
    
    # GRPO 需要 'prompt' 字段（不是 query！）
    # 同时保存 ground_truth 用于奖励函数评估
    return {
        "prompt": e['instruction'],  # GRPO 训练器期望的字段名
        "ground_truth": final_ans.strip()  # 用于奖励函数验证答案正确性
    }

with Timer("数据预处理"):
    ds = ds.map(to_chat)

# 3. 加载 Qwen-2.5-1.5B 模型和分词器
model_id = "Qwen/Qwen2.5-Coder-1.5B"

with Timer("模型加载"):
    # ⚠️ 4-bit量化 + 多GPU训练的特殊要求（TRL文档）:
    # 必须使用 device_map={'': torch.cuda.current_device()} 或 {'': 0}
    # 这确保每个进程的模型加载到当前进程对应的GPU上
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={'': torch.cuda.current_device()},  # TRL推荐：多GPU+量化必须
        torch_dtype=torch.float16,  # 与量化和训练精度保持一致
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    # 确保 Qwen 的 pad_token 设置正确
    tok.pad_token = tok.eos_token
    model.config.use_cache = False

# 4. 配置 LoRA 参数（TRL 最佳实践）
lora_config = LoraConfig(
    r=16, # R值
    lora_alpha=32, # Alpha值（通常是r的2倍）
    # ✅ TRL推荐：target_modules="all-linear" 性能更好
    # 但对于小模型和测试，保持具体模块可减少参数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],  # 包含MLP层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 定义 GRPO 训练参数（基于TRL最佳实践优化）
training_args = GRPOConfig(
    output_dir="qwen_grpo_lora_multigpu",
    num_train_epochs=1,
    
    # ============================================================================
    # 批次配置（多GPU下每张卡的批次）
    # ============================================================================
    per_device_train_batch_size=2,  # 每个GPU的批次大小
    gradient_accumulation_steps=2,   # 减少累积步数（因为多GPU总批次已增大）
    # 如果有2个GPU：总批次 = 2 GPU × 2 batch × 2 accum = 8
    # 如果有4个GPU：总批次 = 4 GPU × 2 batch × 2 accum = 16
    
    # ============================================================================
    # 生成配置（GRPO核心参数）
    # ============================================================================
    num_generations=4,  # 每个prompt生成4个候选回答
    generation_batch_size=4,  # ✅ TRL推荐：明确设置生成批次大小
    max_completion_length=256,  # 缩短长度加速（测试用）
    max_prompt_length=1024,
    
    # ============================================================================
    # 采样参数（增加生成多样性）
    # ============================================================================
    temperature=0.9,  # TRL推荐：0.7-0.9之间
    top_p=0.9,
    top_k=50,  # ✅ 添加top_k采样
    
    # ============================================================================
    # 优化器配置（GRPO/RL训练需要小心设置）
    # ============================================================================
    learning_rate=5e-5,  # 参考TRL GRPO示例
    warmup_ratio=0.1,  # 10%步数warmup
    max_grad_norm=1.0,  # ✅ TRL推荐：梯度裁剪防止训练不稳定
    optim="paged_adamw_8bit",
    
    # ============================================================================
    # GRPO特定参数
    # ============================================================================
    beta=0.0,  # KL散度系数（0=不使用KL惩罚，参考DAPO论文）
    # loss_type="grpo",  # 默认值，可不设置
    
    # ============================================================================
    # 显存优化（多GPU + 4-bit量化）
    # ============================================================================
    gradient_checkpointing=True,  # ✅ TRL强烈推荐：节省显存
    fp16=True,  # 与量化dtype一致
    
    # ============================================================================
    # 数据加载加速
    # ============================================================================
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # ============================================================================
    # 日志和保存策略
    # ============================================================================
    logging_steps=2,  # 多GPU下更频繁日志
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    
    # ============================================================================
    # 分布式训练配置（accelerate自动处理）
    # ============================================================================
    # accelerate 会根据环境变量自动检测多GPU
)

# 6. 定义奖励函数（GRPO 需要）
def reward_func(completions, ground_truth=None, **kwargs):
    """
    GRPO 奖励函数：仅评估最终答案的正确性
    
    设计原则：
    - 每个输出对应一个标量奖励（outcome-based reward）
    - 只评估最终结果，不对推理过程、格式、长度等中间特征打分
    - GRPO 框架会自动归一化: r̃ᵢ = (rᵢ - mean(r)) / std(r)
    - 归一化后的奖励会分配给该输出的所有 token
    
    参数：
    - completions: 生成的文本列表（字符串列表）
    - ground_truth: 正确答案列表（必需，用于评估准确性）
    - **kwargs: 其他参数（prompts, completions_ids 等）
    
    返回: 
    - List[float]: 每个 completion 一个标量分数
        * 1.0: 答案完全正确
        * 0.0: 答案错误
    
    奖励机制说明：
    通过多次采样 (num_generations=4)，GRPO 会对比不同候选的奖励：
    - 高奖励样本（答案正确）→ 增加其生成概率
    - 低奖励样本（答案错误）→ 降低其生成概率
    模型会自然学到"哪些推理路径导致正确答案"，无需人工设计过程奖励
    """
    import re
    
    rewards = []
    
    for i, completion in enumerate(completions):
        # 默认奖励为 0（错误）
        reward = 0.0
        
        # 必须有 ground_truth 才能评估
        if ground_truth is None or i >= len(ground_truth):
            # 如果没有标准答案，无法判断对错，给予中性奖励
            rewards.append(0.5)
            continue
        
        # 1. 提取模型生成的最终答案
        # 策略：优先提取 </think> 标签后的内容，否则使用全文
        predicted_text = completion
        if "</think>" in completion:
            # 如果有 CoT 标签，提取标签后的答案部分
            predicted_text = completion.split("</think>")[-1].strip()
        
        # 2. 提取答案中的数字（GSM8K 数学题的答案通常是数字）
        # 使用正则提取所有数字（包括负数、小数）
        predicted_numbers = re.findall(r'-?\d+\.?\d*', predicted_text)
        gt = str(ground_truth[i]).strip()
        gt_numbers = re.findall(r'-?\d+\.?\d*', gt)
        
        # 3. 二元判断：答案正确 vs 错误
        if predicted_numbers and gt_numbers:
            # 比较最后一个数字（通常是最终答案）
            if predicted_numbers[-1] == gt_numbers[-1]:
                reward = 1.0  # ✅ 答案正确
            else:
                reward = 0.0  # ❌ 答案错误
        else:
            # 如果无法提取数字，尝试直接字符串匹配
            if gt.lower() in predicted_text.lower():
                reward = 1.0  # ✅ 文本匹配成功
            else:
                reward = 0.0  # ❌ 无法匹配
        
        rewards.append(float(reward))
    
    # 返回原始奖励，GRPO 框架会自动执行归一化：
    # r̃ᵢ = (rᵢ - mean(r)) / std(r)
    # 然后将 r̃ᵢ 分配给输出 oᵢ 的所有 token 作为优势函数
    return rewards

# 7. 初始化并开始训练
with Timer("训练器初始化"):
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tok,
        reward_funcs=reward_func,
    )

print("\n" + "="*80)
print("🚀 开始多GPU GRPO 训练...")
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"🎮 GPU数量: {gpu_count}")
print(f"📊 配置: {len(ds)} 样本 × {training_args.num_generations} 生成 = {len(ds) * training_args.num_generations} 次推理")
total_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * gpu_count
print(f"💾 总批次大小: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} × {gpu_count} = {total_batch}")
print("="*80 + "\n")

with Timer("GRPO 完整训练"):
    trainer.train()

print("\n✅ 训练完成！检查点保存在: qwen_grpo_lora_multigpu/")
