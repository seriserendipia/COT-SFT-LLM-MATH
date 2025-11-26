# mamba activate torch-env
# 多GPU GRPO训练版本（使用 accelerate）
# 
# ============================================================================
# 重要：4-bit 量化 + 多GPU DDP 训练的正确启动方式
# ============================================================================
# 
# ❌ 错误方式（会导致 device mismatch 错误）：
#    python 3-5-qwen-grpo-train-multigpu.py
# 
# ✅ 正确方式1（推荐，使用配置文件）：
#    accelerate launch --config_file accelerate_config_2gpu.yaml 3-5-qwen-grpo-train-multigpu.py
# 
# ✅ 正确方式2（命令行指定参数）：
#    accelerate launch --multi_gpu --num_processes=2 3-5-qwen-grpo-train-multigpu.py
# 
# ⚠️ 为什么需要 LOCAL_RANK：
# - accelerate launch 会为每个GPU启动一个独立的Python进程
# - 进程0: LOCAL_RANK=0 → 使用 GPU 0
# - 进程1: LOCAL_RANK=1 → 使用 GPU 1
# - 4-bit量化的模型必须在加载时就绑定到正确的设备
# - device_map={'': LOCAL_RANK} 确保每个进程的模型加载到对应的GPU
# 
# ============================================================================

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import torch
import time
import os
import numpy as np
import json
from datetime import datetime

# 导入统一的答案提取和比较工具
from answer_utils import compare_answers, parse_answer_field, extract_number_from_text

# ============================================================================
# 奖励统计日志（用于训练过程监控）
# ============================================================================
reward_call_count = 0
reward_logs = []

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

# 2. 准备数据 - 使用 response 作为 CoT 推理过程，answer 作为最终答案
with Timer("数据集加载"):
    ds = load_dataset("ankner/gsm8k-CoT", split="train")
    # 小批量测试（10个样本）：
    # ds = load_dataset("ankner/gsm8k-CoT", split="train[:10]")

def to_chat(e):
    """
    将数据集格式转换为 GRPO 训练格式
    输入字段（ankner/gsm8k-CoT）：
    - question: 问题文本（字符串）
    - response: CoT推理步骤（字符串）
    - answer: 最终答案（字符串）
    
    输出格式：
    - prompt: 问题文本 + 格式指导（GRPO 必需字段）
    - ground_truth: 正确答案（用于奖励函数评估）
    """
    # 直接使用新数据集的字段，无需复杂解析
    final_ans = e["answer"].strip()
    
    # ============================================================================
    # 格式指导Prompt：方案 3 - <ans> 标签版
    # ============================================================================
    # 优点：
    # - 清晰的职责分离：<think> 推理，<ans> 答案
    # - 提取逻辑极简：直接匹配 <ans>...</ans>
    # - 容错性好：保留回退机制
    # - Token 效率高（~110 tokens）
    # ============================================================================
    formatted_prompt = f"""Solve this math problem step by step.

Output format:
1. Wrap reasoning in <think>...</think>
2. Put final answer in <ans>...</ans> (number only, no text)

Example:
<think>
Price: $5
Quantity: 3
Total: $5 × 3 = $15
</think>
<ans>15</ans>

Problem: {e['question']}

Solution:"""
    
    return {
        "prompt": formatted_prompt,
        "ground_truth": final_ans
    }

with Timer("数据预处理"):
    ds = ds.map(to_chat)

# 3. 加载 Qwen-2.5-1.5B 模型和分词器
model_id = "Qwen/Qwen2.5-Coder-1.5B"

with Timer("模型加载"):
    # ⚠️ 4-bit量化 + 多GPU训练的特殊要求（TRL文档）:
    # 方案1：使用 device_map={'': int(os.environ.get("LOCAL_RANK", 0))}
    # 方案2：不使用 device_map，让 accelerate.prepare() 自动处理设备分配
    # 这里使用方案1，确保每个进程的模型加载到正确的GPU上
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={'': local_rank},  # ✅ 使用 LOCAL_RANK 环境变量
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
    beta=0.01,  # KL散度系数（2025-11-06修复：0.0→0.01）
    # 
    # 修复原因：
    # - beta=0.0 导致模型无约束偏离预训练分布
    # - 从预训练模型直接GRPO（无SFT基础）需要保持一定约束
    # - beta=0.01 是从预训练开始的推荐值（小惩罚，允许探索但不崩溃）
    # 
    # 调参建议：
    # - 如果模型输出混乱 → 增大beta (0.05, 0.1)
    # - 如果模型太保守不学习 → 减小beta (0.001, 0.005)
    # 
    # loss_type="grpo",  # 默认值，可不设置
    
    # ============================================================================
    # 显存优化（多GPU + 4-bit量化）
    # ============================================================================
    # ⚠️ DDP + gradient_checkpointing + LoRA 存在兼容性问题
    # 错误：RuntimeError: Expected to mark a variable ready only once
    # 
    # 解决方案：关闭 gradient_checkpointing（DDP 模式要求）
    # - DDP 使用 hook 机制跟踪参数梯度
    # - gradient_checkpointing 会重新计算导致参数被多次标记
    # - 与 LoRA 参数组合时触发冲突
    # 
    # 如果显存不足：
    # - 减小 per_device_train_batch_size
    # - 增加 gradient_accumulation_steps
    # - 减小 num_generations 或 max_completion_length
    gradient_checkpointing=False,  # ❌ DDP+LoRA 模式下必须关闭
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
    save_steps=100,       # 每 100 步保存一次
    save_total_limit=10,  # 保留最新 10 个 checkpoint（绘制训练曲线）
    
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
    
    注意：使用统一的答案提取和比较工具（answer_utils），确保与推理阶段一致
    """
    global reward_call_count, reward_logs
    reward_call_count += 1
    
    rewards = []
    
    for i, completion in enumerate(completions):
        # 默认奖励为 0（错误）
        reward = 0.0
        
        # 必须有 ground_truth 才能评估
        if ground_truth is None or i >= len(ground_truth):
            # 如果没有标准答案，无法判断对错，给予中性奖励
            rewards.append(0.5)
            continue
        
        # 使用统一的答案比较函数（支持数值比较）
        # 这确保了训练时的奖励计算与推理时的准确率计算完全一致
        if compare_answers(completion, ground_truth[i]):
            reward = 1.0  # ✅ 答案正确
        else:
            reward = 0.0  # ❌ 答案错误
        
        rewards.append(float(reward))
    
    # ============================================================================
    # 统计奖励分布并保存日志（仅主进程）
    # ============================================================================
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        rewards_array = np.array(rewards)
        mean_reward = rewards_array.mean()
        std_reward = rewards_array.std()
        accuracy = (rewards_array == 1.0).mean()
        
        # 记录到JSON日志
        log_entry = {
            "call_count": reward_call_count,
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(rewards),
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "accuracy": float(accuracy),
            "num_correct": int((rewards_array == 1.0).sum()),
            "num_wrong": int((rewards_array == 0.0).sum()),
            "num_neutral": int((rewards_array == 0.5).sum()),
        }
        reward_logs.append(log_entry)
        
        # 每10次打印统计信息
        if reward_call_count % 10 == 0:
            print(f"\n{'='*70}")
            print(f"📊 Reward Statistics (Call #{reward_call_count}):")
            print(f"   Mean: {mean_reward:.4f} | Std: {std_reward:.4f} | Acc: {accuracy:.2%}")
            print(f"   Correct: {int(accuracy*len(rewards))}/{len(rewards)}")
            print(f"{'='*70}\n")
        
        # 每50次保存日志到文件
        if reward_call_count % 50 == 0:
            output_dir = "qwen_grpo_lora_multigpu"
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "reward_statistics.json")
            with open(log_file, "w") as f:
                json.dump(reward_logs, f, indent=2)
            print(f"💾 Reward logs saved to: {log_file}")
    
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

# ============================================================================
# 保存最终奖励统计日志
# ============================================================================
if int(os.environ.get("LOCAL_RANK", "0")) == 0 and reward_logs:
    output_dir = "qwen_grpo_lora_multigpu"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "reward_statistics.json")
    with open(log_file, "w") as f:
        json.dump(reward_logs, f, indent=2)
    print(f"\n💾 Final reward logs saved to: {log_file}")
    print(f"📊 Total reward function calls: {reward_call_count}")
    print(f"📊 Total log entries: {len(reward_logs)}")

print("\n✅ 训练完成！检查点保存在: qwen_grpo_lora_multigpu/")
