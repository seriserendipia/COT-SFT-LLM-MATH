# mamba activate torch-env
# 多 GPU GRPO 训练版本（使用 accelerate）
# 
# ============================================================================
# 重要：4-bit 量化 + 多 GPU DDP 训练的正确启动方式
# ============================================================================
# ✅ 正确方式（使用配置文件）：
#    accelerate launch --config_file accelerate_config_2gpu.yaml 5-qwen-sft-grpo-train.py
# ============================================================================

import os
import torch
import glob
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import GRPOTrainer, GRPOConfig
from answer_utils import compare_answers
import numpy as np
import json
from datetime import datetime

# 检查 GPU 可用性
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("WARNING: CUDA not available, will use CPU (very slow!)")

# ============================================================================
# 配置：路径与参数
# ============================================================================
# 输入：SFT 阶段生成的 Adapter 路径 (由 1-qwen-cot-sft.py 生成)
SFT_BASE_DIR = "./qwen_peft_sft_lora"  # SFT LoRA 基础目录

# 输出：使用 Job ID 创建唯一目录（支持并行训练）
job_id = os.environ.get("SLURM_JOB_ID", "default")
OUTPUT_DIR = f"./qwen_sft_grpo_lora_{job_id}"
print(f"\n📁 Training Job ID: {job_id}")
print(f"📁 Output Directory: {OUTPUT_DIR}")

# ============================================================================
# 训练指标记录（用于可视化）
# ============================================================================
reward_call_count = 0
reward_logs = []
training_metrics = []

# 1. 数据准备
# ----------------------------------------------------------------------------
# 🚀 生产模式 (全量数据: 7470 条)
ds = load_dataset("ankner/gsm8k-CoT", split="train")

# 🧪 测试模式 (10条样本) - 调试时可用
# ds = load_dataset("ankner/gsm8k-CoT", split="train[:10]")
# ----------------------------------------------------------------------------

def to_chat(e):
    # 格式与 SFT 保持一致，但 GRPO 需要 prompt 和 ground_truth
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
        "ground_truth": e["answer"].strip()
    }

ds = ds.map(to_chat)

# 2. 模型加载 (Base + SFT → Merge → 准备 GRPO)
model_id = "Qwen/Qwen2.5-Coder-1.5B"

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ============================================================================
# 多 GPU 设备配置：使用 LOCAL_RANK 环境变量
# ============================================================================
# accelerate launch 会为每个 GPU 启动一个独立进程：
# - 进程 0: LOCAL_RANK=0 → GPU 0
# - 进程 1: LOCAL_RANK=1 → GPU 1
# 4-bit 量化模型必须在加载时就绑定到正确的设备
local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"\n📍 Process LOCAL_RANK: {local_rank}")

print(f"Loading Base Model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={'': local_rank},  # 多 GPU 关键：每个进程加载到对应 GPU
    torch_dtype=torch.float16,
)

# ============================================================================
# 方案 B：分层 LoRA 训练
# ============================================================================
# 步骤 1：智能查找并加载 SFT Adapter（最新 checkpoint）
print(f"Searching for SFT checkpoints in: {SFT_BASE_DIR}")
sft_checkpoints = glob.glob(os.path.join(SFT_BASE_DIR, "checkpoint-*"))

if sft_checkpoints:
    # 按 checkpoint 编号排序，选择最新的
    sft_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    SFT_ADAPTER_PATH = sft_checkpoints[-1]
    print(f"Found {len(sft_checkpoints)} SFT checkpoints, using latest: {SFT_ADAPTER_PATH}")
elif os.path.exists(os.path.join(SFT_BASE_DIR, "adapter_config.json")):
    # 如果根目录有配置文件（最佳模型），直接使用
    SFT_ADAPTER_PATH = SFT_BASE_DIR
    print(f"Using SFT adapter from root directory: {SFT_ADAPTER_PATH}")
else:
    raise FileNotFoundError(f"No SFT checkpoints found in {SFT_BASE_DIR}. Please run script 1 first.")

print(f"Loading SFT Adapter from: {SFT_ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
print("✅ SFT Adapter loaded.")

# 步骤 2：合并 SFT 权重到 base 模型（关键步骤）
print("Merging SFT adapter into base model...")
model = model.merge_and_unload()
print("✅ SFT weights merged. Now the base model includes SFT knowledge.")
# 现在 model = Base + SFT（固化），原始 SFT LoRA 参数不再需要

tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token
model.config.use_cache = False

# 3. 配置新的 LoRA（专门用于 GRPO 训练）
# 这个 LoRA 将在 (Base+SFT) 的基础上学习 GRPO 的增量知识
grpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. 训练参数配置
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    
    # ========================================================================
    # 🚀 生产参数 (全量训练: 7470 条数据)
    # ========================================================================
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # 有效批次 = 2*4 = 8
    num_generations=8,              # 增加采样数量以获得更好的基线
    max_completion_length=512,      # 允许更长的推理链
    
    # ========================================================================
    # 🧪 测试参数 (快速验证) - 调试时可用
    # ========================================================================
    # per_device_train_batch_size=2,
    # gradient_accumulation_steps=2,
    # num_generations=4,
    # max_completion_length=128,
    
    # 通用参数
    learning_rate=5e-6, # ⚠️ 方案B可以用稍大的学习率，因为是在固化的SFT基础上训练新LoRA
    beta=0.01,          # KL 惩罚系数
    logging_steps=10,   # 多 GPU 下减少日志频率
    save_strategy="steps",
    save_steps=100,      # 🚀 生产模式：每 100 步保存
    save_total_limit=10, # 保留最新 10 个 checkpoint
    fp16=True,
    gradient_checkpointing=False, # ⚠️ DDP+LoRA 模式下必须关闭
    dataloader_num_workers=4,  # 多 GPU 数据加载加速
    dataloader_pin_memory=True,
    report_to="none",
)

# 5. 奖励函数（增强版：详细统计和日志）
def reward_func(completions, ground_truth=None, **kwargs):
    """
    GRPO 奖励函数：评估最终答案的正确性并记录详细统计信息
    
    记录指标：
    - mean_reward: 平均奖励
    - std_reward: 奖励标准差
    - accuracy: 准确率
    - num_correct/wrong: 正确/错误数量
    """
    global reward_call_count, reward_logs
    reward_call_count += 1
    
    rewards = []
    for i, completion in enumerate(completions):
        if ground_truth is None:
            rewards.append(0.0)
            continue
        # 使用你的工具函数比较
        if compare_answers(completion, ground_truth[i]):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    # ============================================================================
    # 统计奖励分布并保存日志
    # ============================================================================
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
    }
    reward_logs.append(log_entry)
    
    # 每5次打印统计信息（测试模式下更频繁）
    if reward_call_count % 5 == 0:
        print(f"\n{'='*70}")
        print(f"📊 Reward Statistics (Call #{reward_call_count}):")
        print(f"   Mean: {mean_reward:.4f} | Std: {std_reward:.4f} | Acc: {accuracy:.2%}")
        print(f"   Correct: {int(accuracy*len(rewards))}/{len(rewards)}")
        print(f"{'='*70}\n")
    
    return rewards

# 6. 开始训练
# ⚠️ 关键：传入 peft_config，让 GRPO 在 (Base+SFT) 上训练一个新的 LoRA
trainer = GRPOTrainer(
    model=model,  # 这是已经包含 SFT 的 base 模型
    args=training_args,
    train_dataset=ds,
    peft_config=grpo_lora_config,  # 新的 LoRA 层（方案 B 的关键）
    processing_class=tok,
    reward_funcs=reward_func,
)

print("🚀 Starting SFT+GRPO Training (Two-Layer LoRA Approach)...")
print(f"Architecture: Base + SFT (merged) + GRPO LoRA (new)")
print(f"Input SFT Adapter: {SFT_ADAPTER_PATH} (will be merged)")
print(f"Output GRPO Adapter: {OUTPUT_DIR} (incremental LoRA)")

# 训练并捕获训练历史
train_result = trainer.train()

# 保存最终模型
trainer.save_model(OUTPUT_DIR)

# ============================================================================
# 保存训练指标用于可视化
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 保存奖励统计日志
if reward_logs:
    reward_log_file = os.path.join(OUTPUT_DIR, "reward_statistics.json")
    with open(reward_log_file, "w") as f:
        json.dump(reward_logs, f, indent=2)
    print(f"\n💾 Reward logs saved to: {reward_log_file}")
    print(f"📊 Total reward function calls: {reward_call_count}")

# 2. 保存训练历史（loss, learning_rate 等）
if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
    training_log_file = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(training_log_file, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"💾 Training history saved to: {training_log_file}")

# 3. 生成训练摘要
training_summary = {
    "job_id": job_id,
    "start_time": datetime.now().isoformat(),
    "model_id": model_id,
    "sft_adapter_path": SFT_ADAPTER_PATH,
    "output_dir": OUTPUT_DIR,
    "num_samples": len(ds),
    "num_epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "learning_rate": training_args.learning_rate,
    "beta": training_args.beta,
    "num_generations": training_args.num_generations,
    "max_completion_length": training_args.max_completion_length,
    "total_reward_calls": reward_call_count,
}

# 添加最终奖励统计
if reward_logs:
    final_rewards = [log["accuracy"] for log in reward_logs]
    training_summary["final_avg_accuracy"] = float(np.mean(final_rewards))
    training_summary["final_max_accuracy"] = float(np.max(final_rewards))
    training_summary["final_min_accuracy"] = float(np.min(final_rewards))

summary_file = os.path.join(OUTPUT_DIR, "training_summary.json")
with open(summary_file, "w") as f:
    json.dump(training_summary, f, indent=2)
print(f"💾 Training summary saved to: {summary_file}")

print(f"\n✅ Training finished. Model saved to {OUTPUT_DIR}")
print(f"\n📊 Quick Summary:")
print(f"   Total Reward Calls: {reward_call_count}")
if reward_logs:
    print(f"   Final Accuracy: {training_summary.get('final_avg_accuracy', 0):.2%}")
print(f"   Logs available for visualization in: {OUTPUT_DIR}")
