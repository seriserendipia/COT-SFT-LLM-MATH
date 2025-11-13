# mamba activate torch-env

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import torch
import time

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
    print(f"GPU count: {torch.cuda.device_count()}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU name: {gpu_name}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # 检查是否是不兼容的 P100
    if "P100" in gpu_name:
        print("\n" + "="*70)
        print("❌ ERROR: P100 GPU (CUDA 6.0) is not compatible with PyTorch 2.8+")
        print("   Minimum requirement: CUDA capability 7.0 (V100 or newer)")
        print("   Please resubmit the job to get a different GPU.")
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
    - prompt: 问题文本（GRPO 必需字段，不是 query！）
    - ground_truth: 正确答案（用于奖励函数评估）
    """
    # 直接使用新数据集的字段，无需复杂解析
    final_ans = e["answer"].strip()
    
    # GRPO 需要 'prompt' 字段（不是 query！）
    # 同时保存 ground_truth 用于奖励函数评估
    return {
        "prompt": e['question'],  # GRPO 训练器期望的字段名
        "ground_truth": final_ans  # 用于奖励函数验证答案正确性
    }

with Timer("数据预处理"):
    ds = ds.map(to_chat)

# 3. 加载 Qwen-2.5-1.5B 模型和分词器
model_id = "Qwen/Qwen2.5-Coder-1.5B"

with Timer("模型加载"):
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
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
    # 但对于小模型和测试，包含主要线性层即可
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],  # 包含MLP层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 定义 GRPO 训练参数（基于 TRL 最佳实践优化）
training_args = GRPOConfig(
    output_dir="qwen_grpo_lora",
    num_train_epochs=1,
    
    # ============================================================================
    # 批次配置（利用 A100 80GB 的充足显存）
    # ============================================================================
    per_device_train_batch_size=2,  # 从 1 增加到 2（TRL 推荐值）
    gradient_accumulation_steps=4,   # 保持不变，有效批次 = 2*4 = 8
    
    # ============================================================================
    # 生成配置（GRPO核心参数）
    # ============================================================================
    num_generations=4,  # 每个prompt生成4个候选回答
    generation_batch_size=4,  # ✅ TRL推荐：明确设置生成批次大小
    max_completion_length=512,  # 适合数学 CoT 推理
    max_prompt_length=1024,
    
    # ============================================================================
    # 采样参数（增加生成多样性）
    # ============================================================================
    temperature=0.9,  # TRL 推荐值，增加探索
    top_p=0.9,        # Nucleus 采样
    top_k=50,         # ✅ 添加Top-K采样
    
    # ============================================================================
    # 优化器配置（RL 训练需要较小学习率）
    # ============================================================================
    learning_rate=5e-5,  # 从 2e-4 降低到 5e-5（参考 TRL LoRA 示例）
    warmup_ratio=0.1,    # 添加 warmup（10% 步数），防止早期训练不稳定
    max_grad_norm=1.0,   # ✅ TRL推荐：梯度裁剪防止训练不稳定
    optim="paged_adamw_8bit",
    
    # ============================================================================
    # GRPO特定参数
    # ============================================================================
    beta=0.0,  # KL散度系数（0=不使用KL惩罚，参考DAPO论文）
    
    # ============================================================================
    # 显存优化（4-bit量化 + 单GPU）
    # ============================================================================
    gradient_checkpointing=True,  # ✅ TRL强烈推荐：节省显存
    fp16=True,  # 使用 FP16（与量化一致）
    
    # ============================================================================
    # 数据加载加速（多进程预处理）
    # ============================================================================
    dataloader_num_workers=4,  # 利用8个CPU中的4个，加速数据加载
    dataloader_pin_memory=True,  # 加速 CPU->GPU 数据传输
    
    # ============================================================================
    # 日志和保存策略
    # ============================================================================
    logging_steps=5,     # 更频繁的日志（从 10 改为 5）
    save_strategy="steps",    # 按步数保存而非 epoch
    save_steps=50,            # 每 50 步保存一次
    save_total_limit=3,       # 只保留最近 3 个检查点
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
        processing_class=tok, # TRL 0.23.0 推荐使用 processing_class
        reward_funcs=reward_func, # 注意：参数名是 reward_funcs（复数），不是 reward_fn！
    )

print("\n" + "="*80)
print("🚀 开始 GRPO 训练...")
print(f"📊 配置: {len(ds)} 样本 × {training_args.num_generations} 生成 = {len(ds) * training_args.num_generations} 次推理")
print(f"💾 批次大小: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps} (有效)")
print("="*80 + "\n")

with Timer("GRPO 完整训练"):
    trainer.train()
