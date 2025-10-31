# mamba activate torch-env

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

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

# 1. 配置 4-bit 量化参数 (取代 Unsloth 的 load_in_4bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # 使用 NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 使用 bfloat16 进行计算，V100支持
    bnb_4bit_use_double_quant=True,
)

# 2. 准备数据 - 使用 step_list 作为 CoT 推理过程
ds = load_dataset("Kanan275/GSM8k-CoT", "default", split="train[:923]")

def to_chat(e):
    """
    将数据集格式转换为训练格式
    输入字段：
    - instruction: 问题文本（字符串）
    - step_list: CoT推理步骤（可能是字符串或列表）
    - final_answer: 最终答案（可能是字符串或列表）
    
    输出格式：
    - question: 问题文本
    - answer: <think>推理步骤</think>\n最终答案
    """
    import json
    import ast
    
    # 处理 step_list - 可能是字符串、JSON字符串或列表
    steps = e["step_list"]
    if isinstance(steps, str):
        # 尝试多种解析方式
        try:
            steps = json.loads(steps)
        except json.JSONDecodeError:
            try:
                # 使用 ast.literal_eval 处理 Python 字面量格式
                steps = ast.literal_eval(steps)
            except (ValueError, SyntaxError):
                # 如果都失败，将整个字符串作为单个步骤
                steps = [steps]
    
    # 合并步骤为文本
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
                pass  # 保持原字符串
    elif isinstance(final_ans, list) and len(final_ans) > 0:
        final_ans = final_ans[0]
    
    final_ans = str(final_ans)
    
    # 构建完整答案：<think>推理过程</think>\n最终答案
    full_answer = f"<think>{think_process.strip()}</think>\n{final_ans.strip()}"
    return {"question": e['instruction'], "answer": full_answer}

ds = ds.map(to_chat)

# 3. 加载 Qwen-2.5-1.5B 模型和分词器
# 说明：使用 Transformers 的标准方法加载模型，并配合 BitsAndBytesConfig 进行 4-bit 量化
model_id = "Qwen/Qwen2.5-Coder-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(model_id)
# 确保 Qwen 的 pad_token 设置正确
tok.pad_token = tok.eos_token
model.config.use_cache = False

# 4. 配置 LoRA 参数 (与您原代码目的相同)
lora_config = LoraConfig(
    r=16, # R值
    lora_alpha=32, # Alpha值
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 定义训练参数 (SFTConfig - TRL 0.23.0 推荐使用)
training_args = SFTConfig(
    output_dir="qwen_peft_sft_lora",
    num_train_epochs=1,
    per_device_train_batch_size=1, # micro_batch_size
    gradient_accumulation_steps=4, # gradient_accumulation
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True, # V100 不支持 bf16，使用 fp16
    optim="paged_adamw_8bit", # 优化器
    dataset_text_field="text", # 指定文本字段名
)

# 重新准备数据，使用标准的 text 字段格式
def prepare_text(e):
    text = f"用户: {e['question']}\n助手: {e['answer']}"
    return {"text": text}

ds = ds.map(prepare_text)

# 6. 初始化并开始训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    peft_config=lora_config,
    processing_class=tok, # TRL 0.23.0 推荐使用 processing_class 而不是 tokenizer
)

trainer.train()