# mamba activate torch-env

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

# 1. 配置 4-bit 量化参数 (取代 Unsloth 的 load_in_4bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # 使用 NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 使用 bfloat16 进行计算，V100支持
    bnb_4bit_use_double_quant=True,
)

# 2. 准备数据（与您原代码相同）
ds = load_dataset("openai/gsm8k", "main", split="train[:200]")
def to_chat(e):
    think, ans = e["answer"].split("####")
    # 使用 SFTTrainer 兼容的格式，它会处理为对话格式
    full_answer = f"<think>{think.strip()}</think>\n{ans.strip()}"
    return {"question": e['question'], "answer": full_answer}
ds = ds.map(to_chat)

# 3. 加载 Qwen-2.5-1.5B 模型和分词器
# 说明：使用 Transformers 的标准方法加载模型，并配合 BitsAndBytesConfig 进行 4-bit 量化
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

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
    bf16=True, # 启用 bfloat16 优化
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