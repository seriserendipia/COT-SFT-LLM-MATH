# mamba activate torch-env
# GRPO 模型推理和评估脚本

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json
import re
from tqdm import tqdm
import os

# 检查GPU可用性
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU name: {gpu_name}")
    
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
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. 加载数据集（测试集：train[924:1319]）
print("Loading dataset...")
# 先用前10条测试
# ds = load_dataset("Kanan275/GSM8k-CoT", "default", split="train[924:934]")
# 完整测试集（取消注释下面一行）：
ds = load_dataset("Kanan275/GSM8k-CoT", "default", split="train[924:1319]")

def to_chat(e):
    """
    将数据集格式转换为推理格式
    输入字段：
    - instruction: 问题文本（字符串）
    - step_list: CoT推理步骤（可能是字符串或列表）
    - final_answer: 最终答案（可能是字符串或列表）
    
    输出格式：
    - question: 问题文本
    - answer: <think>推理步骤</think>\n最终答案
    - ground_truth: 最终答案（用于评估）
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
    
    # 构建完整答案
    full_answer = f"<think>{think_process.strip()}</think>\n{final_ans.strip()}"
    ground_truth = final_ans.strip()  # 用于评估的标准答案
    
    return {
        "question": e['instruction'], 
        "answer": full_answer,
        "ground_truth": ground_truth
    }

ds = ds.map(to_chat)
print(f"Loaded {len(ds)} test samples")

# 3. 加载基础模型
print("Loading base model...")
model_id = "Qwen/Qwen2.5-Coder-1.5B"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 4. 加载GRPO训练的LoRA权重
print("Loading GRPO LoRA adapter...")

# 自动查找最新的 checkpoint
import glob
base_lora_dir = "qwen_grpo_lora_multigpu"

# 查找所有 checkpoint-* 目录
checkpoint_dirs = glob.glob(os.path.join(base_lora_dir, "checkpoint-*"))

if checkpoint_dirs:
    # 按 checkpoint 编号排序，选择最大的（最新的）
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    lora_path = checkpoint_dirs[-1]  # 最后一个 = 编号最大 = 最新
    print(f"Found {len(checkpoint_dirs)} checkpoints, using latest: {lora_path}")
else:
    # 如果没有 checkpoint 子目录，尝试直接使用根目录
    lora_path = base_lora_dir
    print(f"No checkpoints found, using root directory: {lora_path}")

# 验证路径存在
if not os.path.exists(lora_path):
    print(f"\n{'='*70}")
    print(f"❌ ERROR: LoRA adapter directory not found!")
    print(f"   Looking for: {os.path.abspath(lora_path)}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"\n   Possible solutions:")
    print(f"   1. Run this script from the same directory where training was done")
    print(f"   2. Check if GRPO training completed successfully")
    print(f"   3. Use absolute path if needed")
    print("="*70)
    import sys
    sys.exit(1)

# 检查关键文件
adapter_config = os.path.join(lora_path, "adapter_config.json")
if not os.path.exists(adapter_config):
    print(f"\n{'='*70}")
    print(f"❌ ERROR: adapter_config.json not found in {lora_path}")
    print(f"   This file is required to load the LoRA adapter.")
    print(f"   GRPO training may not have completed successfully.")
    print("="*70)
    import sys
    sys.exit(1)

print(f"✅ Found GRPO LoRA adapter at: {os.path.abspath(lora_path)}")

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# 5. 加载分词器
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tok.pad_token = tok.eos_token

# 6. 推理函数
def extract_answer(text):
    """从生成的文本中提取最终答案"""
    # 尝试提取 </think> 之后的内容
    if "</think>" in text:
        answer = text.split("</think>")[-1].strip()
    else:
        answer = text.strip()
    
    # 提取数字
    numbers = re.findall(r'-?\d+\.?\d*', answer)
    if numbers:
        return numbers[-1]  # 返回最后一个数字
    return answer

def generate_answer(question, max_length=2048):
    """
    生成答案（使用确定性解码）
    
    生成参数说明：
    - temperature=0.0: 使用贪心解码（选择概率最高的token）
    - do_sample=False: 关闭随机采样
    - 这样可以保证每次运行结果一致，便于评估和对比
    """
    prompt = f"用户: {question}\n助手:"
    
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,           # ✅ 确定性生成（贪心解码）
            do_sample=False,           # ✅ 关闭采样
            pad_token_id=tok.eos_token_id,
        )
    
    generated_text = tok.decode(outputs[0], skip_special_tokens=True)
    # 移除输入部分
    response = generated_text[len(prompt):].strip()
    return response

# 7. 批量推理
print("\n" + "="*70)
print("🚀 Starting GRPO model inference...")
print(f"📊 Test samples: {len(ds)}")
print(f"🎯 Evaluation metric: Accuracy")
print(f"🔧 Generation mode: Deterministic (temperature=0.0)")
print("="*70 + "\n")

results = []
correct = 0
total = 0

for i, example in enumerate(tqdm(ds, desc="Evaluating")):
    question = example["question"]
    ground_truth = example["ground_truth"]
    
    # 生成答案
    generated_response = generate_answer(question)
    predicted_answer = extract_answer(generated_response)
    gt_answer = extract_answer(ground_truth)
    
    # 判断是否正确
    is_correct = predicted_answer == gt_answer
    if is_correct:
        correct += 1
    total += 1
    
    # 保存结果
    result = {
        "index": i,
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_extracted": gt_answer,
        "generated_response": generated_response,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct
    }
    results.append(result)
    
    # 每5个样本打印一次进度（测试集较小）
    if (i + 1) % 5 == 0:
        acc = correct / total
        print(f"\nProgress: {i+1}/{len(ds)}, Accuracy so far: {acc:.4f} ({acc*100:.2f}%)")

# 8. 计算评估指标
accuracy = correct / total
print(f"\n{'='*70}")
print(f"📈 Final Evaluation Results (GRPO Model):")
print(f"{'='*70}")
print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {total - correct}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*70}\n")

# 9. 保存结果
output_dir = "inference_results_grpo"  # GRPO 模型的推理结果
os.makedirs(output_dir, exist_ok=True)

# 保存详细推理结果
results_file = os.path.join(output_dir, "inference_results.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"✅ Inference results saved to: {results_file}")

# 保存评估指标
metrics = {
    "total_samples": total,
    "correct_predictions": correct,
    "wrong_predictions": total - correct,
    "accuracy": accuracy,
    "model_type": "GRPO",
    "model_path": lora_path,
    "base_model": model_id,
    "dataset": "Kanan275/GSM8k-CoT",
    "split": "train[924:934]",  # 更新为实际使用的split
    "generation_config": {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "do_sample": False,
        "method": "greedy_decoding"
    }
}

metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"✅ Evaluation metrics saved to: {metrics_file}")

# 10. 保存错误案例分析
wrong_cases = [r for r in results if not r["is_correct"]]
wrong_cases_file = os.path.join(output_dir, "wrong_cases.json")
with open(wrong_cases_file, "w", encoding="utf-8") as f:
    json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
print(f"✅ Wrong cases saved to: {wrong_cases_file}")
print(f"   Total wrong cases: {len(wrong_cases)}")

# 11. 打印一些示例结果
if len(results) > 0:
    print(f"\n{'='*70}")
    print("📝 Sample Results:")
    print(f"{'='*70}")
    
    # 打印第一个正确的案例
    correct_cases = [r for r in results if r["is_correct"]]
    if correct_cases:
        sample = correct_cases[0]
        print(f"\n✅ Correct Example (Index {sample['index']}):")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Predicted: {sample['predicted_answer']}")
        print(f"Ground Truth: {sample['ground_truth_extracted']}")
    
    # 打印第一个错误的案例
    if wrong_cases:
        sample = wrong_cases[0]
        print(f"\n❌ Wrong Example (Index {sample['index']}):")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Predicted: {sample['predicted_answer']}")
        print(f"Ground Truth: {sample['ground_truth_extracted']}")
        print(f"Full Response: {sample['generated_response'][:200]}...")

print(f"\n{'='*70}")
print("🎉 GRPO Model Inference and Evaluation Completed!")
print(f"{'='*70}")
