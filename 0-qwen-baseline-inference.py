# mamba activate torch-env
# Baseline 评估：直接使用 Qwen2.5-Coder-1.5B 原始模型，不做任何微调

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
from tqdm import tqdm
import os

# 导入统一的答案提取和比较工具
from answer_utils import extract_answer, compare_answers

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

# 2. 加载数据集（测试集：1320 samples）
print("Loading dataset...")
ds = load_dataset("ankner/gsm8k-CoT", split="test")
# 小批量测试（200个样本）：
# ds = load_dataset("ankner/gsm8k-CoT", split="test[:200]")

def to_chat(e):
    """
    将数据集格式转换为推理格式
    输入字段（ankner/gsm8k-CoT）：
    - question: 问题文本（字符串）
    - response: CoT推理步骤（字符串）
    - answer: 最终答案（字符串）
    
    输出格式：
    - question: 问题文本
    - answer: <think>推理步骤</think>\n最终答案
    - ground_truth: 最终答案（用于评估）
    """
    # 直接使用新数据集的字段，无需复杂解析
    think_process = e["response"].strip()
    final_ans = e["answer"].strip()
    
    # 构建完整答案
    full_answer = f"<think>{think_process}</think>\n{final_ans}"
    ground_truth = final_ans  # 用于评估的标准答案
    
    return {
        "question": e['question'], 
        "answer": full_answer,
        "ground_truth": ground_truth
    }

ds = ds.map(to_chat)
print(f"Loaded {len(ds)} test samples")

# 3. 加载原始基础模型（不加载任何 LoRA adapter）
print("\n" + "="*70)
print("🔧 Loading BASELINE model (no fine-tuning)...")
print("   Model: Qwen/Qwen2.5-Coder-1.5B")
print("   Quantization: 4-bit (for memory efficiency)")
print("   Fine-tuning: None (this is the baseline)")
print("="*70 + "\n")

model_id = "Qwen/Qwen2.5-Coder-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

# 4. 加载分词器
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tok.pad_token = tok.eos_token

# 5. 推理函数
def generate_answer(question, max_length=2048):
    """
    生成答案（Baseline 使用 Qwen 官方 chat template）
    
    设计说明：
    - Baseline 应该展示预训练模型的最佳性能
    - 使用官方 chat template 确保模型在最熟悉的格式下工作
    - 不强制特定输出格式，让模型自然发挥
    
    生成参数说明：
    - temperature=0.0: 使用贪心解码（选择概率最高的token）
    - do_sample=False: 关闭随机采样
    - 这样可以保证每次运行结果一致，便于评估和对比
    """
    # 使用 Qwen 官方的 chat template
    # 根据 Qwen2.5-Coder 官方文档，所有 Qwen 模型都支持 apply_chat_template
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves math problems step by step."
        },
        {
            "role": "user",
            "content": f"Solve this problem:\n{question}\n\nPlease think step by step and provide your final numerical answer."
        }
    ]
    
    # 应用 chat template（Qwen2.5-Coder 原生支持）
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,           # ✅ 确定性生成（贪心解码）
            do_sample=False,           # ✅ 关闭采样
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    
    # 只提取新生成的内容（去除 prompt 部分）
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tok.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response

# 6. 批量推理
print("\n" + "="*70)
print("🚀 Starting BASELINE model inference...")
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
    
    # 判断是否正确（使用数值比较）
    is_correct = compare_answers(predicted_answer, gt_answer)
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
    
    # 每50个样本打印一次进度
    if (i + 1) % 50 == 0:
        acc = correct / total
        print(f"\nProgress: {i+1}/{len(ds)}, Accuracy so far: {acc:.4f} ({acc*100:.2f}%)")

# 7. 计算评估指标
accuracy = correct / total
print(f"\n{'='*70}")
print(f"📈 Final Evaluation Results (BASELINE Model):")
print(f"{'='*70}")
print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {total - correct}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*70}\n")

# 8. 保存结果
output_dir = "inference_results_baseline"  # Baseline 模型的推理结果
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
    "model_type": "BASELINE",
    "model_path": model_id,
    "base_model": model_id,
    "fine_tuning": "None",
    "dataset": "ankner/gsm8k-CoT",
    "split": "test",
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

# 9. 保存错误案例分析
wrong_cases = [r for r in results if not r["is_correct"]]
wrong_cases_file = os.path.join(output_dir, "wrong_cases.json")
with open(wrong_cases_file, "w", encoding="utf-8") as f:
    json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
print(f"✅ Wrong cases saved to: {wrong_cases_file}")
print(f"   Total wrong cases: {len(wrong_cases)}")

# 10. 打印一些示例结果
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
print("🎉 BASELINE Model Inference and Evaluation Completed!")
print(f"{'='*70}")
print("\n💡 This baseline result can be compared with:")
print("   - SFT model: inference_results_sft/")
print("   - GRPO model: inference_results_grpo/")
print(f"{'='*70}\n")
