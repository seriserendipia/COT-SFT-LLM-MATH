# mamba activate torch-env
import torch
import os
import json
import time
import re
import glob
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from answer_utils import extract_answer, compare_answers
import numpy as np
from datetime import datetime

# ============================================================================
# 配置
# ============================================================================
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
SFT_BASE_DIR = "./qwen_peft_sft_lora"          # SFT LoRA 基础目录

# 使用 TRAINING_JOB_ID 加载对应的训练输出（支持并行评估）
training_job_id = os.environ.get("TRAINING_JOB_ID", "default")
GRPO_BASE_DIR = f"./qwen_sft_grpo_lora_{training_job_id}"  # GRPO LoRA 基础目录
OUTPUT_DIR = f"./inference_results_sft_grpo_{training_job_id}"  # 评估输出目录

print(f"\n📁 Evaluating Training Job ID: {training_job_id}")
print(f"📁 Loading GRPO from: {GRPO_BASE_DIR}")
print(f"📁 Saving results to: {OUTPUT_DIR}")

# 1. 数据加载
# 🚀 生产模式 (全量测试集: 1319 条)
ds = load_dataset("ankner/gsm8k-CoT", split="test")
# 🧪 测试模式 (10条样本) - 调试时可用
# ds = load_dataset("ankner/gsm8k-CoT", split="test[:10]")

print(f"Loaded {len(ds)} test samples")

# 2. 模型加载（两层 LoRA：SFT + GRPO）
print("Loading Base Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# ============================================================================
# 方案 B 推理：分层加载 SFT + GRPO
# ============================================================================
# 步骤 1：智能查找并加载 SFT LoRA（最新 checkpoint）
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
    print(f"❌ No SFT checkpoints found in {SFT_BASE_DIR}!")
    exit(1)

print(f"Loading SFT Adapter from: {SFT_ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
print("✅ SFT Adapter loaded.")

# 步骤 2：合并 SFT 到 base
print("Merging SFT adapter into base model...")
model = model.merge_and_unload()
print("✅ SFT weights merged.")

# 步骤 3：智能查找并加载 GRPO LoRA（最新 checkpoint）
print(f"Searching for GRPO checkpoints in: {GRPO_BASE_DIR}")
grpo_checkpoints = glob.glob(os.path.join(GRPO_BASE_DIR, "checkpoint-*"))

if grpo_checkpoints:
    # 按 checkpoint 编号排序，选择最新的
    grpo_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    GRPO_ADAPTER_PATH = grpo_checkpoints[-1]
    print(f"Found {len(grpo_checkpoints)} GRPO checkpoints, using latest: {GRPO_ADAPTER_PATH}")
elif os.path.exists(os.path.join(GRPO_BASE_DIR, "adapter_config.json")):
    # 如果根目录有配置文件，直接使用
    GRPO_ADAPTER_PATH = GRPO_BASE_DIR
    print(f"Using GRPO adapter from root directory: {GRPO_ADAPTER_PATH}")
else:
    print(f"❌ No GRPO checkpoints found in {GRPO_BASE_DIR}!")
    exit(1)

print(f"Loading GRPO Adapter from: {GRPO_ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, GRPO_ADAPTER_PATH)
print("✅ GRPO Adapter loaded.")
print("📊 Final Model Architecture: Base + SFT (merged) + GRPO LoRA")

model.eval()
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tok.pad_token = tok.eos_token

# 3. 推理函数（增强版：记录详细指标）
def generate_answer(question):
    """
    生成答案并记录性能指标
    
    返回:
    - response: 生成的文本
    - metrics: 包含推理时间、token数等指标的字典
    """
    prompt = f"""Solve this math problem step by step.

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

Problem: {question}

Solution:"""
    
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs['input_ids'].shape[1]
    
    # 记录推理时间
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.0, # 贪心解码用于评估
            do_sample=False
        )
    
    inference_time = time.time() - start_time
    
    # 只解码生成部分
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tok.decode(generated_ids, skip_special_tokens=True)
    
    # 计算生成指标
    generation_length = len(generated_ids)
    tokens_per_second = generation_length / inference_time if inference_time > 0 else 0
    
    # 检查格式遵循情况
    has_think_tag = '<think>' in response and '</think>' in response
    has_ans_tag = '<ans>' in response and '</ans>' in response
    
    metrics = {
        "inference_time": inference_time,
        "prompt_length": prompt_length,
        "generation_length": generation_length,
        "total_length": prompt_length + generation_length,
        "tokens_per_second": tokens_per_second,
        "has_think_tag": has_think_tag,
        "has_ans_tag": has_ans_tag,
        "format_correct": has_think_tag and has_ans_tag,
    }
    
    return response, metrics

# 4. 批量评估（增强版：记录详细指标）
results = []
correct = 0
all_metrics = []
print("🚀 Starting Inference...")

eval_start_time = time.time()

for i, example in enumerate(tqdm(ds)):
    q = example["question"]
    gt = example["answer"] # 原始数据集的 answer 字段
    
    # 生成答案并获取指标
    pred_raw, gen_metrics = generate_answer(q)
    pred_ans = extract_answer(pred_raw)
    gt_ans = extract_answer(gt)
    
    is_correct = compare_answers(pred_ans, gt_ans)
    if is_correct: correct += 1
    
    # 合并结果和指标
    result_entry = {
        "sample_id": i,
        "question": q,
        "ground_truth": gt_ans,
        "prediction": pred_ans,
        "full_response": pred_raw,
        "correct": is_correct,
        **gen_metrics  # 包含所有生成指标
    }
    results.append(result_entry)
    all_metrics.append(gen_metrics)
    
    # 每10个样本打印一次进度统计
    if (i + 1) % 10 == 0:
        current_acc = correct / (i + 1)
        avg_time = np.mean([m["inference_time"] for m in all_metrics[-10:]])
        print(f"\n  Progress: {i+1}/{len(ds)} | Acc: {current_acc:.2%} | Avg Time: {avg_time:.2f}s")

total_eval_time = time.time() - eval_start_time

# ============================================================================
# 统计分析
# ============================================================================
acc = correct / len(ds)

# 性能指标统计
inference_times = [m["inference_time"] for m in all_metrics]
generation_lengths = [m["generation_length"] for m in all_metrics]
tokens_per_second = [m["tokens_per_second"] for m in all_metrics]
format_correct = [m["format_correct"] for m in all_metrics]

# 准确率分层统计（按生成长度）
length_bins = [0, 50, 100, 200, 512]
acc_by_length = {}
for i in range(len(length_bins) - 1):
    bin_name = f"{length_bins[i]}-{length_bins[i+1]}"
    samples_in_bin = [r for r in results if length_bins[i] <= r["generation_length"] < length_bins[i+1]]
    if samples_in_bin:
        bin_acc = sum(r["correct"] for r in samples_in_bin) / len(samples_in_bin)
        acc_by_length[bin_name] = {
            "accuracy": bin_acc,
            "count": len(samples_in_bin)
        }

# 错误案例分析
error_cases = [r for r in results if not r["correct"]]
error_by_format = {
    "format_correct": sum(1 for e in error_cases if e["format_correct"]),
    "format_incorrect": sum(1 for e in error_cases if not e["format_correct"]),
}

# 生成评估报告
evaluation_summary = {
    "training_job_id": training_job_id,
    "timestamp": datetime.now().isoformat(),
    "eval_mode": "full (SFT + GRPO)",
    "model_architecture": "Base + SFT (merged) + GRPO LoRA",
    "sft_adapter_path": SFT_ADAPTER_PATH,
    "grpo_adapter_path": GRPO_ADAPTER_PATH,
    "note": "Using latest checkpoints (auto-selected)",
    "dataset": "ankner/gsm8k-CoT test split",
    "num_samples": len(ds),
    "total_eval_time": total_eval_time,
    "avg_time_per_sample": total_eval_time / len(ds),
    
    # 准确率指标
    "accuracy": acc,
    "num_correct": correct,
    "num_wrong": len(ds) - correct,
    
    # 性能指标
    "inference_time": {
        "mean": float(np.mean(inference_times)),
        "std": float(np.std(inference_times)),
        "min": float(np.min(inference_times)),
        "max": float(np.max(inference_times)),
        "median": float(np.median(inference_times)),
    },
    
    "generation_length": {
        "mean": float(np.mean(generation_lengths)),
        "std": float(np.std(generation_lengths)),
        "min": int(np.min(generation_lengths)),
        "max": int(np.max(generation_lengths)),
        "median": float(np.median(generation_lengths)),
    },
    
    "tokens_per_second": {
        "mean": float(np.mean(tokens_per_second)),
        "median": float(np.median(tokens_per_second)),
    },
    
    # 格式遵循情况
    "format_compliance": {
        "format_correct_rate": sum(format_correct) / len(format_correct),
        "format_correct_count": sum(format_correct),
    },
    
    # 准确率分层
    "accuracy_by_length": acc_by_length,
    
    # 错误分析
    "error_analysis": {
        "total_errors": len(error_cases),
        "errors_by_format": error_by_format,
    }
}

# ============================================================================
# 打印评估结果
# ============================================================================
print(f"\n{'='*70}")
print(f"📊 Evaluation Results Summary")
print(f"{'='*70}")
print(f"Overall Accuracy: {acc:.2%} ({correct}/{len(ds)})")
print(f"Total Eval Time: {total_eval_time:.2f}s ({total_eval_time/60:.2f} min)")
print(f"Avg Time/Sample: {evaluation_summary['avg_time_per_sample']:.2f}s")
print(f"\n📈 Generation Metrics:")
print(f"  Avg Length: {evaluation_summary['generation_length']['mean']:.1f} tokens")
print(f"  Avg Speed: {evaluation_summary['tokens_per_second']['mean']:.1f} tok/s")
print(f"  Format Correct: {evaluation_summary['format_compliance']['format_correct_rate']:.2%}")
print(f"\n🔍 Error Analysis:")
print(f"  Total Errors: {len(error_cases)}")
if error_cases:
    print(f"  Errors with correct format: {error_by_format['format_correct']}")
    print(f"  Errors with incorrect format: {error_by_format['format_incorrect']}")
print(f"{'='*70}\n")

# ============================================================================
# 保存所有结果
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 详细结果（每个样本）
output_file = os.path.join(OUTPUT_DIR, "results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"✅ Detailed results saved to: {output_file}")

# 2. 评估摘要（用于快速查看）
summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
with open(summary_file, "w") as f:
    json.dump(evaluation_summary, f, indent=2)
print(f"✅ Evaluation summary saved to: {summary_file}")

# 3. 错误案例（方便分析）
if error_cases:
    error_file = os.path.join(OUTPUT_DIR, "error_cases.json")
    with open(error_file, "w") as f:
        json.dump(error_cases, f, indent=2)
    print(f"✅ Error cases saved to: {error_file}")

# 4. 可视化数据（用于绘图）
viz_data = {
    "sample_ids": list(range(len(results))),
    "correctness": [int(r["correct"]) for r in results],
    "inference_times": inference_times,
    "generation_lengths": generation_lengths,
    "tokens_per_second": tokens_per_second,
    "format_correct": [int(f) for f in format_correct],
}
viz_file = os.path.join(OUTPUT_DIR, "visualization_data.json")
with open(viz_file, "w") as f:
    json.dump(viz_data, f, indent=2)
print(f"✅ Visualization data saved to: {viz_file}")

print(f"\n📁 All files saved to: {OUTPUT_DIR}/")
