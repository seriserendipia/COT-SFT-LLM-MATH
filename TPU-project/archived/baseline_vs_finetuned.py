"""
Baseline vs Fine-tuned 对比（小样本跑通版）
1. Baseline: 原始模型直接推理
2. SFT 训练 10 条
3. Fine-tuned: 训练后模型推理
4. 对比准确率

Prompt 格式与 GPU 端 2-qwen-cot-inference.py 保持一致
"""
import os, sys, json, time, copy
assert os.environ.get('HF_TOKEN'), "Set HF_TOKEN env var first: export HF_TOKEN=hf_xxx"

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from tunix.cli.utils import model as model_lib
from tunix.generate.sampler import Sampler, CacheConfig
from tunix.sft import peft_trainer, utils as sft_utils
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, '/home/serendipity/project')
from answer_utils import extract_answer, compare_answers

# ============================== Config ==============================
MODEL_NAME = "qwen2.5-1.5b"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
MODEL_DOWNLOAD_PATH = "/home/serendipity/models/qwen2.5-coder-1.5b"
MESH_SHAPE = (2, 2)
MESH_NAMES = ('fsdp', 'tp')
LORA_CONFIG = {
    'module_path': '.*attn.*proj|.*mlp.*(gate|up|down)_proj',
    'rank': 8, 'alpha': 16.0,
}

NUM_TRAIN_SAMPLES = 10
NUM_EVAL_SAMPLES = 10
MAX_SEQ_LEN = 256
BATCH_SIZE = 4
TRAIN_STEPS = 3        # 小样本跑通
LR = 5e-5
MAX_GEN_TOKENS = 256
MAX_PROMPT_LENGTH = 256  # PAD 统一长度，避免 JIT 重编译

# ============================== Prompt (与 GPU 端一致) ==============================
PROMPT_TEMPLATE = """Solve this math problem step by step.

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

# SFT 训练数据格式（与 GPU 端 1-qwen-cot-sft.py 完全一致）
# GPU 端处理链:
#   1. to_chat(): answer = f"<think>{response}</think>\n{final_ans}"
#   2. prepare_text(): text = f"{prompt} {answer}"
# 所以最终训练文本: "...Solution: <think>推理过程</think>\n121"
def format_sft_example(example):
    """将数据集转为 SFT 训练格式"""
    question = example['question']
    response = example['response'].strip()  # CoT 推理步骤
    answer = example['answer'].strip()       # 纯数字
    prompt = PROMPT_TEMPLATE.format(question=question)
    # 与 GPU 端一致: "<think>推理</think>\n数字"
    completion = f" <think>{response}</think>\n{answer}"
    return prompt, completion

# ============================== 1. Mesh + Model ==============================
mesh = jax.make_mesh(
    MESH_SHAPE, MESH_NAMES,
    axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_NAMES),
)

model_config = {
    'model_name': MODEL_NAME, 'model_id': MODEL_ID,
    'model_source': 'huggingface', 'model_download_path': MODEL_DOWNLOAD_PATH,
    'model_path': '', 'intermediate_ckpt_dir': '/tmp/intermediate_ckpt/',
    'rng_seed': 0, 'model_display': False, 'lora_config': LORA_CONFIG,
}
tokenizer_config = {
    'tokenizer_path': MODEL_ID, 'tokenizer_type': 'huggingface',
    'add_bos': True, 'add_eos': True,
}
model, _ = model_lib.create_model(model_config, tokenizer_config, mesh)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ['HF_TOKEN'])
print("1. Model + LoRA loaded")

# ============================== 2. Sampler (Baseline) ==============================
cache_config = CacheConfig(cache_size=768, num_layers=28, num_kv_heads=2, head_dim=128)
Sampler.dtype = property(lambda self: jnp.bfloat16)

with mesh:
    sampler = Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)

# ============================== 3. Eval function ==============================
def run_eval(sampler, eval_ds, label=""):
    """在 eval_ds 上跑推理，返回 (accuracy, results)"""
    results = []
    for i, example in enumerate(eval_ds):
        prompt = PROMPT_TEMPLATE.format(question=example['question'])
        with mesh:
            output = sampler(
                input_strings=[prompt],
                max_generation_steps=MAX_GEN_TOKENS,
                max_prompt_length=MAX_PROMPT_LENGTH,
                temperature=0.0,
            )
        generated = output.text[0]
        pred_answer = extract_answer(generated)
        gt_answer = extract_answer(example['answer'])
        correct = compare_answers(pred_answer, gt_answer)
        results.append({
            'question': example['question'],
            'generated': generated[:300],
            'pred_answer': pred_answer,
            'gt_answer': gt_answer,
            'correct': correct,
        })
        marker = '✓' if correct else '✗'
        print(f"  [{label}][{i+1}/{len(eval_ds)}] pred={pred_answer} gt={gt_answer} {marker}")

    acc = sum(r['correct'] for r in results) / len(results)
    print(f"  [{label}] Accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
    return acc, results

# ============================== 4. Baseline eval ==============================
eval_ds = load_dataset('ankner/gsm8k-CoT', split=f'test[:{NUM_EVAL_SAMPLES}]')
print(f"\n2. BASELINE eval ({NUM_EVAL_SAMPLES} samples)...")
t0 = time.time()
baseline_acc, baseline_results = run_eval(sampler, eval_ds, label="BASELINE")
baseline_time = time.time() - t0

# ============================== 5. SFT Training ==============================
print(f"\n3. SFT Training ({NUM_TRAIN_SAMPLES} samples, {TRAIN_STEPS} steps)...")
train_ds = load_dataset('ankner/gsm8k-CoT', split=f'train[:{NUM_TRAIN_SAMPLES}]')

def tokenize_for_training(example):
    prompt, completion = format_sft_example(example)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_ids + completion_ids
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
        mask = mask[:MAX_SEQ_LEN]
    else:
        pad_len = MAX_SEQ_LEN - len(tokens)
        tokens += [tokenizer.pad_token_id or 0] * pad_len
        mask += [0] * pad_len
    return np.array(tokens, dtype=np.int32), np.array(mask, dtype=np.int32)

all_tokens, all_masks = [], []
for ex in train_ds:
    t, m = tokenize_for_training(ex)
    all_tokens.append(t); all_masks.append(m)
all_tokens = np.stack(all_tokens)
all_masks = np.stack(all_masks)

train_batches = []
for i in range(0, len(all_tokens) // BATCH_SIZE * BATCH_SIZE, BATCH_SIZE):
    train_batches.append(peft_trainer.TrainingInput(
        input_tokens=all_tokens[i:i + BATCH_SIZE],
        input_mask=all_masks[i:i + BATCH_SIZE],
    ))
train_batches = train_batches * (TRAIN_STEPS + 5)

def gen_model_input_fn(x):
    pad_mask = x.input_tokens != 0
    positions = sft_utils.build_positions_from_mask(pad_mask)
    attention_mask = sft_utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens, 'input_mask': x.input_mask,
        'positions': positions, 'attention_mask': attention_mask,
    }

peft_trainer.PeftTrainer._shard_optimizer = lambda self, mesh: None
optimizer = optax.adamw(learning_rate=LR)
trainer = peft_trainer.PeftTrainer(
    model, optimizer,
    peft_trainer.TrainingConfig(eval_every_n_steps=999, max_steps=TRAIN_STEPS, data_sharding_axis=("fsdp",)),
)
trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

t_train_start = time.time()
with mesh:
    trainer.train(train_batches, None)
train_time = time.time() - t_train_start
print(f"   Training done in {train_time:.1f}s")

# ============================== 6. Fine-tuned eval ==============================
# 重建 sampler（模型权重已更新）
with mesh:
    sampler_ft = Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)

print(f"\n4. FINE-TUNED eval ({NUM_EVAL_SAMPLES} samples)...")
t0 = time.time()
finetuned_acc, finetuned_results = run_eval(sampler_ft, eval_ds, label="FINETUNED")
finetuned_time = time.time() - t0

# ============================== 7. Summary ==============================
print(f"\n{'='*60}")
print(f"BASELINE vs FINE-TUNED COMPARISON")
print(f"{'='*60}")
print(f"Baseline accuracy:   {baseline_acc:.1%}")
print(f"Fine-tuned accuracy: {finetuned_acc:.1%}")
print(f"Improvement:         {finetuned_acc - baseline_acc:+.1%}")
print(f"Training time:       {train_time:.1f}s ({TRAIN_STEPS} steps)")
print(f"Baseline eval time:  {baseline_time:.1f}s")
print(f"Finetuned eval time: {finetuned_time:.1f}s")
print(f"{'='*60}")

output = {
    'baseline_accuracy': baseline_acc,
    'finetuned_accuracy': finetuned_acc,
    'improvement': finetuned_acc - baseline_acc,
    'train_steps': TRAIN_STEPS,
    'train_samples': NUM_TRAIN_SAMPLES,
    'eval_samples': NUM_EVAL_SAMPLES,
    'train_time_sec': train_time,
    'baseline_results': baseline_results,
    'finetuned_results': finetuned_results,
}
output_path = '/home/serendipity/project/TPU-project/baseline_vs_finetuned.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved to {output_path}")
