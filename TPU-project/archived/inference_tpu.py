"""
TPU Inference: Tunix Sampler for Qwen2.5-Coder-1.5B (with LoRA)
小样本验证: 训练后模型 + baseline 模型对比
"""
import os, sys
assert os.environ.get('HF_TOKEN'), "Set HF_TOKEN env var first: export HF_TOKEN=hf_xxx"

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.cli.utils import model as model_lib
from tunix.generate.sampler import Sampler, CacheConfig
from tunix.sft import peft_trainer
from transformers import AutoTokenizer
from datasets import load_dataset
import json, time

# 复用 GPU 端的答案工具
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
    'rank': 8,
    'alpha': 16.0,
}

NUM_EVAL_SAMPLES = 10
MAX_GEN_TOKENS = 256

# ============================== 1. Mesh + Model ==============================
mesh = jax.make_mesh(
    MESH_SHAPE, MESH_NAMES,
    axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_NAMES),
)

model_config = {
    'model_name': MODEL_NAME, 'model_id': MODEL_ID,
    'model_source': 'huggingface',
    'model_download_path': MODEL_DOWNLOAD_PATH,
    'model_path': '', 'intermediate_ckpt_dir': '/tmp/intermediate_ckpt/',
    'rng_seed': 0, 'model_display': False,
    'lora_config': LORA_CONFIG,
}
tokenizer_config = {
    'tokenizer_path': MODEL_ID, 'tokenizer_type': 'huggingface',
    'add_bos': True, 'add_eos': True,
}

model, _ = model_lib.create_model(model_config, tokenizer_config, mesh)
print("1. Model + LoRA loaded")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ['HF_TOKEN'])

# ============================== 2. Quick SFT (same as sft_coder_lora.py) ======
import optax
import numpy as np
from tunix.sft import utils as sft_utils

peft_trainer.PeftTrainer._shard_optimizer = lambda self, mesh: None

MAX_SEQ_LEN = 256
BATCH_SIZE = 4
train_ds = load_dataset('ankner/gsm8k-CoT', split='train[:10]')

def format_and_tokenize(example):
    question = example['question']
    cot = example.get('chain_of_thought', example.get('cot', ''))
    answer = example.get('answer', '')
    prompt = f"Solve: {question}\n"
    response = f"<think>{cot}</think><ans>{answer}</ans>"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    tokens = prompt_ids + response_ids
    mask = [0] * len(prompt_ids) + [1] * len(response_ids)
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
    t, m = format_and_tokenize(ex)
    all_tokens.append(t); all_masks.append(m)
all_tokens = np.stack(all_tokens)
all_masks = np.stack(all_masks)

train_batches = []
for i in range(0, len(all_tokens) // BATCH_SIZE * BATCH_SIZE, BATCH_SIZE):
    train_batches.append(peft_trainer.TrainingInput(
        input_tokens=all_tokens[i:i + BATCH_SIZE],
        input_mask=all_masks[i:i + BATCH_SIZE],
    ))

def gen_model_input_fn(x):
    pad_mask = x.input_tokens != 0
    positions = sft_utils.build_positions_from_mask(pad_mask)
    attention_mask = sft_utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens, 'input_mask': x.input_mask,
        'positions': positions, 'attention_mask': attention_mask,
    }

optimizer = optax.adamw(learning_rate=5e-5)
trainer = peft_trainer.PeftTrainer(
    model, optimizer,
    peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=3, data_sharding_axis=("fsdp",)),
)
trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

print("2. Training 3 steps...")
with mesh:
    trainer.train(train_batches * 3, None)
print("   Training done")

# ============================== 3. Sampler setup ==============================
# CacheConfig needs model dimensions
cache_config = CacheConfig(
    cache_size=512,       # max prompt + generation
    num_layers=28,
    num_kv_heads=2,       # GQA: 2 KV heads
    head_dim=128,
)

print("3. Setting up Sampler...")
with mesh:
    sampler = Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=cache_config,
    )
    # Fix: LoRA params are float32 but model runs in bfloat16.
    # Sampler.dtype picks the first param's dtype (may be LoRA float32).
    # Override to bfloat16 so KV cache matches model output dtype.
    Sampler.dtype = property(lambda self: jnp.bfloat16)

# ============================== 4. Inference ==============================
eval_ds = load_dataset('ankner/gsm8k-CoT', split=f'test[:{NUM_EVAL_SAMPLES}]')
print(f"4. Running inference on {len(eval_ds)} samples...")

results = []
t0 = time.time()
for i, example in enumerate(eval_ds):
    prompt = f"Solve: {example['question']}\n<think>"

    with mesh:
        output = sampler(
            input_strings=[prompt],
            max_generation_steps=MAX_GEN_TOKENS,
            temperature=0.0,
        )

    generated = output.text[0]
    pred_answer = extract_answer(generated)
    gt_answer = extract_answer(example['answer'])
    correct = compare_answers(pred_answer, gt_answer)

    results.append({
        'question': example['question'],
        'generated': generated[:200],
        'pred_answer': pred_answer,
        'gt_answer': gt_answer,
        'correct': correct,
    })
    print(f"  [{i+1}/{NUM_EVAL_SAMPLES}] pred={pred_answer} gt={gt_answer} {'✓' if correct else '✗'}")

elapsed = time.time() - t0

# ============================== 5. Summary ==============================
accuracy = sum(r['correct'] for r in results) / len(results)
print(f"\n=== Results ===")
print(f"Accuracy: {accuracy:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
print(f"Inference time: {elapsed:.1f}s ({elapsed/len(results):.1f}s/sample)")

output_path = '/home/serendipity/project/TPU-project/eval_results_tpu.json'
with open(output_path, 'w') as f:
    json.dump({
        'accuracy': accuracy,
        'num_samples': len(results),
        'inference_time_sec': elapsed,
        'results': results,
    }, f, indent=2)
print(f"Saved to {output_path}")
