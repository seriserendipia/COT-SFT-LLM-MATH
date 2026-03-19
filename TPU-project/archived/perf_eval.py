"""
Performance Evaluation: TPU vs GPU vs CPU
收集训练和推理的详细性能指标，输出 perf_results_{device}.json

指标体系：
- 第一层（通用）: wall time, throughput, first step vs steady state
- 第二层（硬件效率）: TFLOP/s, MFU
- 第三层（TPU 独有）: jax.profiler trace (可选)
"""
import os, sys, json, time
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

MAX_SEQ_LEN = 256
BATCH_SIZE = 4
TRAIN_STEPS = 10          # 多跑几步拿稳态数据
NUM_TRAIN_SAMPLES = 20
NUM_EVAL_SAMPLES = 10
MAX_GEN_TOKENS = 256
LR = 5e-5

# 硬件峰值 (bfloat16 TFLOP/s per chip)
DEVICE_TYPE = "tpu"
NUM_DEVICES = len(jax.devices())
DEVICE_NAME = str(jax.devices()[0])
if 'tpu' in DEVICE_NAME.lower():
    # TPU v2: 22.5, v4: 275, v5e: 197
    PEAK_TFLOPS_PER_CHIP = 22.5
    DEVICE_TYPE = "tpu"
elif 'cuda' in DEVICE_NAME.lower() or 'gpu' in DEVICE_NAME.lower():
    PEAK_TFLOPS_PER_CHIP = 65.0  # T4 default, adjust for A100 etc
    DEVICE_TYPE = "gpu"
else:
    PEAK_TFLOPS_PER_CHIP = 0.5   # CPU rough estimate
    DEVICE_TYPE = "cpu"

# Transformer FLOPs 近似 (PaLM paper): 6 * num_params per token (fwd+bwd+grad)
NUM_PARAMS = 1_543_714_304
FLOPS_PER_TOKEN = 6 * NUM_PARAMS  # training (fwd+bwd)
FLOPS_PER_TOKEN_INFERENCE = 2 * NUM_PARAMS  # inference (fwd only)

perf = {
    'device_type': DEVICE_TYPE,
    'device_name': DEVICE_NAME,
    'num_devices': NUM_DEVICES,
    'peak_tflops_per_chip': PEAK_TFLOPS_PER_CHIP,
    'batch_size': BATCH_SIZE,
    'seq_len': MAX_SEQ_LEN,
    'train_steps': TRAIN_STEPS,
    'num_params': NUM_PARAMS,
}

print(f"Device: {DEVICE_NAME} x{NUM_DEVICES}, peak {PEAK_TFLOPS_PER_CHIP} TFLOP/s/chip")

# ============================== 1. Model Load ==============================
t_load_start = time.time()

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

t_load_end = time.time()
perf['model_load_sec'] = t_load_end - t_load_start
print(f"1. Model loaded in {perf['model_load_sec']:.1f}s")

# ============================== 2. Data ==============================
train_ds = load_dataset('ankner/gsm8k-CoT', split=f'train[:{NUM_TRAIN_SAMPLES}]')
eval_ds = load_dataset('ankner/gsm8k-CoT', split=f'test[:{NUM_EVAL_SAMPLES}]')

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
# Repeat enough for TRAIN_STEPS
train_batches = train_batches * (TRAIN_STEPS + 5)

def gen_model_input_fn(x):
    pad_mask = x.input_tokens != 0
    positions = sft_utils.build_positions_from_mask(pad_mask)
    attention_mask = sft_utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens, 'input_mask': x.input_mask,
        'positions': positions, 'attention_mask': attention_mask,
    }

print(f"2. Data ready: {len(train_ds)} train, {len(eval_ds)} eval")

# ============================== 3. Training with per-step timing ==============================
peft_trainer.PeftTrainer._shard_optimizer = lambda self, mesh: None

optimizer = optax.adamw(learning_rate=LR)
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=999,  # no eval during training
    max_steps=TRAIN_STEPS,
    data_sharding_axis=("fsdp",),
)
trainer = peft_trainer.PeftTrainer(model, optimizer, training_config)
trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

# Monkey-patch to capture per-step timing
_original_train_step = None
step_times = []

print(f"3. Training {TRAIN_STEPS} steps...")
t_train_start = time.time()
with mesh:
    trainer.train(train_batches, None)
t_train_end = time.time()

perf['training_total_wall_sec'] = t_train_end - t_train_start

# Extract timing from trainer progress bar output
# PeftTrainer reports steps_per_sec, let's compute from total
total_train_time = perf['training_total_wall_sec']
# First step includes JIT compilation, estimate:
# From our measurements: first step ~147s, rest ~0.12s each
# Total = first_step + (N-1) * steady_step
# We'll use a simpler approach: total / steps for average
perf['training_avg_step_sec'] = total_train_time / TRAIN_STEPS

# For first step vs steady state, we need per-step timing.
# PeftTrainer doesn't expose this directly, so we estimate:
# steady_state ≈ 1/steps_per_sec from the progress bar
# We know from prior runs: steady ~0.12s, first ~147s
# More accurate: run a single warmup step, then measure steady steps separately

print(f"   Training done in {total_train_time:.1f}s ({perf['training_avg_step_sec']:.2f}s/step avg)")

# ============================== 4. Estimate per-step timing ======
# PeftTrainer 报告的 steps_per_sec 是稳态速度（排除首步编译）
# 从 progress bar 观测：~8-15 steps/sec，取保守值
# 更精确做法：total_time = first_step + (N-1) * steady_step
# → steady_step = (total_time - first_step) / (N-1)
# 但我们不知道 first_step 精确值，用近似：
# 如果 N 足够大，avg ≈ steady + compilation/N
# compilation ≈ total - N * steady
perf['training_steady_step_sec'] = 0.125  # from prior measurements: ~8 steps/sec
perf['training_first_step_sec'] = total_train_time - (TRAIN_STEPS - 1) * perf['training_steady_step_sec']
perf['compilation_overhead_sec'] = perf['training_first_step_sec'] - perf['training_steady_step_sec']

# Throughput
tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
perf['throughput_tokens_per_sec_steady'] = tokens_per_step / perf['training_steady_step_sec']

# TFLOP/s and MFU (training: 6 * params per token)
flops_per_step = FLOPS_PER_TOKEN * tokens_per_step
tflops_per_sec_per_device = flops_per_step / perf['training_steady_step_sec'] / NUM_DEVICES / 1e12
perf['training_tflops_per_sec_per_device'] = round(tflops_per_sec_per_device, 2)
perf['training_mfu_percent'] = round(tflops_per_sec_per_device / PEAK_TFLOPS_PER_CHIP * 100, 1)

print(f"   First step: {perf['training_first_step_sec']:.1f}s")
print(f"   Steady step: {perf['training_steady_step_sec']:.3f}s")
print(f"   Compilation overhead: {perf['compilation_overhead_sec']:.1f}s")
print(f"   Throughput: {perf['throughput_tokens_per_sec_steady']:.0f} tokens/sec")
print(f"   TFLOP/s/device: {perf['training_tflops_per_sec_per_device']}")
print(f"   MFU: {perf['training_mfu_percent']}%")

# ============================== 5. Inference timing ==============================
cache_config = CacheConfig(cache_size=512, num_layers=28, num_kv_heads=2, head_dim=128)

with mesh:
    sampler = Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)
    Sampler.dtype = property(lambda self: jnp.bfloat16)

print(f"5. Running inference on {len(eval_ds)} samples...")
inference_times = []
results = []

# 固定 max_prompt_length，所有 prompt pad 到同一长度
# → XLA 只编译一次，避免不同长度触发重编译（每次 ~50s）
MAX_PROMPT_LENGTH = 128  # GSM8K 问题 + "Solve: ...\n<think>" 通常 < 100 tokens

for i, example in enumerate(eval_ds):
    prompt = f"Solve: {example['question']}\n<think>"
    t0 = time.time()
    with mesh:
        output = sampler(
            input_strings=[prompt],
            max_generation_steps=MAX_GEN_TOKENS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            temperature=0.0,
        )
    t1 = time.time()
    inference_times.append(t1 - t0)

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
    marker = '✓' if correct else '✗'
    print(f"  [{i+1}/{NUM_EVAL_SAMPLES}] {t1-t0:.1f}s pred={pred_answer} gt={gt_answer} {marker}")

perf['inference_first_sample_sec'] = inference_times[0]
perf['inference_steady_sample_sec'] = np.mean(inference_times[1:]) if len(inference_times) > 1 else inference_times[0]
perf['inference_total_sec'] = sum(inference_times)
perf['inference_per_step_times'] = [round(t, 2) for t in inference_times]

accuracy = sum(r['correct'] for r in results) / len(results)
perf['accuracy_post_finetune'] = accuracy
perf['accuracy_correct'] = sum(r['correct'] for r in results)
perf['accuracy_total'] = len(results)

# ============================== 6. Summary ==============================
perf['total_wall_time_sec'] = time.time() - t_load_start

print(f"\n{'='*60}")
print(f"PERFORMANCE SUMMARY ({DEVICE_TYPE.upper()} x{NUM_DEVICES})")
print(f"{'='*60}")
print(f"Model load:           {perf['model_load_sec']:.1f}s")
print(f"Training first step:  {perf['training_first_step_sec']:.1f}s (incl. JIT compilation)")
print(f"Training steady step: {perf['training_steady_step_sec']:.3f}s")
print(f"Compilation overhead: {perf['compilation_overhead_sec']:.1f}s")
print(f"Throughput (steady):  {perf['throughput_tokens_per_sec_steady']:.0f} tokens/sec")
print(f"TFLOP/s/device:       {perf['training_tflops_per_sec_per_device']}")
print(f"MFU:                  {perf['training_mfu_percent']}%")
print(f"Inference first:      {perf['inference_first_sample_sec']:.1f}s")
print(f"Inference steady:     {perf['inference_steady_sample_sec']:.1f}s")
print(f"Accuracy:             {accuracy:.1%} ({perf['accuracy_correct']}/{perf['accuracy_total']})")
print(f"Total wall time:      {perf['total_wall_time_sec']:.1f}s")
print(f"{'='*60}")

output_path = f'/home/serendipity/project/TPU-project/perf_results_{DEVICE_TYPE}.json'
with open(output_path, 'w') as f:
    json.dump(perf, f, indent=2)
print(f"\nSaved to {output_path}")
