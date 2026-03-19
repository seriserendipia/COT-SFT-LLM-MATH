"""
TPU SFT Pipeline: 训练 + 推理 + 评估 + 性能指标（统一主脚本）

流程:
  1. 加载模型 + LoRA
  2. Baseline 推理评估
  3. SFT 训练
  4. Fine-tuned 推理评估
  5. 输出性能指标 + 准确率对比

每次运行自动产出:
  - results_tpu.json: 完整结果（准确率 + 性能 + 逐条推理结果）

Prompt 格式与 GPU 端 (1-qwen-cot-sft.py / 2-qwen-cot-inference.py) 完全对齐。
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
from tunix.sft import hooks as sft_hooks


# ======================================================================
# Training Hooks — 收集 per-step timing
# ======================================================================
class TimingHooks(sft_hooks.TrainingHooks):
    def __init__(self):
        self.step_data = []  # [(step, step_time, loss)]

    def on_train_start(self, train_ctx): pass
    def on_train_end(self, train_ctx): pass
    def on_train_step_start(self, train_ctx): pass
    def on_eval_step_start(self, train_ctx): pass
    def on_eval_step_end(self, train_ctx, eval_loss): pass

    def on_train_step_end(self, train_ctx, train_step, train_loss, step_time):
        self.step_data.append((train_step, step_time, float(train_loss)))
        print(f"     step {train_step}: loss={float(train_loss):.4f}, time={step_time:.3f}s")


# ======================================================================
# Config — 改这里控制实验规模
# ======================================================================
MODEL_NAME = "qwen2.5-1.5b"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
MODEL_DOWNLOAD_PATH = "/home/serendipity/models/qwen2.5-coder-1.5b"

# Mesh
MESH_SHAPE = (2, 2)
MESH_NAMES = ('fsdp', 'tp')

# LoRA
LORA_CONFIG = {
    'module_path': '.*attn.*proj|.*mlp.*(gate|up|down)_proj',
    'rank': 8, 'alpha': 16.0,
}

# 数据
NUM_TRAIN_SAMPLES = 7465  # 全量训练集
NUM_EVAL_SAMPLES = 1316   # 全量测试集

# 训练
MAX_SEQ_LEN = 512         # P95=353, P99=403, 512 覆盖 100%
BATCH_SIZE = 4
TRAIN_STEPS = None        # None = 自动 1 epoch
LR = 5e-5

# 推理
MAX_GEN_TOKENS = 256
MAX_PROMPT_LENGTH = 256   # PAD 统一长度，避免 JIT 重编译
EVAL_BATCH_SIZE = 16      # 推理 batch size，固定以避免 JIT 重编译

# 硬件峰值 (bfloat16 TFLOP/s per chip, i.e. per jax device)
# TPU v2: 每 chip 2 个 MXU × 22.5 = 45; v2-8 = 4 chips × 45 = 180 total
NUM_PARAMS = 1_543_714_304
NUM_LORA_PARAMS = 9_232_384
PEAK_TFLOPS = {
    'tpu_v2': 45, 'tpu_v4': 275, 'tpu_v5e': 197,
    'gpu_t4': 65, 'gpu_a100': 312, 'cpu': 0.5,
}

# 输出
OUTPUT_PATH = '/home/serendipity/project/TPU-project/results_tpu.json'

# ======================================================================
# Prompt（与 GPU 端 1-qwen-cot-sft.py / 2-qwen-cot-inference.py 一致）
# ======================================================================
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


# ======================================================================
# Helpers
# ======================================================================
def detect_device():
    """检测设备类型和峰值算力"""
    devices = jax.devices()
    name = str(devices[0]).lower()
    n = len(devices)
    if 'tpu' in name:
        # 从设备名推断版本
        if 'v4' in name: peak = PEAK_TFLOPS['tpu_v4']
        elif 'v5' in name: peak = PEAK_TFLOPS['tpu_v5e']
        else: peak = PEAK_TFLOPS['tpu_v2']  # v2/v3
        dtype = 'tpu'
    elif 'cuda' in name or 'gpu' in name:
        if 'a100' in name: peak = PEAK_TFLOPS['gpu_a100']
        else: peak = PEAK_TFLOPS['gpu_t4']
        dtype = 'gpu'
    else:
        peak = PEAK_TFLOPS['cpu']
        dtype = 'cpu'
    return dtype, n, peak, str(devices[0])


def format_sft_example(example):
    """SFT 训练数据格式（与 GPU 端完全一致）
    GPU 端处理链:
      1. to_chat(): answer = f"<think>{response}</think>\\n{final_ans}"
      2. prepare_text(): text = f"{prompt} {answer}"
    """
    prompt = PROMPT_TEMPLATE.format(question=example['question'])
    response = example['response'].strip()
    answer = example['answer'].strip()
    completion = f" <think>{response}</think>\n{answer}"
    return prompt, completion


def tokenize_for_training(example, tokenizer):
    """Tokenize 并 pad 到 MAX_SEQ_LEN，返回 (tokens, mask)"""
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


def run_eval(sampler, eval_ds, mesh, label=""):
    """批量推理评估，返回 (accuracy, results, timing_info)"""
    examples = list(eval_ds)
    n = len(examples)
    results = []
    batch_times = []

    all_prompts = [PROMPT_TEMPLATE.format(question=ex['question']) for ex in examples]

    for batch_start in range(0, n, EVAL_BATCH_SIZE):
        batch_end = min(batch_start + EVAL_BATCH_SIZE, n)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_examples = examples[batch_start:batch_end]
        actual_size = len(batch_prompts)

        # Pad 到固定 batch size 避免最后一个 batch 触发 JIT 重编译
        if actual_size < EVAL_BATCH_SIZE:
            batch_prompts = batch_prompts + [batch_prompts[0]] * (EVAL_BATCH_SIZE - actual_size)

        t0 = time.time()
        with mesh:
            output = sampler(
                input_strings=batch_prompts,
                max_generation_steps=MAX_GEN_TOKENS,
                max_prompt_length=MAX_PROMPT_LENGTH,
                temperature=0.0,
            )
        t1 = time.time()
        batch_time = t1 - t0
        batch_times.append(batch_time)

        for j in range(actual_size):
            generated = output.text[j]
            pred_answer = extract_answer(generated)
            gt_answer = extract_answer(batch_examples[j]['answer'])
            correct = bool(compare_answers(pred_answer, gt_answer))
            results.append({
                'question': batch_examples[j]['question'],
                'generated': generated[:300],
                'pred_answer': pred_answer,
                'gt_answer': gt_answer,
                'correct': correct,
            })

        bi = batch_start // EVAL_BATCH_SIZE + 1
        correct_in_batch = sum(r['correct'] for r in results[batch_start:batch_end])
        print(f"  [{label}] Batch {bi}: {actual_size} samples in {batch_time:.1f}s "
              f"({batch_time/actual_size:.2f}s/sample) | {correct_in_batch}/{actual_size} correct")

    acc = sum(r['correct'] for r in results) / len(results)
    total_time = sum(batch_times)
    print(f"  [{label}] Accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"  [{label}] Total: {total_time:.1f}s, avg {total_time/n:.3f}s/sample")

    timing = {
        'batch_times': batch_times,
        'total_sec': total_time,
        'first_batch_sec': batch_times[0],
        'steady_batch_sec': float(np.mean(batch_times[1:])) if len(batch_times) > 1 else batch_times[0],
        'avg_per_sample': total_time / n,
    }
    return acc, results, timing


# ======================================================================
# Main Pipeline
# ======================================================================
def main():
    t_pipeline_start = time.time()
    perf = {}

    # --- Device detection ---
    device_type, num_devices, peak_tflops, device_name = detect_device()
    perf['device_type'] = device_type
    perf['device_name'] = device_name
    perf['num_devices'] = num_devices
    perf['peak_tflops_per_chip'] = peak_tflops

    print(f"Device: {device_name} x{num_devices}, peak {peak_tflops} TFLOP/s/chip")

    # --- 1. Model Load ---
    t0 = time.time()
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
    perf['model_load_sec'] = time.time() - t0
    print(f"1. Model + LoRA loaded ({perf['model_load_sec']:.1f}s)")

    # --- 2. Sampler + dtype fix ---
    cache_config = CacheConfig(cache_size=768, num_layers=28, num_kv_heads=2, head_dim=128)
    Sampler.dtype = property(lambda self: jnp.bfloat16)
    with mesh:
        sampler = Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)

    # --- 3. Baseline eval ---
    eval_ds = load_dataset('ankner/gsm8k-CoT', split=f'test[:{NUM_EVAL_SAMPLES}]')
    print(f"\n2. BASELINE eval ({NUM_EVAL_SAMPLES} samples)...")
    t0 = time.time()
    baseline_acc, baseline_results, baseline_timing = run_eval(sampler, eval_ds, mesh, "BASELINE")
    perf['baseline_eval_sec'] = time.time() - t0
    perf['baseline_inference_first_batch_sec'] = baseline_timing['first_batch_sec']
    perf['baseline_inference_steady_per_sample'] = baseline_timing['steady_batch_sec'] / EVAL_BATCH_SIZE

    # --- 4. Prepare training data ---
    train_ds = load_dataset('ankner/gsm8k-CoT', split=f'train[:{NUM_TRAIN_SAMPLES}]')
    all_tokens, all_masks = [], []
    for ex in train_ds:
        t, m = tokenize_for_training(ex, tokenizer)
        all_tokens.append(t); all_masks.append(m)
    all_tokens = np.stack(all_tokens)
    all_masks = np.stack(all_masks)

    train_batches = []
    for i in range(0, len(all_tokens) // BATCH_SIZE * BATCH_SIZE, BATCH_SIZE):
        train_batches.append(peft_trainer.TrainingInput(
            input_tokens=all_tokens[i:i + BATCH_SIZE],
            input_mask=all_masks[i:i + BATCH_SIZE],
        ))

    # Auto-compute steps for 1 epoch if not set
    train_steps = TRAIN_STEPS
    if train_steps is None:
        train_steps = len(train_batches)
    train_batches_repeated = train_batches * (train_steps // len(train_batches) + 2)

    perf['train_samples'] = NUM_TRAIN_SAMPLES
    perf['train_steps'] = train_steps
    perf['batch_size'] = BATCH_SIZE
    perf['seq_len'] = MAX_SEQ_LEN

    # --- 5. SFT Training ---
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
        peft_trainer.TrainingConfig(
            eval_every_n_steps=999, max_steps=train_steps,
            data_sharding_axis=("fsdp",),
        ),
    )
    trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

    # 注册 timing hooks
    timing_hooks = TimingHooks()
    trainer.with_training_hooks(timing_hooks)

    print(f"\n3. SFT Training ({NUM_TRAIN_SAMPLES} samples, {train_steps} steps)...")
    t_train_start = time.time()
    with mesh:
        trainer.train(train_batches_repeated, None)
    perf['training_wall_sec'] = time.time() - t_train_start

    # Performance metrics
    # steady_step_sec: 用 hooks 后半段（纯稳态，无 JIT/ramp-up 污染）
    # first_step_overhead: 用 wall_clock - steady × steps 兜底（捕获完整编译开销）
    step_times = [t for _, t, _ in timing_hooks.step_data]
    step_losses = [l for _, _, l in timing_hooks.step_data]
    perf['final_train_loss'] = step_losses[-1] if step_losses else None
    perf['step_times'] = step_times

    if len(step_times) >= 4:
        # 后 50% 的步数 = 纯稳态
        half = len(step_times) // 2
        perf['training_steady_step_sec'] = float(np.mean(step_times[half:]))
    elif len(step_times) >= 1:
        perf['training_steady_step_sec'] = step_times[-1]
    else:
        perf['training_steady_step_sec'] = perf['training_wall_sec'] / max(train_steps, 1)

    # 编译开销 = 总时间 - 稳态 × 总步数（差值包含 JIT + ramp-up）
    perf['compilation_overhead_sec'] = perf['training_wall_sec'] - perf['training_steady_step_sec'] * train_steps
    perf['training_first_step_sec'] = perf['compilation_overhead_sec'] + perf['training_steady_step_sec']

    tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
    perf['throughput_tokens_per_sec'] = tokens_per_step / perf['training_steady_step_sec']

    # MFU: LoRA 训练 ≈ 4N/token (forward 2N + backward 2N, 冻结参数不算梯度累加)
    # 全参数训练用 6N (forward 2N + backward 2N + gradient 2N)
    flops_per_step = 4 * NUM_PARAMS * tokens_per_step
    tflops_per_device = flops_per_step / perf['training_steady_step_sec'] / num_devices / 1e12
    perf['training_tflops_per_device'] = round(tflops_per_device, 2)
    perf['training_mfu_percent'] = round(tflops_per_device / peak_tflops * 100, 1)

    print(f"   Done in {perf['training_wall_sec']:.1f}s")
    print(f"   Steady step: {perf['training_steady_step_sec']:.3f}s ({1/perf['training_steady_step_sec']:.1f} steps/sec)")
    print(f"   Compilation overhead: {perf['compilation_overhead_sec']:.1f}s")
    print(f"   Throughput: {perf['throughput_tokens_per_sec']:.0f} tok/s | MFU: {perf['training_mfu_percent']}%")
    print(f"   Final loss: {perf['final_train_loss']:.4f}" if perf['final_train_loss'] else "")

    # --- 6. Fine-tuned eval ---
    with mesh:
        sampler_ft = Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)

    print(f"\n4. FINE-TUNED eval ({NUM_EVAL_SAMPLES} samples)...")
    t0 = time.time()
    finetuned_acc, finetuned_results, finetuned_timing = run_eval(sampler_ft, eval_ds, mesh, "FINETUNED")
    perf['finetuned_eval_sec'] = time.time() - t0
    perf['finetuned_inference_first_batch_sec'] = finetuned_timing['first_batch_sec']
    perf['finetuned_inference_steady_per_sample'] = finetuned_timing['steady_batch_sec'] / EVAL_BATCH_SIZE

    # --- 7. Statistical significance (McNemar's test) ---
    from scipy import stats as sp_stats
    base_correct = [r['correct'] for r in baseline_results]
    ft_correct = [r['correct'] for r in finetuned_results]
    # Discordant pairs
    b_only = sum(1 for b, f in zip(base_correct, ft_correct) if b and not f)
    ft_only = sum(1 for b, f in zip(base_correct, ft_correct) if not b and f)
    both_right = sum(1 for b, f in zip(base_correct, ft_correct) if b and f)
    both_wrong = sum(1 for b, f in zip(base_correct, ft_correct) if not b and not f)
    # McNemar chi-squared with continuity correction
    if b_only + ft_only > 0:
        chi2 = (abs(ft_only - b_only) - 1)**2 / (ft_only + b_only)
        p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)
    else:
        chi2, p_value = 0.0, 1.0
    sig = "YES" if p_value < 0.05 else "NO"
    perf['mcnemar_chi2'] = round(chi2, 3)
    perf['mcnemar_p_value'] = round(p_value, 6)
    perf['significant_at_005'] = bool(p_value < 0.05)
    perf['discordant_base_only'] = b_only
    perf['discordant_ft_only'] = ft_only

    # --- 8. Summary ---
    perf['total_wall_sec'] = time.time() - t_pipeline_start
    perf['baseline_accuracy'] = baseline_acc
    perf['finetuned_accuracy'] = finetuned_acc
    perf['accuracy_improvement'] = finetuned_acc - baseline_acc
    perf['eval_samples'] = NUM_EVAL_SAMPLES

    print(f"\n{'='*60}")
    print(f"  RESULTS ({device_type.upper()} x{num_devices})")
    print(f"{'='*60}")
    print(f"  Baseline accuracy:     {baseline_acc:.1%}")
    print(f"  Fine-tuned accuracy:   {finetuned_acc:.1%}")
    print(f"  Improvement:           {perf['accuracy_improvement']:+.1%}")
    print(f"  McNemar χ²={chi2:.2f}, p={p_value:.4f} → Significant: {sig}")
    print(f"    (baseline-only={b_only}, finetuned-only={ft_only})")
    print(f"{'='*60}")
    print(f"  PERFORMANCE")
    print(f"{'='*60}")
    print(f"  Model load:            {perf['model_load_sec']:.1f}s")
    print(f"  Compilation overhead:  {perf['compilation_overhead_sec']:.1f}s")
    print(f"  Training steady step:  {perf['training_steady_step_sec']:.3f}s ({1/perf['training_steady_step_sec']:.1f} steps/sec)")
    print(f"  Throughput (steady):   {perf['throughput_tokens_per_sec']:.0f} tokens/sec")
    print(f"  TFLOP/s/device:        {perf['training_tflops_per_device']}")
    print(f"  MFU:                   {perf['training_mfu_percent']}%")
    print(f"  Inference first batch: {perf['finetuned_inference_first_batch_sec']:.1f}s (incl. JIT)")
    print(f"  Inference steady:     {perf['finetuned_inference_steady_per_sample']:.3f}s/sample (batch={EVAL_BATCH_SIZE})")
    print(f"  Total wall time:       {perf['total_wall_sec']:.1f}s")
    print(f"{'='*60}")

    # Save
    output = {
        'perf': perf,
        'config': {
            'model_id': MODEL_ID,
            'lora_rank': LORA_CONFIG['rank'],
            'lora_alpha': LORA_CONFIG['alpha'],
            'train_samples': NUM_TRAIN_SAMPLES,
            'eval_samples': NUM_EVAL_SAMPLES,
            'train_steps': train_steps,
            'batch_size': BATCH_SIZE,
            'seq_len': MAX_SEQ_LEN,
            'lr': LR,
        },
        'baseline_results': baseline_results,
        'finetuned_results': finetuned_results,
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
