"""
GPU SFT Pipeline: 训练 + 推理 + 评估 + 性能指标（GPU/Colab 版）

与 tpu_sft_pipeline.py 的差异（仅这些，其余完全一致）:
  - 框架仍然是 Tunix/JAX（控制变量，纯硬件对比）
  - Mesh shape 动态适应设备数（单 GPU = (1,1)）
  - 模型缓存路径改为 /tmp（Colab 友好）
  - 峰值算力按 GPU 型号自动选取
  - 输出文件: results_gpu.json

Colab 运行前准备（在单独 cell 执行）:
  !pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -q
  !pip install "google-tunix[prod]" datasets transformers scipy -q
  !git clone https://github.com/YOUR_REPO/project.git /content/project 2>/dev/null || true
  import sys; sys.path.insert(0, '/content/project')
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

# answer_utils: 从项目根目录导入；Colab 中需先 git clone 并加入 sys.path
# （见文件头注释）
_ANSWER_UTILS_INLINE = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from answer_utils import extract_answer, compare_answers
except ImportError:
    # Colab fallback: inline 最小实现
    _ANSWER_UTILS_INLINE = True
    import re
    def extract_answer(text):
        m = re.findall(r'<ans>(.+?)</ans>', str(text), re.IGNORECASE | re.DOTALL)
        if m:
            text = m[-1].strip()
        nums = re.findall(r'-?[\d,]+\.?\d*', str(text))
        if not nums: return str(text).strip()
        cleaned = nums[-1].replace(',', '')
        try:
            v = float(cleaned)
            return str(int(v)) if v.is_integer() else str(v)
        except ValueError:
            return cleaned
    def compare_answers(pred, gt, tol=1e-6):
        try:
            return abs(float(extract_answer(str(pred)).replace(',','')) -
                       float(extract_answer(str(gt)).replace(',',''))) < tol
        except Exception:
            return extract_answer(str(pred)).lower() == extract_answer(str(gt)).lower()

from tunix.sft import hooks as sft_hooks


# ======================================================================
# Training Hooks — 收集 per-step timing（与 TPU 版完全一致）
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
# Config
# ======================================================================
MODEL_NAME = "qwen2.5-1.5b"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
# GPU/Colab: 用 /tmp 避免权限问题；已有缓存时直接复用
MODEL_DOWNLOAD_PATH = "/tmp/models/qwen2.5-coder-1.5b"

# LoRA（与 TPU 版一致，方便对比）
LORA_CONFIG = {
    'module_path': '.*attn.*proj|.*mlp.*(gate|up|down)_proj',
    'rank': 8, 'alpha': 16.0,
}

# 数据（与 TPU 版一致）
NUM_TRAIN_SAMPLES = 7465
NUM_EVAL_SAMPLES = 1316

# 训练
MAX_SEQ_LEN = 512
BATCH_SIZE = 4
TRAIN_STEPS = None  # None = 自动 1 epoch
LR = 5e-5

# 推理
MAX_GEN_TOKENS = 256
MAX_PROMPT_LENGTH = 256
EVAL_BATCH_SIZE = 16

# 峰值算力（GPU）
NUM_PARAMS = 1_543_714_304
PEAK_TFLOPS = {
    'tpu_v2': 45, 'tpu_v4': 275, 'tpu_v5e': 197,
    'gpu_t4': 65, 'gpu_a100': 312, 'gpu_v100': 112,
    'gpu_l4': 242, 'cpu': 0.5,
}

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_gpu.json')

# ======================================================================
# Prompt（与 TPU 版 / HPC 版完全一致）
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
    """检测 GPU 类型和峰值算力"""
    devices = jax.devices()
    name = str(devices[0]).lower()
    n = len(devices)

    if 'tpu' in name:
        # 兜底：如果意外跑在 TPU 上也能工作
        if 'v4' in name: peak = PEAK_TFLOPS['tpu_v4']
        elif 'v5' in name: peak = PEAK_TFLOPS['tpu_v5e']
        else: peak = PEAK_TFLOPS['tpu_v2']
        dtype = 'tpu'
    elif 'cuda' in name or 'gpu' in name:
        if 'a100' in name: peak = PEAK_TFLOPS['gpu_a100']
        elif 'l4' in name: peak = PEAK_TFLOPS['gpu_l4']
        elif 'v100' in name: peak = PEAK_TFLOPS['gpu_v100']
        else: peak = PEAK_TFLOPS['gpu_t4']  # Colab 默认 T4
        dtype = 'gpu'
    else:
        peak = PEAK_TFLOPS['cpu']
        dtype = 'cpu'
    return dtype, n, peak, str(devices[0])


def make_mesh(num_devices):
    """
    根据设备数选择合适的 mesh shape。
    TPU v2-8 = 4 chips → (2,2)
    单 GPU/单 TPU    → (1,1)
    双 GPU           → (2,1)
    """
    if num_devices >= 4:
        shape = (2, 2)
    elif num_devices == 2:
        shape = (2, 1)
    else:
        shape = (1, 1)
    return jax.make_mesh(
        shape, ('fsdp', 'tp'),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )


def format_sft_example(example):
    """与 TPU 版完全一致"""
    prompt = PROMPT_TEMPLATE.format(question=example['question'])
    response = example['response'].strip()
    answer = example['answer'].strip()
    completion = f" <think>{response}</think>\n{answer}"
    return prompt, completion


def tokenize_for_training(example, tokenizer):
    """与 TPU 版完全一致"""
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
    """与 TPU 版完全一致"""
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
    if _ANSWER_UTILS_INLINE:
        print("  [注意] answer_utils 未找到，使用 inline 版本")

    # --- 1. Model Load ---
    t0 = time.time()
    mesh = make_mesh(num_devices)
    model_config = {
        'model_name': MODEL_NAME, 'model_id': MODEL_ID,
        'model_source': 'huggingface', 'model_download_path': MODEL_DOWNLOAD_PATH,
        'model_path': '', 'intermediate_ckpt_dir': '/tmp/intermediate_ckpt_gpu/',
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

    # --- 2. Sampler + dtype fix（GPU 上 LoRA float32 vs cache bfloat16 同样需要）---
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

    # Qwen2 sharding bug workaround - 同样适用于 GPU（Tunix 的 bug，非 TPU 独有）
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

    timing_hooks = TimingHooks()
    trainer.with_training_hooks(timing_hooks)

    print(f"\n3. SFT Training ({NUM_TRAIN_SAMPLES} samples, {train_steps} steps)...")
    t_train_start = time.time()
    with mesh:
        trainer.train(train_batches_repeated, None)
    perf['training_wall_sec'] = time.time() - t_train_start

    step_times = [t for _, t, _ in timing_hooks.step_data]
    step_losses = [l for _, _, l in timing_hooks.step_data]
    perf['final_train_loss'] = step_losses[-1] if step_losses else None
    perf['step_times'] = step_times

    if len(step_times) >= 4:
        half = len(step_times) // 2
        perf['training_steady_step_sec'] = float(np.mean(step_times[half:]))
    elif len(step_times) >= 1:
        perf['training_steady_step_sec'] = step_times[-1]
    else:
        perf['training_steady_step_sec'] = perf['training_wall_sec'] / max(train_steps, 1)

    perf['compilation_overhead_sec'] = perf['training_wall_sec'] - perf['training_steady_step_sec'] * train_steps
    perf['training_first_step_sec'] = perf['compilation_overhead_sec'] + perf['training_steady_step_sec']

    tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
    perf['throughput_tokens_per_sec'] = tokens_per_step / perf['training_steady_step_sec']

    # MFU: LoRA 训练 4N/token（同 TPU 版）
    flops_per_step = 4 * NUM_PARAMS * tokens_per_step
    tflops_per_device = flops_per_step / perf['training_steady_step_sec'] / num_devices / 1e12
    perf['training_tflops_per_device'] = round(tflops_per_device, 2)
    perf['training_mfu_percent'] = round(tflops_per_device / peak_tflops * 100, 1) if peak_tflops > 0 else None

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

    # --- 7. McNemar test ---
    from scipy import stats as sp_stats
    base_correct = [r['correct'] for r in baseline_results]
    ft_correct = [r['correct'] for r in finetuned_results]
    b_only = sum(1 for b, f in zip(base_correct, ft_correct) if b and not f)
    ft_only = sum(1 for b, f in zip(base_correct, ft_correct) if not b and f)
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
    print(f"  Inference steady:      {perf['finetuned_inference_steady_per_sample']:.3f}s/sample (batch={EVAL_BATCH_SIZE})")
    print(f"  Total wall time:       {perf['total_wall_sec']:.1f}s")
    print(f"{'='*60}")

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
