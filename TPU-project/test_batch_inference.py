"""Quick benchmark: batch=16 inference speed on 200 samples (no training)"""
import os, sys, time
assert os.environ.get('HF_TOKEN'), "Set HF_TOKEN env var first: export HF_TOKEN=hf_xxx"

import jax
import jax.numpy as jnp
import numpy as np
from tunix.cli.utils import model as model_lib
from tunix.generate.sampler import Sampler, CacheConfig
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, '/home/serendipity/project')
from answer_utils import extract_answer, compare_answers

# Config (same as pipeline)
MODEL_NAME = "qwen2.5-1.5b"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
MODEL_DOWNLOAD_PATH = "/home/serendipity/models/qwen2.5-coder-1.5b"
MESH_SHAPE = (2, 2)
MESH_NAMES = ('fsdp', 'tp')
LORA_CONFIG = {
    'module_path': '.*attn.*proj|.*mlp.*(gate|up|down)_proj',
    'rank': 8, 'alpha': 16.0,
}
MAX_GEN_TOKENS = 256
MAX_PROMPT_LENGTH = 256
EVAL_BATCH_SIZE = 16
NUM_TEST = 200

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


def main():
    # --- Model load ---
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
    print(f"Model loaded in {time.time()-t0:.1f}s")

    cache_config = CacheConfig(cache_size=768, num_layers=28, num_kv_heads=2, head_dim=128)
    Sampler.dtype = property(lambda self: jnp.bfloat16)
    with mesh:
        sampler = Sampler(transformer=model, tokenizer=tokenizer, cache_config=cache_config)

    # --- Load data ---
    eval_ds = load_dataset('ankner/gsm8k-CoT', split=f'test[:{NUM_TEST}]')
    examples = list(eval_ds)
    n = len(examples)
    all_prompts = [PROMPT_TEMPLATE.format(question=ex['question']) for ex in examples]

    # --- Batched inference ---
    print(f"\n{'='*60}")
    print(f"  BATCHED INFERENCE TEST: batch={EVAL_BATCH_SIZE}, {n} samples")
    print(f"{'='*60}")

    results = []
    batch_times = []

    for batch_start in range(0, n, EVAL_BATCH_SIZE):
        batch_end = min(batch_start + EVAL_BATCH_SIZE, n)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_examples = examples[batch_start:batch_end]
        actual_size = len(batch_prompts)

        # Pad to fixed batch size
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
        bt = t1 - t0
        batch_times.append(bt)

        correct_count = 0
        for j in range(actual_size):
            generated = output.text[j]
            pred = extract_answer(generated)
            gt = extract_answer(batch_examples[j]['answer'])
            c = bool(compare_answers(pred, gt))
            correct_count += c
            results.append(c)

        bi = batch_start // EVAL_BATCH_SIZE + 1
        print(f"  Batch {bi:3d}: {actual_size:2d} samples in {bt:6.1f}s "
              f"({bt/actual_size:.2f}s/sample) | {correct_count}/{actual_size} correct")

    # --- Summary ---
    total = sum(batch_times)
    first = batch_times[0]
    steady_batches = batch_times[1:] if len(batch_times) > 1 else batch_times
    steady_avg = float(np.mean(steady_batches))
    steady_per_sample = steady_avg / EVAL_BATCH_SIZE
    acc = sum(results) / len(results)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:             {acc:.1%} ({sum(results)}/{len(results)})")
    print(f"  Total time:           {total:.1f}s")
    print(f"  First batch (JIT):    {first:.1f}s ({EVAL_BATCH_SIZE} samples)")
    print(f"  Steady batch avg:     {steady_avg:.2f}s ({EVAL_BATCH_SIZE} samples)")
    print(f"  Steady per sample:    {steady_per_sample:.3f}s")
    print(f"  Overall per sample:   {total/n:.3f}s")
    print(f"{'='*60}")
    print(f"  COMPARISON vs batch=1")
    print(f"{'='*60}")
    old_per_sample = 0.7
    print(f"  batch=1 (old):        {old_per_sample:.3f}s/sample")
    print(f"  batch={EVAL_BATCH_SIZE} (new):       {steady_per_sample:.3f}s/sample")
    print(f"  Speedup (steady):     {old_per_sample/steady_per_sample:.1f}x")
    print(f"{'='*60}")
    print(f"  PROJECTION: full 1316 samples × 2 eval rounds")
    print(f"{'='*60}")
    n_batches_full = (1316 + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE  # 83 batches
    projected_one_round = first + (n_batches_full - 1) * steady_avg
    projected_two_rounds = first * 2 + (n_batches_full - 1) * 2 * steady_avg
    old_two_rounds = 1316 * 2 * old_per_sample
    print(f"  Per eval round:       {projected_one_round:.0f}s ({projected_one_round/60:.1f} min)")
    print(f"  Two rounds total:     {projected_two_rounds:.0f}s ({projected_two_rounds/60:.1f} min)")
    print(f"  vs old (batch=1):     {old_two_rounds:.0f}s ({old_two_rounds/60:.1f} min)")
    print(f"  Time saved:           {old_two_rounds - projected_two_rounds:.0f}s ({(old_two_rounds - projected_two_rounds)/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
