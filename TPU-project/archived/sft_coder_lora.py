"""
SFT LoRA training for Qwen2.5-Coder-1.5B on TPU
复用 Tunix 官方 CLI 组件, 仅替换数据源为 HuggingFace datasets
"""
import os
assert os.environ.get('HF_TOKEN'), "Set HF_TOKEN env var first: export HF_TOKEN=hf_xxx"

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from tunix.cli.utils import model as model_lib
from tunix.sft import peft_trainer, utils as sft_utils
from transformers import AutoTokenizer
from datasets import load_dataset

# ============================== Config ==============================
MODEL_NAME = "qwen2.5-1.5b"           # Tunix 注册名 (架构一致)
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"  # HuggingFace 权重
MODEL_DOWNLOAD_PATH = "/home/serendipity/models/qwen2.5-coder-1.5b"
MESH_SHAPE = (2, 2)
MESH_NAMES = ('fsdp', 'tp')

LORA_CONFIG = {
    'module_path': '.*attn.*proj|.*mlp.*(gate|up|down)_proj',
    'rank': 8,
    'alpha': 16.0,
}

MAX_SEQ_LEN = 256
BATCH_SIZE = 4
MAX_STEPS = 3
EVAL_EVERY = 2
LR = 5e-5
NUM_TRAIN_SAMPLES = 10  # 最小量测试

# ============================== 1. Mesh ==============================
mesh = jax.make_mesh(
    MESH_SHAPE, MESH_NAMES,
    axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_NAMES),
)
print(f"Mesh: {mesh}")

# ============================== 2. Model + LoRA ==============================
# 复用官方 model_lib.create_model (内部处理 LoRA + reshard)
model_config = {
    'model_name': MODEL_NAME,
    'model_id': MODEL_ID,
    'model_source': 'huggingface',
    'model_download_path': MODEL_DOWNLOAD_PATH,
    'model_path': '',
    'intermediate_ckpt_dir': '/tmp/intermediate_ckpt/',
    'rng_seed': 0,
    'model_display': False,
    'lora_config': LORA_CONFIG,
    'mesh': {'shape': str(MESH_SHAPE), 'axis_names': str(MESH_NAMES)},
}
tokenizer_config = {
    'tokenizer_path': MODEL_ID,
    'tokenizer_type': 'huggingface',
    'add_bos': True,
    'add_eos': True,
}

model, tokenizer_path = model_lib.create_model(model_config, tokenizer_config, mesh)
print("2. Model + LoRA loaded OK")

# LoRA 参数统计
lora_count = sum(
    v.get_value().size for _, v in nnx.iter_graph(model)
    if isinstance(v, nnx.LoRAParam)
)
print(f"   LoRA params: {lora_count:,}")

# ============================== 3. Tokenizer ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ['HF_TOKEN'])

# ============================== 4. Data ==============================
ds = load_dataset('ankner/gsm8k-CoT', split=f'train[:{NUM_TRAIN_SAMPLES}]')
print(f"3. Data loaded: {len(ds)} samples")


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
for ex in ds:
    t, m = format_and_tokenize(ex)
    all_tokens.append(t)
    all_masks.append(m)
all_tokens = np.stack(all_tokens)
all_masks = np.stack(all_masks)
print(f"   Token shape: {all_tokens.shape}, target tokens: {all_masks.sum()}/{all_masks.size}")

# 构建 batch list (官方模式: list of TrainingInput)
train_batches = []
for i in range(0, len(all_tokens) // BATCH_SIZE * BATCH_SIZE, BATCH_SIZE):
    train_batches.append(peft_trainer.TrainingInput(
        input_tokens=all_tokens[i:i + BATCH_SIZE],
        input_mask=all_masks[i:i + BATCH_SIZE],
    ))
# 重复以凑够 max_steps
train_batches = train_batches * (MAX_STEPS + 1)

# ============================== 5. gen_model_input_fn ==============================
# 完全复用官方 peft_main.py 的 gen_model_input_fn
def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != 0
    positions = sft_utils.build_positions_from_mask(pad_mask)
    attention_mask = sft_utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

# ============================== 6. PeftTrainer ==============================
# Workaround: skip _shard_optimizer to avoid Qwen2 3D pspec vs LoRA 2D rank mismatch
# on JAX 0.9.x. JIT will auto-shard on first step (compiles twice, ~1 min extra).
# See tunix_qwen2_sharding_bug.md for full analysis.
peft_trainer.PeftTrainer._shard_optimizer = lambda self, mesh: None

optimizer = optax.adamw(learning_rate=LR)
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY,
    max_steps=MAX_STEPS,
    data_sharding_axis=("fsdp",),
)
trainer = peft_trainer.PeftTrainer(model, optimizer, training_config)
trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

# ============================== 7. Train ==============================
print(f"4. Training {MAX_STEPS} steps (bs={BATCH_SIZE}, seq={MAX_SEQ_LEN})...")
with mesh:
    trainer.train(train_batches, None)

print("\nDone! Pipeline verified.")
