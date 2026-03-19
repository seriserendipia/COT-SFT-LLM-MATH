#!/bin/bash
# SFT LoRA training for Qwen2.5-Coder-1.5B on TPU v2-8
# 基于官方 Tunix CLI (tunix.cli.peft_main)
# 最小量测试: 3 步, seq_len=256

set -x

# export HF_TOKEN=hf_xxx  # Set your HF token before running

source ~/tpu-env-311/bin/activate

python3 -m tunix.cli.peft_main \
  base_config.yaml \
  model_config.model_name="qwen2.5-1.5b" \
  model_config.model_id="Qwen/Qwen2.5-Coder-1.5B" \
  model_config.model_source="huggingface" \
  model_config.model_download_path="/home/serendipity/models/qwen2.5-coder-1.5b" \
  model_config.mesh.shape="(2,2)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.rng_seed=0 \
  model_config.lora_config.module_path=".*attn.*proj|.*mlp.*(gate|up|down)_proj" \
  model_config.lora_config.rank=8 \
  model_config.lora_config.alpha=16.0 \
  tokenizer_config.tokenizer_path="Qwen/Qwen2.5-Coder-1.5B" \
  tokenizer_config.tokenizer_type="huggingface" \
  dataset_name="mtnt/en-fr" \
  optimizer_config.opt_type="adamw" \
  optimizer_config.learning_rate=5e-5 \
  max_target_length=256 \
  training_config.eval_every_n_steps=2 \
  training_config.max_steps=3 \
  training_config.data_sharding_axis='["fsdp"]' \
  training_config.metrics_logging_options.log_dir="/tmp/tensorboard/sft_coder" \
  training_config.metrics_logging_options.flush_every_n_steps=1
