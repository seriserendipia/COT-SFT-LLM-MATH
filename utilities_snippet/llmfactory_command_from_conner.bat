# 切到你的项目目录
cd /home1/jwu56645/LLaMA-Factory
# 确保日志目录存在
mkdir -p logs
# 真正的训练命令（已加验证集 10%）
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset my_train \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir saves/Qwen2.5-Coder-1.5B/lora/train_wjc1\
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --val_size 0.1




