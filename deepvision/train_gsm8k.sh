#!/bin/bash

# Train 4-bit 16-rank llama-2-7b with LoftQ on GSM8K
# Adjust batch size and gradient accumulation based on your GPU memory
# global batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

MODEL_PATH="/home/thami/learn/LoftQ/deepvision/model_zoo/loftq/Llama-2-7b-hf-4bit-16rank"
OUTPUT_DIR="/home/thami/learn/LoftQ/deepvision/results/gsm8k_llama2_7b_4bit_16rank_loftq"

accelerate launch /home/thami/learn/LoftQ/scripts/train_gsm8k.py \
  --model_name_or_path $MODEL_PATH \
  --learning_rate 3e-4 \
  --seed 11 \
  --expt_name gsm8k_llama2_7b_4bit_16rank_loftq \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --save_strategy epoch \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --do_train \
  # --report_to tensorboard
