#!/bin/bash

# Testing script for GSM8K with trained LoftQ model
# Output will be saved to testing.txt

python /home/thami/learn/LoftQ/scripts/test_gsm8k.py \
  --model_name_or_path /home/thami/learn/LoftQ/deepvision/model_zoo/loftq/Llama-2-7b-hf-4bit-16rank \
  --ckpt_dir /home/thami/learn/LoftQ/deepvision/results/gsm8k_llama2_7b_4bit_16rank_loftq/gsm8k_llama2_7b_4bit_16rank_loftq/Llama-2-7b-hf-4bit-16rank/ep_6/lr_0.0003/seed_11/ \
  --batch_size 16 \
    2>&1 | tee testing.txt
