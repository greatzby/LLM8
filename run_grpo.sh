#!/bin/bash
set -e

# ============================================================
# GRPO training — 三个 seed 顺序执行
# 与原 Q-learning 命令的唯一区别：
#   1. 脚本名改为 train_grpo_gpt.py
#   2. --gamma / --target_ema 仍可传入但会被忽略
#   3. 新增 --group_size 和 --clip_eps（已有默认值，也可显式指定）
# ============================================================

COMMON_ARGS="
  --train_paths_per_pair 20
  --device cuda:0
  --seed 42
  --max_iters 20000
  --batch_size 32
  --max_rollout_steps 20
  --group_size 4
  --clip_eps 0.2
  --reward_type outcome
  --rollout_temp_start 0.0
  --rollout_temp_end 0.3
  --temp_warmup_iters 6000
  --epsilon_start 0.10
  --epsilon_end 0.01
  --epsilon_warmup_iters 12000
  --kl_coef 0.1
  --kl_warmup_iters 0
  --kl_anneal_iters 20000
  --eval_temperature 1e-3
  --eval_interval 1000
  --save_interval 2000
"

echo "===== [1/3] seed987 ====="
python train_grpo_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed987_100P0 \
  --sft_checkpoint out/composition_20260325_134020/ckpt_5000.pt \
  --log_dir out_nano_grpo_kl010_seed987 \
  ${COMMON_ARGS}

echo "===== [2/3] seed321 ====="
python train_grpo_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed321_100P0 \
  --sft_checkpoint out/composition_20260325_103615/ckpt_5000.pt \
  --log_dir out_nano_grpo_kl010_seed321 \
  ${COMMON_ARGS}

echo "===== [3/3] seed654 ====="
python train_grpo_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed654_100P0 \
  --sft_checkpoint out/composition_20260325_070623/ckpt_5000.pt \
  --log_dir out_nano_grpo_kl010_seed654 \
  ${COMMON_ARGS}

echo "All GRPO runs finished."