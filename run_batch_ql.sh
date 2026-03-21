#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="batch_logs"
mkdir -p "${LOG_ROOT}"

run_cmd () {
  local name="$1"
  shift
  echo "============================================================"
  echo "[START] ${name}"
  echo "Time: $(date '+%F %T')"
  echo "Command: $*"
  echo "============================================================"

  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"

  echo "============================================================"
  echo "[DONE] ${name}"
  echo "Time: $(date '+%F %T')"
  echo "Log: ${LOG_ROOT}/${name}.log"
  echo "============================================================"
}

# 1) shaping + KL=0.50
run_cmd "exp1_shaping_kl050_seed987" \
python train_qlearning_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed987_100P0 \
  --sft_checkpoint out/composition_20260321_063305/ckpt_5000.pt \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --seed 42 \
  --max_iters 20000 \
  --batch_size 32 \
  --max_rollout_steps 20 \
  --gamma 0.96 \
  --reward_type process \
  --reward_valid_transition 0.1 \
  --reward_stage_bridge 0.3 \
  --reward_stage_bridge_only_once \
  --penalty_stage2_detour 0.2 \
  --penalty_stage3_detour 0.2 \
  --penalty_repeat_node 0.15 \
  --step_penalty 0.02 \
  --rollout_temp_start 0.0 \
  --rollout_temp_end 0.3 \
  --temp_warmup_iters 6000 \
  --epsilon_start 0.10 \
  --epsilon_end 0.01 \
  --epsilon_warmup_iters 12000 \
  --kl_coef 0.50 \
  --kl_warmup_iters 0 \
  --kl_anneal_iters 20000 \
  --eval_temperature 1e-3 \
  --eval_interval 1000 \
  --save_interval 2000 \
  --target_ema 0 \
  --log_dir out_nano_shaping_kl050

# 2) shaping + KL=2.00
run_cmd "exp2_shaping_kl200_seed987" \
python train_qlearning_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed987_100P0 \
  --sft_checkpoint out/composition_20260321_063305/ckpt_5000.pt \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --seed 42 \
  --max_iters 20000 \
  --batch_size 32 \
  --max_rollout_steps 20 \
  --gamma 0.96 \
  --reward_type process \
  --reward_valid_transition 0.1 \
  --reward_stage_bridge 0.3 \
  --reward_stage_bridge_only_once \
  --penalty_stage2_detour 0.2 \
  --penalty_stage3_detour 0.2 \
  --penalty_repeat_node 0.15 \
  --step_penalty 0.02 \
  --rollout_temp_start 0.0 \
  --rollout_temp_end 0.3 \
  --temp_warmup_iters 6000 \
  --epsilon_start 0.10 \
  --epsilon_end 0.01 \
  --epsilon_warmup_iters 12000 \
  --kl_coef 2.00 \
  --kl_warmup_iters 0 \
  --kl_anneal_iters 20000 \
  --eval_temperature 1e-3 \
  --eval_interval 1000 \
  --save_interval 2000 \
  --target_ema 0 \
  --log_dir out_nano_shaping_kl200

# 3) seed654 shaping + KL=0.05
run_cmd "exp3_shaping_kl005_seed654" \
python train_qlearning_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed654_100P0 \
  --sft_checkpoint out/composition_20260322_031835/ckpt_5000.pt \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --seed 42 \
  --max_iters 20000 \
  --batch_size 32 \
  --max_rollout_steps 20 \
  --gamma 0.96 \
  --reward_type process \
  --reward_valid_transition 0.1 \
  --reward_stage_bridge 0.3 \
  --reward_stage_bridge_only_once \
  --penalty_stage2_detour 0.2 \
  --penalty_stage3_detour 0.2 \
  --penalty_repeat_node 0.15 \
  --step_penalty 0.02 \
  --rollout_temp_start 0.0 \
  --rollout_temp_end 0.3 \
  --temp_warmup_iters 6000 \
  --epsilon_start 0.10 \
  --epsilon_end 0.01 \
  --epsilon_warmup_iters 12000 \
  --kl_coef 0.05 \
  --kl_warmup_iters 0 \
  --kl_anneal_iters 20000 \
  --eval_temperature 1e-3 \
  --eval_interval 1000 \
  --save_interval 2000 \
  --target_ema 0 \
  --log_dir out_nano_shaping+kl_seed654

# 4) seed654 terminal-only + KL=0.05
run_cmd "exp4_terminal_kl005_seed654" \
python train_qlearning_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed654_100P0 \
  --sft_checkpoint out/composition_20260322_031835/ckpt_5000.pt \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --seed 42 \
  --max_iters 20000 \
  --batch_size 32 \
  --max_rollout_steps 20 \
  --gamma 0.96 \
  --reward_type outcome \
  --rollout_temp_start 0.0 \
  --rollout_temp_end 0.3 \
  --temp_warmup_iters 6000 \
  --epsilon_start 0.10 \
  --epsilon_end 0.01 \
  --epsilon_warmup_iters 12000 \
  --kl_coef 0.05 \
  --kl_warmup_iters 0 \
  --kl_anneal_iters 20000 \
  --eval_temperature 1e-3 \
  --eval_interval 1000 \
  --save_interval 2000 \
  --target_ema 0 \
  --log_dir out_nano_terminal_kl005_seed654

# 5) no-KL
# 注意：这里严格按你提供的命令保留。
# data_dir 是 seed987，但 log_dir 名字是 seed654，请运行前确认是否符合你的本意。
run_cmd "exp5_shaping_no_kl_mixed_seedname" \
python train_qlearning_gpt.py \
  --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed654_100P0 \
  --sft_checkpoint out/composition_20260322_031835/ckpt_5000.pt \
  --train_paths_per_pair 20 \
  --device cuda:0 \
  --seed 42 \
  --max_iters 20000 \
  --batch_size 32 \
  --max_rollout_steps 20 \
  --gamma 0.96 \
  --reward_type process \
  --reward_valid_transition 0.1 \
  --reward_stage_bridge 0.3 \
  --reward_stage_bridge_only_once \
  --penalty_stage2_detour 0.2 \
  --penalty_stage3_detour 0.2 \
  --penalty_repeat_node 0.15 \
  --step_penalty 0.02 \
  --rollout_temp_start 0.0 \
  --rollout_temp_end 0.3 \
  --temp_warmup_iters 6000 \
  --epsilon_start 0.10 \
  --epsilon_end 0.01 \
  --epsilon_warmup_iters 12000 \
  --kl_coef 0.0 \
  --kl_warmup_iters 0 \
  --kl_anneal_iters 0 \
  --eval_temperature 1e-3 \
  --eval_interval 1000 \
  --save_interval 2000 \
  --target_ema 0 \
  --log_dir out_nano_shaping_no_kl_seed654

echo "All experiments finished."