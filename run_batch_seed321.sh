#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="batch_logs_seed321_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"
mkdir -p out_nano_321

run_cmd () {
  local name="$1"
  shift

  echo "============================================================"
  echo "[START] ${name}"
  echo "Time: $(date '+%F %T')"
  echo "Log : ${LOG_ROOT}/${name}.log"
  echo "============================================================"

  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"

  echo "============================================================"
  echo "[DONE] ${name}"
  echo "Time: $(date '+%F %T')"
  echo "============================================================"
  echo
}

echo "Logs will be saved to: ${LOG_ROOT}"
echo

# ============================================================
# seed321 组
# 注意：这里仍然严格按你之前贴出来的命令保留，
# 没有额外补充 --train_paths_per_pair / --device / --seed / --reward_type
# ============================================================

for item in \
  "0.01 001" \
  "0.1 010" \
  "1.0 100" \
  "10.0 1000"
do
  read -r kl tag <<< "${item}"

  run_cmd "seed321_shaping_kl${tag}" \
    python train_qlearning_gpt.py \
      --data_dir data/datasets/graphnano_pg030_nps30_ns3_seed321_100P0 \
      --sft_checkpoint out/composition_20260322_101940/ckpt_5000.pt \
      --max_iters 20000 \
      --batch_size 32 \
      --max_rollout_steps 20 \
      --gamma 0.96 \
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
      --kl_coef "${kl}" \
      --kl_warmup_iters 0 \
      --kl_anneal_iters 20000 \
      --eval_temperature 1e-3 \
      --eval_interval 1000 \
      --save_interval 2000 \
      --target_ema 0 \
      --log_dir "out_nano_321/out_nano_shaping+kl${tag}_seed321"
done

echo "All seed321 experiments finished."
echo "Logs are under: ${LOG_ROOT}"