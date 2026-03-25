#!/bin/bash
set -e

# ============================================================
# GRPO training (process reward) — 6 组实验
#   3 组 kl_coef=0.10  +  3 组 kl_coef=0
#   三个 seed (987 / 654 / 321) 各跑两次
# ============================================================

# ---------- 数据 / checkpoint 映射 ----------
DATA_DIRS=(
  "data/datasets/graphnano_pg030_nps30_ns3_seed987_100P0"
  "data/datasets/graphnano_pg030_nps30_ns3_seed654_100P0"
  "data/datasets/graphnano_pg030_nps30_ns3_seed321_100P0"
)
SFT_CKPTS=(
  "out/composition_20260321_063305/ckpt_5000.pt"
  "out/composition_20260322_031835/ckpt_5000.pt"
  "out/composition_20260322_101940/ckpt_5000.pt"
)
SEEDS=(987 654 321)

# ---------- 所有 run 共享的参数 ----------
COMMON_ARGS="
  --train_paths_per_pair 20
  --device cuda:0
  --seed 42
  --max_iters 20000
  --batch_size 32
  --max_rollout_steps 20
  --group_size 4
  --clip_eps 0.2
  --reward_type process
  --reward_valid_transition 0.1
  --reward_stage_bridge 0.3
  --reward_stage_bridge_only_once
  --penalty_stage2_detour 0.2
  --penalty_stage3_detour 0.2
  --penalty_repeat_node 0.15
  --step_penalty 0.02
  --rollout_temp_start 0.0
  --rollout_temp_end 0.3
  --temp_warmup_iters 6000
  --epsilon_start 0.10
  --epsilon_end 0.01
  --epsilon_warmup_iters 12000
  --kl_warmup_iters 0
  --kl_anneal_iters 20000
  --eval_temperature 1e-3
  --eval_interval 1000
  --save_interval 2000
"

# ---------- kl_coef 取值 ----------
KL_COEFS=(0.10 0)
KL_TAGS=("kl010" "kl0")

RUN=0
TOTAL=6

for k in 0 1; do
  kl_coef=${KL_COEFS[$k]}
  kl_tag=${KL_TAGS[$k]}

  for i in 0 1 2; do
    RUN=$((RUN + 1))
    seed=${SEEDS[$i]}
    echo "===== [${RUN}/${TOTAL}] ${kl_tag} / seed${seed} ====="

    python train_grpo_gpt.py \
      --data_dir "${DATA_DIRS[$i]}" \
      --sft_checkpoint "${SFT_CKPTS[$i]}" \
      --kl_coef "${kl_coef}" \
      --log_dir "out_nano_grpo_shaping_${kl_tag}_seed${seed}" \
      ${COMMON_ARGS}

    echo "----- [${RUN}/${TOTAL}] done -----"
  done
done

echo "All 6 GRPO runs finished."