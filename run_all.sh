#!/usr/bin/env bash
set -euo pipefail

# === 全局可配置参数 ==========================================================
NODES_PER_STAGE=30
TOTAL_STAGES=3                    # 若 create_graph_graphA.py 使用不同 stage 数，请同步修改
TOTAL_NODES=$((NODES_PER_STAGE * TOTAL_STAGES))

GRAPH_SEED=42
GRAPH_NAME="graphA"

TRAIN_PATHS_PER_PAIR=20
EVAL_PATHS_PER_PAIR=1
TRAIN_RATIO=0.5                   # 如需改回 0.85，可直接改这里
DATASET_SEED=42

BLOCK_MULTIPLE=32

TRAIN_N_LAYER=1
TRAIN_N_HEAD=1
TRAIN_N_EMBD=92
TRAIN_BATCH_SIZE=512
TRAIN_MAX_ITERS=50000
TRAIN_CHECKPOINT_INTERVAL=5000
TRAIN_SEED=42
TRAIN_TEST_INTERVAL=1000
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_SCRIPT="${ROOT_DIR}/data/simple_graph/create_graph_graphA.py"
DATASET_SCRIPT="${ROOT_DIR}/generate_alpine_from_graph.py"
PREP_SCRIPT="${ROOT_DIR}/data/simple_graph/prepare_compositionnew.py"
TRAIN_SCRIPT="${ROOT_DIR}/train_composition_fixed_final.py"

GRAPH_OUT_ROOT="${ROOT_DIR}/data/graphs"
DATASET_ROOT="${ROOT_DIR}/data/datasets"

declare -a P_GLOBAL_LIST=(0.10 0.15 0.20 0.25 0.30)

format_pg_tag() {
  python - <<PY
p = float("${1}")
print(f"{int(round(p*100)):03d}")
PY
}

for P_GLOBAL in "${P_GLOBAL_LIST[@]}"; do
  PG_TAG="$(format_pg_tag "${P_GLOBAL}")"

  GRAPH_OUT_DIR="${GRAPH_OUT_ROOT}/${GRAPH_NAME}_pg${PG_TAG}_nps${NODES_PER_STAGE}_seed${GRAPH_SEED}"
  DATASET_DIR="${DATASET_ROOT}/${GRAPH_NAME}_pg${PG_TAG}_tier3"

  echo
  echo "==================== p_global = ${P_GLOBAL} ===================="

  echo "==== [1/4] 构建图结构 (p_global=${P_GLOBAL}) ===="
  python "${GRAPH_SCRIPT}" \
    --nodes_per_stage "${NODES_PER_STAGE}" \
    --p_global "${P_GLOBAL}" \
    --seed "${GRAPH_SEED}" \
    --experiment_name "${GRAPH_NAME}" \
    --output_root "${GRAPH_OUT_ROOT}"

  echo "==== [2/4] 构建 Tier-3 数据集 (p_global=${P_GLOBAL}) ===="
  python "${DATASET_SCRIPT}" \
    --input_graph "${GRAPH_OUT_DIR}/composition_graph.graphml" \
    --stage_info "${GRAPH_OUT_DIR}/stage_info.pkl" \
    --output_dir "${DATASET_DIR}" \
    --train_paths_per_pair "${TRAIN_PATHS_PER_PAIR}" \
    --eval_paths_per_pair "${EVAL_PATHS_PER_PAIR}" \
    --train_ratio "${TRAIN_RATIO}" \
    --seed "${DATASET_SEED}"

  echo "==== [3/4] 准备二进制数据 (p_global=${P_GLOBAL}) ===="
  python "${PREP_SCRIPT}" \
    --data_dir "${DATASET_DIR}" \
    --total_nodes "${TOTAL_NODES}" \
    --train_paths_per_pair "${TRAIN_PATHS_PER_PAIR}" \
    --block_multiple "${BLOCK_MULTIPLE}"

  echo "==== [4/4] 开始训练 (p_global=${P_GLOBAL}) ===="
  python "${TRAIN_SCRIPT}" \
    --data_dir "${DATASET_DIR}" \
    --train_paths_per_pair "${TRAIN_PATHS_PER_PAIR}" \
    --n_layer "${TRAIN_N_LAYER}" \
    --n_head "${TRAIN_N_HEAD}" \
    --n_embd "${TRAIN_N_EMBD}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --max_iters "${TRAIN_MAX_ITERS}" \
    --checkpoint_interval "${TRAIN_CHECKPOINT_INTERVAL}" \
    --seed "${TRAIN_SEED}" \
    --test_interval "${TRAIN_TEST_INTERVAL}"

  echo "==== p_global=${P_GLOBAL} 完成 ===="
done

echo
echo "==== 全部 5 组实验已完成 ===="