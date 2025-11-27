#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1→S3 组合任务分析脚本（阶段拆解版 v3）

事件链条定义（更新）：
- Event A：合法起步（第一步属于合法出边）。
- Event B：路径中命中“可桥接 Stage2”（bridge_candidates 集合），允许多跳后再命中。
- Event C：命中可桥接 Stage2 后，是否出现 Stage2→Stage3 的转移（意味着真正从桥节点进入 Stage3）。
- Event D：正确收尾（到达目标并正确停下），在 Event A/B/C 成立的前提下统计。

核心特性：
1. 事件统计基于真实生成行为频率（非 softmax 概率质量）。
2. Event A/B/C/D 的判定互相依赖（A→B→C→D），并会输出链式期望成功率。
3. 保留行为统计（behavior_summary.csv）与阶段分析（phase_summary.csv）。
4. 可选输出 per-pair 明细（per_pair_step_*.csv），便于逐样本排查。
5. Prompt 结尾仍手动拼接 Source，确保路径解析正确。

用法示例：
python analyze_s1s3_phases_v3.py \
    --data-dir data/datasets/graphA_pg020_tier3 \
    --checkpoints-dir out/ql_run \
    --run-type ql \
    --step-start 2000 \
    --step-end 20000 \
    --step-interval 2000 \
    --output-dir analysis/ql_phases \
    --save-per-pair
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class PairInfo:
    source: int
    target: int
    path_tokens: List[int]
    first_stage2: Optional[int]
    bridge_candidates: List[int]


@dataclass
class BehaviorRecord:
    step: int
    pair_index: int
    source: int
    target: int
    category: str
    stop_reached: bool
    path_length: int
    stage2_count: int
    first_action: Optional[int]
    target_index: Optional[int]
    tokens: List[int]
    raw_tokens: List[str]


@dataclass
class PhaseEventRecord:
    step: int
    pair_index: int
    source: int
    target: int

    first_action: Optional[int]
    first_valid: bool
    first_is_stage2: bool
    first_is_bridge: bool
    first_is_direct_target: bool
    first_is_invalid: bool

    hit_stage2: bool
    first_stage2: Optional[int]
    first_stage2_index: Optional[int]
    first_stage2_is_bridge: bool
    bridge_hit_any: bool
    stage2_available: bool
    bridge_candidates_count: int
    bridge_candidates: Tuple[int, ...]

    used_stage2: bool
    path_success: bool
    category: str

    legal_start: bool
    stage3_after_bridge: bool
    stage3_entry_index: Optional[int]
    bridge_to_stage3_hops: Optional[int]


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze S1→S3 composition behavior with phase decomposition (v3).")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--run-type", type=str, choices=["sft", "pg", "ql"], required=True)
    parser.add_argument("--ckpt-pattern", type=str, default=None)
    parser.add_argument("--step-start", type=int, required=True)
    parser.add_argument("--step-end", type=int, required=True)
    parser.add_argument("--step-interval", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=2025)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--save-per-pair", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据加载与预处理
# ---------------------------------------------------------------------------

def load_stage_info(stage_info_path: Path) -> List[List[int]]:
    with open(stage_info_path, "rb") as f:
        stage_info = torch.load(f, map_location="cpu") if stage_info_path.suffix == ".pt" else None
    if stage_info is None:
        import pickle
        with open(stage_info_path, "rb") as f:
            stage_info = pickle.load(f)
    return [list(map(int, stage)) for stage in stage_info.get("stages", [])]


def load_meta(meta_path: Path) -> dict:
    import pickle
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def load_graph(graph_path: Path) -> nx.DiGraph:
    return nx.read_graphml(graph_path)


def build_successor_map(G: nx.DiGraph) -> Dict[int, List[int]]:
    succ_map: Dict[int, List[int]] = {}
    for node in G.nodes:
        succ_map[int(node)] = [int(nbr) for nbr in G.successors(node)]
    return succ_map


def parse_test_pairs(test_path: Path, stages: List[List[int]]) -> List[PairInfo]:
    S1, S2, S3 = map(set, stages[:3])
    pairs: List[PairInfo] = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src, tgt = int(parts[0]), int(parts[1])
            if src not in S1 or tgt not in S3:
                continue
            path_tokens = list(map(int, parts[2:]))
            stage2_nodes = [n for n in path_tokens if n in S2]
            pairs.append(
                PairInfo(
                    source=src,
                    target=tgt,
                    path_tokens=path_tokens,
                    first_stage2=stage2_nodes[0] if stage2_nodes else None,
                    bridge_candidates=[]
                )
            )
    return pairs


def assign_bridge_candidates(
    pairs: List[PairInfo],
    descendants_map: Dict[int, set[int]],
    stage_sets: Dict[str, set[int]],
) -> None:
    S2 = stage_sets["S2"]
    for info in pairs:
        reachable = descendants_map.get(info.source, set())
        stage2_reachable = reachable.intersection(S2)
        candidates = []
        for node in stage2_reachable:
            if info.target in descendants_map.get(node, set()):
                candidates.append(node)
        info.bridge_candidates = sorted(set(candidates))


def default_ckpt_pattern(run_type: str) -> str:
    if run_type == "sft":
        return "ckpt_{step}.pt"
    if run_type == "pg":
        return "ckpt_pg_{step}.pt"
    if run_type == "ql":
        return "ckpt_ql_{step}.pt"
    return "ckpt_{step}.pt"


def create_model_from_checkpoint(ckpt_path: Path, device: torch.device, vocab_size: int) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_args = ckpt.get("model_args", {})
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)

    state_dict = ckpt["model"]
    ckpt_block_size = model_args.get("block_size")
    model_block_size = config.block_size

    if ckpt_block_size and model_block_size > ckpt_block_size:
        wpe = state_dict.get("transformer.wpe.weight")
        if wpe is not None:
            new_wpe = model.transformer.wpe.weight.detach().clone()
            new_wpe[: wpe.size(0)] = wpe
            state_dict["transformer.wpe.weight"] = new_wpe

    keys_to_remove = [k for k in state_dict.keys() if k.endswith("attn.bias") or k.endswith("attn.mask")]
    for k in keys_to_remove:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 路径生成与分类
# ---------------------------------------------------------------------------

def run_greedy_generation(
    model: GPT,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    source: int,
    target: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    stop_token_id: int,
) -> Tuple[List[int], List[str], bool]:
    prompt_ids = [stoi[str(source)], stoi[str(target)], stoi[str(source)]]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k if top_k > 0 else None,
    }
    with torch.no_grad():
        generated = model.generate(context, **gen_kwargs)[0].tolist()

    new_ids = generated[len(prompt_ids):]
    raw_tokens: List[str] = []
    digits: List[int] = []
    stop_reached = False

    for tid in new_ids:
        if tid == stop_token_id:
            stop_reached = True
            break
        token = itos.get(tid, "[UNK]")
        raw_tokens.append(token)
        if token.isdigit():
            digits.append(int(token))
        else:
            digits.append(math.inf)

    return digits, raw_tokens, stop_reached


def build_path_from_digits(digits: List[int], source: int, stop_reached: bool) -> Tuple[List[int], str]:
    if not digits:
        return [source], "STOP_BEFORE_START"
    if digits[0] == math.inf:
        return [source], "INVALID_TOKEN"

    if digits[0] != source:
        clean = [source] + [d for d in digits if d != math.inf]
        return clean, "SRC_MISMATCH"

    clean_digits: List[int] = []
    for val in digits:
        if val == math.inf:
            return clean_digits, "INVALID_TOKEN"
        clean_digits.append(val)

    if not stop_reached:
        return clean_digits, "NO_EOS"

    return clean_digits, "OK"


def classify_behavior(
    path_nodes: List[int],
    base_status: str,
    source: int,
    target: int,
    stage_sets: Dict[str, set[int]],
    graph: nx.DiGraph,
) -> Tuple[str, int, int, Optional[int]]:
    S2 = stage_sets["S2"]

    if base_status == "STOP_BEFORE_START":
        return "STOP_BEFORE_START", 0, 0, None
    if base_status == "INVALID_TOKEN":
        return "INVALID_TOKEN", len(path_nodes), 0, None

    valid_edges = True
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            valid_edges = False
            break

    stage2_count = sum(1 for n in path_nodes if n in S2)
    target_index: Optional[int] = None
    for i, n in enumerate(path_nodes):
        if n == target:
            target_index = i
            break

    if base_status == "NO_EOS":
        if valid_edges and target_index is not None:
            return "OVER_SHOOT", len(path_nodes), stage2_count, target_index
        return "NO_EOS", len(path_nodes), stage2_count, target_index

    if base_status == "SRC_MISMATCH":
        return "SRC_MISMATCH", len(path_nodes), stage2_count, target_index

    if not valid_edges:
        return "INVALID_EDGE", len(path_nodes), stage2_count, target_index
    if target_index is None:
        return "MISSING_TARGET", len(path_nodes), stage2_count, target_index
    if target_index != len(path_nodes) - 1:
        return "OVER_SHOOT", len(path_nodes), stage2_count, target_index

    if stage2_count == 0:
        if len(path_nodes) >= 2 and path_nodes[1] == target:
            return "DIRECT_JUMP", len(path_nodes), stage2_count, target_index
        return "NO_STAGE2", len(path_nodes), stage2_count, target_index

    return "SUCCESS", len(path_nodes), stage2_count, target_index


# ---------------------------------------------------------------------------
# 汇总函数
# ---------------------------------------------------------------------------

def aggregate_behavior(records: List[BehaviorRecord]) -> Dict[str, float]:
    total = len(records)
    summary: Dict[str, float] = {
        "num_pairs": total,
        "avg_path_length": float(np.mean([r.path_length for r in records])) if records else 0.0,
        "avg_stage2_count": float(np.mean([r.stage2_count for r in records])) if records else 0.0,
    }

    counter = Counter(r.category for r in records)
    for cat in set(counter.keys()) | {"SUCCESS"}:
        summary[f"count_{cat}"] = counter[cat]
        summary[f"rate_{cat}"] = counter[cat] / total if total else 0.0

    return summary


def aggregate_phase_events(records: List[PhaseEventRecord]) -> Dict[str, float]:
    if not records:
        return {}

    total = len(records)
    success_total = sum(1 for r in records if r.path_success)
    success_rate = success_total / total if total else 0.0

    stage2_available_total = sum(1 for r in records if r.stage2_available)
    stage2_available_rate = stage2_available_total / total if total else 0.0

    eventA_total = sum(1 for r in records if r.legal_start)
    eventA_rate = eventA_total / total if total else 0.0

    eventB_success = sum(1 for r in records if r.legal_start and r.bridge_hit)
    eventB_total = eventA_total
    eventB_rate_given_A = eventB_success / eventB_total if eventB_total else 0.0

    eventB_total_available = sum(1 for r in records if r.legal_start and r.stage2_available)
    eventB_success_available = sum(
        1 for r in records if r.legal_start and r.stage2_available and r.bridge_hit
    )
    eventB_rate_given_A_and_available = (
        eventB_success_available / eventB_total_available if eventB_total_available else 0.0
    )

    eventC_success = sum(
        1 for r in records if r.legal_start and r.bridge_hit and r.stage3_after_bridge
    )
    eventC_total = eventB_success
    eventC_rate_given_AB = eventC_success / eventC_total if eventC_total else 0.0

    eventD_success = sum(
        1 for r in records if r.legal_start and r.bridge_hit and r.stage3_after_bridge and r.path_success
    )
    eventD_total = eventC_success
    eventD_rate_given_ABC = eventD_success / eventD_total if eventD_total else 0.0

    expected_success = (
        eventA_rate * eventB_rate_given_A * eventC_rate_given_AB * eventD_rate_given_ABC
    )
    success_gap = success_rate - expected_success

    bridge_hit_any_total = sum(1 for r in records if r.bridge_hit)
    bridge_hit_any_rate = bridge_hit_any_total / total if total else 0.0

    first_stage2_is_bridge_total = sum(
        1 for r in records if r.hit_stage2 and r.first_stage2_is_bridge
    )
    first_stage2_bridge_rate_given_A = (
        first_stage2_is_bridge_total / sum(1 for r in records if r.hit_stage2) if sum(1 for r in records if r.hit_stage2) else 0.0
    )

    first_stage2_indices = [
        r.first_stage2_index for r in records if r.first_stage2_index is not None
    ]
    first_stage2_index_mean = float(np.mean(first_stage2_indices)) if first_stage2_indices else 0.0
    first_stage2_index_median = float(np.median(first_stage2_indices)) if first_stage2_indices else 0.0

    bridge_to_stage3_hops = [
        r.bridge_to_stage3_hops for r in records if r.bridge_to_stage3_hops is not None
    ]
    bridge_to_stage3_hops_mean = float(np.mean(bridge_to_stage3_hops)) if bridge_to_stage3_hops else 0.0
    bridge_to_stage3_hops_median = float(np.median(bridge_to_stage3_hops)) if bridge_to_stage3_hops else 0.0

    first_valid_total = sum(1 for r in records if r.first_valid)
    first_valid_rate = first_valid_total / total if total else 0.0
    first_invalid_total = sum(1 for r in records if r.first_is_invalid)
    first_invalid_rate = first_invalid_total / total if total else 0.0
    first_stage2_total = sum(1 for r in records if r.first_is_stage2)
    first_stage2_rate = first_stage2_total / total if total else 0.0
    first_bridge_total = sum(1 for r in records if r.first_is_bridge)
    first_bridge_rate = first_bridge_total / total if total else 0.0
    first_direct_target_total = sum(1 for r in records if r.first_is_direct_target)
    first_direct_target_rate = first_direct_target_total / total if total else 0.0

    used_stage2_total = sum(1 for r in records if r.used_stage2)
    used_stage2_rate = used_stage2_total / total if total else 0.0

    success_without_stage2 = sum(1 for r in records if r.path_success and not r.hit_stage2)
    success_with_stage2_no_bridge = sum(
        1 for r in records if r.path_success and r.hit_stage2 and not r.bridge_hit
    )
    success_with_bridge = sum(
        1 for r in records if r.path_success and r.bridge_hit
    )

    avg_bridge_candidates_available = (
        float(
            np.mean(
                [r.bridge_candidates_count for r in records if r.bridge_candidates_count > 0]
            )
        )
        if stage2_available_total > 0
        else 0.0
    )

    stage3_after_bridge_total = sum(1 for r in records if r.stage3_after_bridge)
    stage3_after_bridge_rate = stage3_after_bridge_total / total if total else 0.0

    summary: Dict[str, float] = {
        "total_pairs": total,
        "success_total": success_total,
        "success_rate": success_rate,
        "stage2_available_total": stage2_available_total,
        "stage2_available_rate": stage2_available_rate,

        "eventA_total": eventA_total,
        "eventA_rate": eventA_rate,

        "eventB_total": eventB_total,
        "eventB_success": eventB_success,
        "eventB_rate_given_A": eventB_rate_given_A,
        "eventB_total_available": eventB_total_available,
        "eventB_success_available": eventB_success_available,
        "eventB_rate_given_A_and_available": eventB_rate_given_A_and_available,

        "eventC_total": eventC_total,
        "eventC_success": eventC_success,
        "eventC_rate_given_AB": eventC_rate_given_AB,

        "eventD_total": eventD_total,
        "eventD_success": eventD_success,
        "eventD_rate_given_ABC": eventD_rate_given_ABC,

        "expected_success_from_chain": expected_success,
        "success_gap_vs_expected": success_gap,

        "bridge_hit_any_total": bridge_hit_any_total,
        "bridge_hit_any_rate": bridge_hit_any_rate,
        "stage3_after_bridge_total": stage3_after_bridge_total,
        "stage3_after_bridge_rate": stage3_after_bridge_rate,

        "first_stage2_is_bridge_total": first_stage2_is_bridge_total,
        "first_stage2_bridge_rate_given_A": first_stage2_bridge_rate_given_A,
        "first_stage2_index_mean": first_stage2_index_mean,
        "first_stage2_index_median": first_stage2_index_median,

        "bridge_to_stage3_hops_mean": bridge_to_stage3_hops_mean,
        "bridge_to_stage3_hops_median": bridge_to_stage3_hops_median,

        "first_valid_total": first_valid_total,
        "first_valid_rate": first_valid_rate,
        "first_invalid_total": first_invalid_total,
        "first_invalid_rate": first_invalid_rate,
        "first_stage2_total": first_stage2_total,
        "first_stage2_rate": first_stage2_rate,
        "first_bridge_total": first_bridge_total,
        "first_bridge_rate": first_bridge_rate,
        "first_direct_target_total": first_direct_target_total,
        "first_direct_target_rate": first_direct_target_rate,

        "used_stage2_total": used_stage2_total,
        "used_stage2_rate": used_stage2_rate,
        "success_without_stage2": success_without_stage2,
        "success_with_stage2_no_bridge": success_with_stage2_no_bridge,
        "success_with_bridge": success_with_bridge,
        "avg_bridge_candidates_available": avg_bridge_candidates_available,
    }

    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = sorted(all_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_info_path = data_dir / "stage_info.pkl"
    meta = load_meta(data_dir / "meta.pkl")
    stages = load_stage_info(stage_info_path)
    stage_sets = {
        "S1": set(stages[0]),
        "S2": set(stages[1]),
        "S3": set(stages[2]),
    }
    graph = load_graph(data_dir / "composition_graph.graphml")

    pairs_raw = parse_test_pairs(data_dir / "test.txt", stages)
    succ_map = build_successor_map(graph)
    succ_set_map = {node: set(neighs) for node, neighs in succ_map.items()}
    descendants_map: Dict[int, set[int]] = {
        int(node): {int(x) for x in nx.descendants(graph, node)}
        for node in graph.nodes
    }
    assign_bridge_candidates(pairs_raw, descendants_map, stage_sets)

    if args.max_samples > 0:
        random.seed(args.sample_seed)
        pairs = random.sample(pairs_raw, min(args.max_samples, len(pairs_raw)))
    else:
        pairs = pairs_raw

    if not args.quiet:
        print(f"Analyzing {len(pairs)} S1→S3 pairs...")

    steps = range(args.step_start, args.step_end + 1, args.step_interval)
    pattern = args.ckpt_pattern or default_ckpt_pattern(args.run_type)

    behavior_rows: List[Dict[str, object]] = []
    phase_rows: List[Dict[str, object]] = []

    iterator_steps = steps
    if args.progress and not args.quiet:
        from tqdm import tqdm
        iterator_steps = tqdm(list(steps), desc="Checkpoints")

    for step in iterator_steps:
        ckpt_path = ckpt_dir / pattern.format(step=step)
        if not ckpt_path.exists():
            if not args.quiet:
                print(f"[Skip] checkpoint not found: {ckpt_path}")
            continue

        if not args.quiet:
            print(f"Processing checkpoint: {ckpt_path}")

        model = create_model_from_checkpoint(ckpt_path, device, meta["vocab_size"])

        beh_recs: List[BehaviorRecord] = []
        phase_recs: List[PhaseEventRecord] = []

        pair_iterator = enumerate(pairs)
        if args.progress and not args.quiet:
            from tqdm import tqdm
            pair_iterator = enumerate(tqdm(pairs, leave=False, desc=f"Pairs@{step}"))

        for idx, info in pair_iterator:
            digits, raw_tokens, stop = run_greedy_generation(
                model=model,
                stoi=meta["stoi"],
                itos=meta["itos"],
                source=info.source,
                target=info.target,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
                stop_token_id=meta["stoi"]["\n"],
            )

            full_digits = [info.source] + digits
            path_nodes, status = build_path_from_digits(full_digits, info.source, stop)

            category, path_len, stage2_cnt, target_idx = classify_behavior(
                path_nodes,
                status,
                info.source,
                info.target,
                stage_sets,
                graph,
            )

            first_action = path_nodes[1] if len(path_nodes) > 1 else None
            valid_neighbors = succ_set_map.get(info.source, set())

            first_valid = first_action is not None and first_action in valid_neighbors
            first_is_stage2 = first_action in stage_sets["S2"] if first_action is not None else False
            first_is_bridge = (
                first_valid and first_action in info.bridge_candidates
            )
            first_is_direct_target = first_action == info.target if first_action is not None else False
            first_is_invalid = not first_valid

            stage2_nodes_in_path = [n for n in path_nodes if n in stage_sets["S2"]]
            hit_stage2 = len(stage2_nodes_in_path) > 0
            first_stage2 = stage2_nodes_in_path[0] if hit_stage2 else None
            first_stage2_index = (
                path_nodes.index(first_stage2) if hit_stage2 else None
            )
            bridge_candidate_set = set(info.bridge_candidates)
            first_stage2_is_bridge = (
                first_stage2 in bridge_candidate_set if hit_stage2 else False
            )
            bridge_hit_any = any(n in bridge_candidate_set for n in stage2_nodes_in_path)
            stage2_available = len(info.bridge_candidates) > 0

            # Stage3 after bridge (Event C 判定基础)
            bridge_indices = [
                i for i, node in enumerate(path_nodes) if node in bridge_candidate_set
            ]
            first_bridge_index = bridge_indices[0] if bridge_indices else None
            stage3_after_bridge = False
            stage3_entry_index: Optional[int] = None
            bridge_to_stage3_hops: Optional[int] = None

            if first_bridge_index is not None:
                for j in range(first_bridge_index, len(path_nodes) - 1):
                    curr_node = path_nodes[j]
                    next_node = path_nodes[j + 1]
                    if curr_node in bridge_candidate_set and next_node in stage_sets["S3"]:
                        stage3_after_bridge = True
                        stage3_entry_index = j + 1
                        bridge_to_stage3_hops = stage3_entry_index - first_bridge_index
                        break

            beh_recs.append(
                BehaviorRecord(
                    step=step,
                    pair_index=idx,
                    source=info.source,
                    target=info.target,
                    category=category,
                    stop_reached=stop,
                    path_length=path_len,
                    stage2_count=stage2_cnt,
                    first_action=first_action,
                    target_index=target_idx,
                    tokens=path_nodes,
                    raw_tokens=raw_tokens,
                )
            )

            phase_recs.append(
                PhaseEventRecord(
                    step=step,
                    pair_index=idx,
                    source=info.source,
                    target=info.target,
                    first_action=first_action,
                    first_valid=first_valid,
                    first_is_stage2=first_is_stage2,
                    first_is_bridge=first_is_bridge,
                    first_is_direct_target=first_is_direct_target,
                    first_is_invalid=first_is_invalid,
                    hit_stage2=hit_stage2,
                    first_stage2=first_stage2,
                    first_stage2_index=first_stage2_index,
                    first_stage2_is_bridge=first_stage2_is_bridge,
                    bridge_hit_any=bridge_hit_any,
                    stage2_available=stage2_available,
                    bridge_candidates_count=len(info.bridge_candidates),
                    bridge_candidates=tuple(info.bridge_candidates),
                    used_stage2=stage2_cnt > 0,
                    path_success=(category == "SUCCESS"),
                    category=category,
                    legal_start=first_valid,
                    stage3_after_bridge=stage3_after_bridge,
                    stage3_entry_index=stage3_entry_index,
                    bridge_to_stage3_hops=bridge_to_stage3_hops,
                )
            )

        beh_summary = aggregate_behavior(beh_recs)
        beh_summary["step"] = step
        behavior_rows.append(beh_summary)

        phase_summary = aggregate_phase_events(phase_recs)
        phase_summary["step"] = step
        phase_rows.append(phase_summary)

        if args.save_per_pair:
            per_pair_rows: List[Dict[str, object]] = []
            for beh, phase in zip(beh_recs, phase_recs):
                per_pair_rows.append(
                    {
                        "step": step,
                        "pair_index": beh.pair_index,
                        "source": beh.source,
                        "target": beh.target,
                        "category": beh.category,
                        "path_success": int(phase.path_success),
                        "path_length": beh.path_length,
                        "stage2_count": beh.stage2_count,
                        "stop_reached": int(beh.stop_reached),
                        "first_action": "" if phase.first_action is None else phase.first_action,
                        "first_valid": int(phase.first_valid),
                        "first_is_stage2": int(phase.first_is_stage2),
                        "first_is_bridge": int(phase.first_is_bridge),
                        "first_is_direct_target": int(phase.first_is_direct_target),
                        "first_is_invalid": int(phase.first_is_invalid),
                        "hit_stage2": int(phase.hit_stage2),
                        "first_stage2": "" if phase.first_stage2 is None else phase.first_stage2,
                        "first_stage2_index": "" if phase.first_stage2_index is None else phase.first_stage2_index,
                        "first_stage2_is_bridge": int(phase.first_stage2_is_bridge),
                        "bridge_hit_any": int(phase.bridge_hit_any),
                        "stage2_available": int(phase.stage2_available),
                        "bridge_candidates_count": phase.bridge_candidates_count,
                        "bridge_candidates": " ".join(map(str, phase.bridge_candidates)),
                        "used_stage2": int(phase.used_stage2),
                        "legal_start": int(phase.legal_start),
                        "stage3_after_bridge": int(phase.stage3_after_bridge),
                        "stage3_entry_index": "" if phase.stage3_entry_index is None else phase.stage3_entry_index,
                        "bridge_to_stage3_hops": "" if phase.bridge_to_stage3_hops is None else phase.bridge_to_stage3_hops,
                        "path_tokens": " ".join(map(str, beh.tokens)),
                        "raw_tokens": " ".join(beh.raw_tokens),
                    }
                )
            per_pair_path = out_dir / f"per_pair_step_{step}.csv"
            write_csv(per_pair_path, per_pair_rows)

    write_csv(out_dir / "behavior_summary.csv", behavior_rows)
    write_csv(out_dir / "phase_summary.csv", phase_rows)

    if not args.quiet:
        print("Analysis complete.")
        print(f"behavior_summary.csv saved to {out_dir}")
        print(f"phase_summary.csv saved to {out_dir}")
        if args.save_per_pair:
            print("Per-pair CSV files were also generated.")


if __name__ == "__main__":
    main()