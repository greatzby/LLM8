#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1→S3 组合任务深度机制分析脚本 (Based on Paper Methodology)

功能：
1. 行为分析 (Behavior): 统计 Success, Invalid Edge, No Stage 2 (Wandering) 等宏观现象。
2. 阶段动力学 (Stage Dynamics): 将推理能力分解为三个条件概率嵌套：
   - P(Valid): 是否遵循图的基本连通性？ (Syntax)
   - P(Structure | Valid): 是否知道要往 Stage 2 走？ (Latent Structure)
   - P(Reasoning | Structure): 在 Stage 2 中是否选对了能到达目标的桥梁？ (Exact Reasoning)

使用方法示例：
python analysis_paper_mechanisms.py \
    --data-dir data/datasets/graphA_pg020_tier3 \
    --checkpoints-dir out/ql_run_2025 \
    --run-type ql \
    --output-dir analysis_results/ql_group3 \
    --step-start 0 --step-end 20000 --step-interval 1000 \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 假设 model.py 在当前目录下
from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------

@dataclass
class PairInfo:
    source: int
    target: int
    path_tokens: List[int]

@dataclass
class BehaviorRecord:
    step: int
    pair_index: int
    source: int
    target: int
    category: str          # SUCCESS, INVALID_EDGE, NO_STAGE2, etc.
    stop_reached: bool
    path_length: int
    stage2_count: int
    tokens: List[int]

@dataclass
class StageEventRecord:
    step: int
    pair_index: int
    source: int
    target: int
    # --- 核心 Paper 指标 ---
    prob_valid: float              # P(Next in Neighbors)
    prob_bridge_given_valid: float # P(Next in S2 | Next in Neighbors)
    prob_causal_given_bridge: float# P(Next reaches Target | Next in S2)
    # --- 辅助统计 ---
    support_valid: int
    support_bridge: int
    support_causal: int
    chance_level: float            # Random Guess Accuracy

# ---------------------------------------------------------------------------
# 工具函数：加载与预处理
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze S1->S3 composition mechanisms.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--checkpoints-dir", type=str, required=True, help="Path to checkpoints")
    parser.add_argument("--run-type", type=str, choices=["sft", "pg", "ql"], required=True)
    parser.add_argument("--ckpt-pattern", type=str, default=None, help="e.g., ckpt_{step}.pt")
    
    parser.add_argument("--step-start", type=int, required=True)
    parser.add_argument("--step-end", type=int, required=True)
    parser.add_argument("--step-interval", type=int, required=True)
    
    parser.add_argument("--max-samples", type=int, default=0, help="Max pairs to analyze (0 for all)")
    parser.add_argument("--sample-seed", type=int, default=2025)
    
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0, help="Greedy decoding by default")
    parser.add_argument("--top-k", type=int, default=0)
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()

def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def load_stage_info(stage_info_path: Path) -> List[List[int]]:
    with open(stage_info_path, "rb") as f:
        info = pickle.load(f)
    return [list(map(int, stage)) for stage in info.get("stages", [])]

def load_graph(graph_path: Path) -> nx.DiGraph:
    return nx.read_graphml(graph_path)

def parse_test_pairs(test_path: Path, stages: List[List[int]]) -> List[PairInfo]:
    """只加载 S1 -> S3 的测试样本"""
    S1, S2, S3 = set(stages[0]), set(stages[1]), set(stages[2])
    pairs = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            src, tgt = int(parts[0]), int(parts[1])
            # 仅筛选 S1 -> S3 任务
            if src in S1 and tgt in S3:
                path_tokens = list(map(int, parts[2:]))
                pairs.append(PairInfo(source=src, target=tgt, path_tokens=path_tokens))
    return pairs

def precompute_reachability_map(stage_sets: Dict[str, set], graph: nx.DiGraph) -> Dict[int, Set[int]]:
    """
    预计算：对于每个 Target (S3)，哪些 S2 节点是它的祖先（即合法的桥梁）？
    返回: {target_id: {valid_bridge_id_1, valid_bridge_id_2, ...}}
    """
    print("Precomputing reachability map (S2 -> S3)...")
    s3_to_valid_s2 = defaultdict(set)
    s2_nodes = stage_sets["S2"]
    s3_nodes = stage_sets["S3"]
    
    # 反向图或者遍历 descendants 都可以。这里遍历 S2 的 descendants
    for s2 in tqdm(s2_nodes, desc="Mapping S2 reachability"):
        descendants = nx.descendants(graph, str(s2))
        for d in descendants:
            d_int = int(d)
            if d_int in s3_nodes:
                s3_to_valid_s2[d_int].add(s2)
    return s3_to_valid_s2

def create_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_args = ckpt.get("model_args", {})
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)
    
    state_dict = ckpt["model"]
    
    # 处理 Block Size 扩容问题 (如果 RL 阶段扩容了)
    ckpt_block_size = model_args.get("block_size")
    model_block_size = config.block_size
    if ckpt_block_size and model_block_size > ckpt_block_size:
        wpe = state_dict.get("transformer.wpe.weight")
        if wpe is not None:
            new_wpe = model.transformer.wpe.weight.detach().clone()
            new_wpe[:wpe.size(0)] = wpe
            state_dict["transformer.wpe.weight"] = new_wpe
            
    # 移除不需要的 bias/mask 键 (如果存在)
    keys_to_remove = [k for k in state_dict.keys() if k.endswith("attn.bias") or k.endswith("attn.mask")]
    for k in keys_to_remove:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# 核心分析逻辑 1: 行为分类 (Behavior Classification)
# ---------------------------------------------------------------------------

def run_greedy_generation(model, stoi, itos, source, target, max_new_tokens, device, stop_token_id,temperature,top_k):
    # Prompt: S -> T -> S
    prompt_ids = [stoi[str(source)], stoi[str(target)], stoi[str(source)]]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Greedy decoding (top_k=1 implicitly via argmax if temp=0, or use generate default)
        generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    
    new_ids = generated[len(prompt_ids):]
    digits = []
    stop_reached = False
    
    for tid in new_ids:
        if tid == stop_token_id:
            stop_reached = True
            break
        token = itos.get(tid, "[UNK]")
        if token.isdigit():
            digits.append(int(token))
        else:
            digits.append(math.inf) # 标记非法 token
            
    return digits, stop_reached

def classify_behavior(full_path_nodes: List[int], stop_reached: bool, target: int, stage_sets: Dict[str, set], graph: nx.DiGraph) -> str:
    """
    对完整路径进行分类。
    full_path_nodes: [Source, Next, ..., Target?]
    """
    # 1. 检查 Token 合法性
    if any(n == math.inf for n in full_path_nodes):
        return "INVALID_TOKEN"
    
    # 2. 检查边合法性 (Syntax/Graph Knowledge)
    for u, v in zip(full_path_nodes[:-1], full_path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            return "INVALID_EDGE"
            
    # 3. 检查是否到达 Target
    try:
        target_idx = full_path_nodes.index(target)
    except ValueError:
        target_idx = -1
        
    if target_idx == -1:
        # 没到 Target，检查是否在 S1 打转 (Wandering)
        s2_nodes = [n for n in full_path_nodes if n in stage_sets["S2"]]
        if not s2_nodes:
            return "NO_STAGE2" # 甚至没去 S2
        if not stop_reached:
            return "NO_EOS" # 可能是 Loop 或者太长
        return "MISSING_TARGET" # 去了 S2 但没到 S3
        
    # 4. 到达了 Target
    # 检查是否 Overshoot (到了 Target 还没停)
    if target_idx != len(full_path_nodes) - 1:
        return "OVER_SHOOT"
        
    # 5. 检查是否跳过 Stage 2 (Direct Jump S1->S3)
    # 路径中间的部分
    intermediate = full_path_nodes[1:target_idx]
    has_s2 = any(n in stage_sets["S2"] for n in intermediate)
    if not has_s2:
        return "DIRECT_JUMP"
        
    return "SUCCESS"

# ---------------------------------------------------------------------------
# 核心分析逻辑 2: 阶段动力学 (Paper Metrics)
# ---------------------------------------------------------------------------

def compute_stage_event(model, stoi, src, tgt, idx, graph, stage_sets, s3_to_valid_s2, step, device) -> StageEventRecord:
    """
    计算 S1 -> S3 任务中第一步选择的条件概率。
    逻辑链条：
    1. Valid: Next Token 是 Source 的邻居。
    2. Structure: Next Token 属于 Stage 2。
    3. Reasoning: Next Token 是能到达 Target 的 Stage 2 节点。
    """
    # 构造 Prompt: [S, T, S]
    prompt_ids = [stoi[str(src)], stoi[str(tgt)], stoi[str(src)]]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, _ = model(x)
    # 获取最后一个 Token (Source) 之后的 Logits
    next_token_probs = F.softmax(logits[0, -1, :], dim=-1).cpu()
    
    # --- 集合定义 ---
    # A. 所有邻居 (Valid Neighbors)
    neighbors = [int(n) for n in graph.successors(str(src))]
    valid_ids = [stoi[str(n)] for n in neighbors if str(n) in stoi]
    
    # B. S2 邻居 (Stage 2 Neighbors)
    s2_neighbors = [n for n in neighbors if n in stage_sets["S2"]]
    s2_ids = [stoi[str(n)] for n in s2_neighbors if str(n) in stoi]
    
    # C. 正确桥梁 (Correct Bridges)
    # 既是 S2 邻居，又是 Target 的祖先
    valid_bridges = s3_to_valid_s2.get(tgt, set())
    correct_bridge_nodes = [n for n in s2_neighbors if n in valid_bridges]
    correct_ids = [stoi[str(n)] for n in correct_bridge_nodes if str(n) in stoi]
    
    # --- 概率质量求和 ---
    p_valid = float(next_token_probs[valid_ids].sum().item()) if valid_ids else 0.0
    p_s2 = float(next_token_probs[s2_ids].sum().item()) if s2_ids else 0.0
    p_correct = float(next_token_probs[correct_ids].sum().item()) if correct_ids else 0.0
    
    # --- 条件概率指标 (Paper Metrics) ---
    
    # Metric 1: P(Valid) - 语法/图知识保持得怎么样？
    metric_valid = p_valid
    
    # Metric 2: P(Stage 2 | Valid) - 是否知道结构方向？
    # 如果 p_valid 极小，说明模型已经崩了，此时条件概率无意义，置 0
    metric_structure = p_s2 / p_valid if p_valid > 1e-6 else 0.0
    
    # Metric 3: P(Correct | Stage 2) - 推理能力核心指标
    # 在知道要去 S2 的前提下，选对桥的概率
    metric_reasoning = p_correct / p_s2 if p_s2 > 1e-6 else 0.0
    
    # Chance Level: 随机选 S2 选对的概率
    chance = len(correct_bridge_nodes) / len(s2_neighbors) if s2_neighbors else 0.0
    
    return StageEventRecord(
        step=step, pair_index=idx, source=src, target=tgt,
        prob_valid=metric_valid,
        prob_bridge_given_valid=metric_structure,
        prob_causal_given_bridge=metric_reasoning,
        support_valid=len(valid_ids),
        support_bridge=len(s2_ids),
        support_causal=len(correct_ids),
        chance_level=chance
    )

# ---------------------------------------------------------------------------
# 聚合与写入
# ---------------------------------------------------------------------------

def aggregate_behavior(records: List[BehaviorRecord]) -> dict:
    total = len(records)
    if total == 0: return {}
    
    counter = Counter(r.category for r in records)
    summary = {
        "num_pairs": total,
        "avg_path_length": float(np.mean([r.path_length for r in records])),
        "avg_stage2_count": float(np.mean([r.stage2_count for r in records])),
    }
    
    # 确保所有类别都有字段，方便画图
    categories = ["SUCCESS", "INVALID_EDGE", "INVALID_TOKEN", "NO_STAGE2", "MISSING_TARGET", "OVER_SHOOT", "NO_EOS"]
    for cat in categories:
        summary[f"rate_{cat}"] = counter[cat] / total
        
    return summary

def aggregate_stage_events(records: List[StageEventRecord]) -> dict:
    if not records: return {}
    
    # 过滤掉 support 为 0 的情况计算平均值 (避免除零导致的偏差)
    valid_recs = [r.prob_valid for r in records]
    struct_recs = [r.prob_bridge_given_valid for r in records if r.support_valid > 0]
    reason_recs = [r.prob_causal_given_bridge for r in records if r.support_bridge > 0]
    chance_recs = [r.chance_level for r in records if r.support_bridge > 0]
    
    return {
        "prob_valid_mean": float(np.mean(valid_recs)) if valid_recs else 0.0,
        "prob_structure_mean": float(np.mean(struct_recs)) if struct_recs else 0.0,
        "prob_reasoning_mean": float(np.mean(reason_recs)) if reason_recs else 0.0,
        "chance_level_mean": float(np.mean(chance_recs)) if chance_recs else 0.0,
    }

def write_csv(path: Path, rows: List[dict]):
    if not rows: return
    # 扫描所有 key
    keys = set()
    for r in rows:
        keys.update(r.keys())
    sorted_keys = sorted(list(keys))
    
    # 把 'step' 放到第一列
    if 'step' in sorted_keys:
        sorted_keys.remove('step')
        sorted_keys.insert(0, 'step')
        
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys, restval=0)
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Analysis ({args.run_type}) ---")
    print(f"Data: {data_dir}")
    print(f"Output: {out_dir}")
    
    # 1. 加载元数据
    meta = load_meta(data_dir / "meta.pkl")
    stoi, itos = meta["stoi"], meta["itos"]
    stop_token_id = stoi["\n"]
    
    # 2. 加载图结构与 Stage 信息
    stage_info = load_stage_info(data_dir / "stage_info.pkl")
    stage_sets = {"S1": set(stage_info[0]), "S2": set(stage_info[1]), "S3": set(stage_info[2])}
    graph = load_graph(data_dir / "composition_graph.graphml")
    
    # 3. 预计算 Reachability (加速 Paper Metric 计算)
    s3_to_valid_s2 = precompute_reachability_map(stage_sets, graph)
    
    # 4. 准备测试对
    pairs_all = parse_test_pairs(data_dir / "test.txt", stage_info)
    if args.max_samples > 0 and len(pairs_all) > args.max_samples:
        random.seed(args.sample_seed)
        pairs = random.sample(pairs_all, args.max_samples)
    else:
        pairs = pairs_all
    print(f"Analyzing {len(pairs)} S1->S3 pairs.")
    
    # 5. 确定 Checkpoint 列表
    steps = range(args.step_start, args.step_end + 1, args.step_interval)
    default_pattern = {
        "sft": "ckpt_{step}.pt",
        "pg": "ckpt_pg_{step}.pt",
        "ql": "ckpt_ql_{step}.pt"
    }
    pattern = args.ckpt_pattern or default_pattern.get(args.run_type, "ckpt_{step}.pt")
    
    summary_behavior = []
    summary_stage_events = []
    
    # 6. 循环分析
    for step in steps:
        ckpt_path = ckpt_dir / pattern.format(step=step)
        if not ckpt_path.exists():
            if not args.quiet: print(f"Skipping missing checkpoint: {ckpt_path}")
            continue
            
        if not args.quiet: print(f"Analyzing Step {step}...")
        
        model = create_model_from_checkpoint(ckpt_path, device)
        
        beh_records = []
        stg_records = []
        
        iterator = tqdm(pairs, desc=f"Step {step}", leave=False) if not args.quiet else pairs
        
        for i, info in enumerate(iterator):
            # --- A. 行为生成 (Greedy) ---
            gen_digits, stop_reached = run_greedy_generation(
                model, stoi, itos, info.source, info.target, 
                args.max_new_tokens, device, stop_token_id,args.temperature,args.top_k
            )
            
            # 关键修正：手动拼接 Source
            full_path = [info.source] + gen_digits
            
            cat = classify_behavior(full_path, stop_reached, info.target, stage_sets, graph)
            
            s2_count = sum(1 for n in full_path if n != math.inf and n in stage_sets["S2"])
            beh_records.append(BehaviorRecord(
                step=step, pair_index=i, source=info.source, target=info.target,
                category=cat, stop_reached=stop_reached, 
                path_length=len(full_path), stage2_count=s2_count, tokens=full_path
            ))
            
            # --- B. 阶段动力学 (Logits Analysis) ---
            stg_rec = compute_stage_event(
                model, stoi, info.source, info.target, i, 
                graph, stage_sets, s3_to_valid_s2, step, device
            )
            stg_records.append(stg_rec)
            
        # 聚合本 Step 的结果
        agg_beh = aggregate_behavior(beh_records)
        agg_beh["step"] = step
        summary_behavior.append(agg_beh)
        
        agg_stg = aggregate_stage_events(stg_records)
        agg_stg["step"] = step
        summary_stage_events.append(agg_stg)
        
        # 实时保存 (防止程序中断丢失数据)
        write_csv(out_dir / "behavior_summary.csv", summary_behavior)
        write_csv(out_dir / "stage_event_summary.csv", summary_stage_events)
        
    print(f"Analysis complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()