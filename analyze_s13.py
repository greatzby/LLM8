#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1→S3组合任务分析脚本（强化版）。

核心功能概览
------------
1. 行为统计（Behavior Summary）：
   - 成功率、直接跳转、非法边、未终止等分类统计。
   - 平均路径长度、Stage2 访问次数等。

2. Logits 事件统计（Event Probabilities）：
   - P(重复 source)、P(桥节点)、P(target|桥)、P(EOS|target) 等。

3. Stage Event Factorization（全新）：
   - Event A（Syntax）：模型把概率质量分配给合法邻居节点的程度 P(A)。
   - Event B（Orientation）：在合法邻居中，分配给 S2 层节点的概率 P(B|A)。
   - Event C（Planning）：在合法 S2 节点中，分配给能通向目标的节点的概率 P(C|A,B)。
   - 同时输出每个 pair 的 chance level（随机选正确桥节点的期望概率）。

附加能力：
   - 支持按 SFT/PG/QL 不同 checkpoint 命名规则加载。
   - 可选保存 per-pair 的详细 CSV。
   - 支持抽样、进度条等实用选项。

论文映射说明
------------
* Event A ≈ Alchemy 的 "In-Support"。
* Event B ≈ "Reachable"（转向中间层/正确子空间）。
* Event C ≈ "Exact Match" / Causal Planning（真正走向能达目标的桥节点）。

运行示例参见文末。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig  # 请确保该模块可用


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
class EventRecord:
    step: int
    pair_index: int
    source: int
    target: int
    prob_src_repeat: float
    prob_eos_after_prompt: float
    prob_bridge_after_src: float
    prob_target_direct_after_src: float
    prob_eos_after_src: float
    prob_target_after_bridge: Optional[float]
    prob_eos_after_bridge: Optional[float]
    prob_eos_after_target: Optional[float]
    prob_continue_after_target: Optional[float]


@dataclass
class StageEventRecord:
    step: int
    pair_index: int
    source: int
    target: int
    prob_valid: float              # P(A)
    prob_bridge_given_valid: float # P(B|A)
    prob_causal_given_bridge: float# P(C|A,B)
    support_valid: int             # |V_valid|
    support_bridge: int            # |V_bridge|
    support_causal: int            # |V_connected|
    chance_level: float            # 1 / |V_bridge|（仅供参考）


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze S1→S3 composition behavior & logits.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="数据集目录（含 train/test/meta/stage_info/graphml）。")
    parser.add_argument("--checkpoints-dir", type=str, required=True,
                        help="存放 checkpoint 的目录。")
    parser.add_argument("--run-type", type=str, choices=["sft", "pg", "ql"], required=True,
                        help="指定训练类型（决定默认的 ckpt 间隔、命名等）。")
    parser.add_argument("--ckpt-pattern", type=str, default=None,
                        help="checkpoint 文件名模式，需包含 {step}。默认根据 run-type 选择。")

    parser.add_argument("--step-start", type=int, required=True,
                        help="起始迭代步（含）。")
    parser.add_argument("--step-end", type=int, required=True,
                        help="终止迭代步（含）。")
    parser.add_argument("--step-interval", type=int, required=True,
                        help="迭代步间隔。")

    parser.add_argument("--max-samples", type=int, default=0,
                        help="随机抽样的 pair 数量（<=0 表示全量）。")
    parser.add_argument("--sample-seed", type=int, default=2025,
                        help="抽样 random seed。")

    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="自回归生成的最大新 token 数。")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度，0 表示 greedy。")
    parser.add_argument("--top-k", type=int, default=0,
                        help="top-k 采样阈值，<=0 表示关闭。")

    parser.add_argument("--device", type=str, default="cuda:0",
                        help="运行设备，没有 GPU 可改 cpu。")

    parser.add_argument("--output-dir", type=str, required=True,
                        help="结果输出目录。")
    parser.add_argument("--save-per-pair", action="store_true",
                        help="保存每个 checkpoint 的 per-pair CSV。")

    parser.add_argument("--quiet", action="store_true",
                        help="减少控制台输出。")
    parser.add_argument("--progress", action="store_true",
                        help="显示 tqdm 进度条。")
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
    stages = stage_info.get("stages")
    if not stages or len(stages) < 3:
        raise ValueError("stage_info 中需要至少前三层 stage 信息。")
    return [list(map(int, stage)) for stage in stages]


def load_meta(meta_path: Path) -> dict:
    import pickle
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta


def load_graph(graph_path: Path) -> nx.DiGraph:
    G = nx.read_graphml(graph_path)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("composition_graph.graphml 不是 DAG。")
    return G


def build_successor_map(G: nx.DiGraph) -> Dict[int, List[int]]:
    succ_map: Dict[int, List[int]] = {}
    for node in G.nodes:
        succ_map[int(node)] = [int(nbr) for nbr in G.successors(node)]
    return succ_map


def build_descendants_map(G: nx.DiGraph) -> Dict[int, set]:
    desc: Dict[int, set] = {}
    for node in G.nodes:
        desc[int(node)] = {int(x) for x in nx.descendants(G, node)}
    return desc


def parse_test_pairs(test_path: Path, stages: List[List[int]]) -> List[PairInfo]:
    S1, S2, S3 = map(set, stages[:3])
    pairs: List[PairInfo] = []

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src = int(parts[0])
            tgt = int(parts[1])
            if src not in S1 or tgt not in S3:
                continue
            path_tokens = list(map(int, parts[2:]))
            stage2_nodes = [n for n in path_tokens if n in S2]
            first_stage2 = stage2_nodes[0] if stage2_nodes else None
            pairs.append(PairInfo(
                source=src,
                target=tgt,
                path_tokens=path_tokens,
                first_stage2=first_stage2,
                bridge_candidates=[],
            ))
    if not pairs:
        raise ValueError("test.txt 中未找到任何 S1→S3 pair。")
    return pairs


def assign_bridge_candidates(
    pairs: List[PairInfo],
    succ_map: Dict[int, List[int]],
    descendants_map: Dict[int, set],
    stage2_set: set,
) -> None:
    for info in pairs:
        source = info.source
        target = info.target
        candidates = []
        for neighbor in succ_map.get(source, []):
            if neighbor in stage2_set and target in descendants_map.get(neighbor, set()):
                candidates.append(neighbor)
        info.bridge_candidates = candidates


def precompute_reachability_map(
    stage_sets: Dict[str, set],
    descendants_map: Dict[int, set],
) -> Dict[int, set]:
    """
    构建 map: target -> {能够到达该 target 的 S2 节点}。
    使用 descendants_map（target 是否在 S2 节点的后裔集合中）。
    """
    s2_nodes = stage_sets["S2"]
    s3_nodes = stage_sets["S3"]
    s3_to_valid_s2: Dict[int, set] = defaultdict(set)

    for s2 in s2_nodes:
        reachable = descendants_map.get(s2, set())
        for t in reachable:
            if t in s3_nodes:
                s3_to_valid_s2[t].add(s2)
    return s3_to_valid_s2


def maybe_subsample_pairs(
    pairs: List[PairInfo],
    max_samples: int,
    seed: int,
) -> List[PairInfo]:
    if max_samples <= 0 or max_samples >= len(pairs):
        return pairs
    rng = random.Random(seed)
    return rng.sample(pairs, max_samples)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def default_ckpt_pattern(run_type: str) -> str:
    if run_type == "sft":
        return "ckpt_{step}.pt"
    if run_type == "pg":
        return "ckpt_pg_{step}.pt"
    if run_type == "ql":
        return "ckpt_ql_{step}.pt"
    raise ValueError(f"未知 run_type: {run_type}")


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_state_dict_with_block_resize(
    model: GPT,
    state_dict: dict,
    ckpt_block_size: Optional[int],
) -> None:
    model_block_size = model.config.block_size
    if ckpt_block_size is None or ckpt_block_size == model_block_size:
        model.load_state_dict(state_dict, strict=True)
        return
    if model_block_size < ckpt_block_size:
        raise ValueError(
            f"模型 block_size={model_block_size} 小于 checkpoint block_size={ckpt_block_size}，无法加载。"
        )

    copy_state = dict(state_dict)
    wpe_key = "transformer.wpe.weight"
    if wpe_key in copy_state:
        old_weight = copy_state[wpe_key]
        new_weight = model.transformer.wpe.weight.detach().clone()
        new_weight[:old_weight.size(0)] = old_weight
        copy_state[wpe_key] = new_weight

    bias_like = [k for k in list(copy_state.keys()) if k.endswith("attn.bias") or k.endswith("attn.mask")]
    for key in bias_like:
        copy_state.pop(key, None)

    missing, unexpected = model.load_state_dict(copy_state, strict=False)
    allowed_missing = {key for key in missing if key.endswith("attn.bias") or key.endswith("attn.mask")}
    leftover_missing = [k for k in missing if k not in allowed_missing]
    if leftover_missing:
        raise RuntimeError(f"加载 checkpoint 时缺失关键权值: {leftover_missing}")
    if unexpected:
        raise RuntimeError(f"加载 checkpoint 时出现未预期键: {unexpected}")


def create_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    vocab_size: int,
) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_args = ckpt.get("model_args")
    if model_args is None:
        raise ValueError(f"{ckpt_path} 中未找到 model_args。")
    if model_args.get("vocab_size") != vocab_size:
        raise ValueError(
            f"{ckpt_path} 的 vocab_size={model_args.get('vocab_size')} 与数据集 meta 不一致（{vocab_size}）。"
        )
    config = GPTConfig(**model_args)
    model = GPT(config).to(device)
    ckpt_block_size = model_args.get("block_size")
    load_state_dict_with_block_resize(model, ckpt["model"], ckpt_block_size)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 自回归与分类
# ---------------------------------------------------------------------------

def tokenize_context(
    stoi: Dict[str, int],
    nodes: Sequence[int],
) -> List[int]:
    return [stoi[str(node)] for node in nodes]


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
    prompt_nodes = [source, target, source]
    prompt_ids = tokenize_context(stoi, prompt_nodes)
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


def build_path_from_digits(
    digits: List[int],
    source: int,
    stop_reached: bool,
) -> Tuple[List[int], str]:
    if not digits:
        return [source], "STOP_BEFORE_START"

    if digits[0] == math.inf:
        return [source], "INVALID_TOKEN"

    if digits[0] != source:
        clean_digits = [source] + [d for d in digits if d != math.inf]
        return clean_digits, "SRC_MISMATCH"

    clean_digits = []
    for val in digits:
        if val == math.inf:
            return [source] + clean_digits, "INVALID_TOKEN"
        clean_digits.append(val)

    if not stop_reached:
        return clean_digits, "NO_EOS"

    return clean_digits, "OK"


def classify_behavior(
    path_nodes: List[int],
    base_status: str,
    source: int,
    target: int,
    stage_sets: Dict[str, set],
    graph: nx.DiGraph,
) -> Tuple[str, int, Optional[int], Optional[int]]:
    S2 = stage_sets["S2"]

    if base_status == "STOP_BEFORE_START":
        return "STOP_BEFORE_START", 0, None, None
    if base_status == "INVALID_TOKEN":
        return "INVALID_TOKEN", len(path_nodes), None, None

    valid_edges = True
    target_positions = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            valid_edges = False
            break
        if v == target:
            target_positions.append(len(target_positions) + 1)

    stage2_count = sum(1 for node in path_nodes if node in S2)
    first_action = path_nodes[1] if len(path_nodes) >= 2 else None

    target_index = None
    for idx, node in enumerate(path_nodes):
        if node == target:
            target_index = idx
            break

    if base_status == "NO_EOS":
        if valid_edges and target_index is not None:
            return "OVER_SHOOT", len(path_nodes), stage2_count, target_index
        return "NO_EOS", len(path_nodes), stage2_count, target_index

    if base_status == "SRC_MISMATCH":
        if not valid_edges:
            return "SRC_MISMATCH_INVALID_EDGE", len(path_nodes), stage2_count, target_index
        if target_index is None:
            return "SRC_MISMATCH_NO_TARGET", len(path_nodes), stage2_count, target_index
        if stage2_count == 0:
            return "SRC_MISMATCH_NO_STAGE2", len(path_nodes), stage2_count, target_index
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


def get_next_token_probs(
    model: GPT,
    context_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    x = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs.cpu()


# ---------------------------------------------------------------------------
# Stage Event 分析
# ---------------------------------------------------------------------------

def compute_stage_event(
    probs: torch.Tensor,
    src: int,
    tgt: int,
    pair_index: int,
    stoi: Dict[str, int],
    graph: nx.DiGraph,
    stage_sets: Dict[str, set],
    s3_to_valid_s2: Dict[int, set],
    step: int,
) -> StageEventRecord:
    neighbor_nodes = [int(n) for n in graph.successors(str(src))]
    valid_ids = [stoi[str(n)] for n in neighbor_nodes if str(n) in stoi]
    bridge_nodes = [n for n in neighbor_nodes if n in stage_sets["S2"]]
    bridge_ids = [stoi[str(n)] for n in bridge_nodes if str(n) in stoi]

    causal_nodes = [
        n for n in bridge_nodes
        if n in s3_to_valid_s2.get(tgt, set())
    ]
    causal_ids = [stoi[str(n)] for n in causal_nodes if str(n) in stoi]

    p_valid = float(probs[valid_ids].sum().item()) if valid_ids else 0.0
    p_bridge = float(probs[bridge_ids].sum().item()) if bridge_ids else 0.0
    p_causal = float(probs[causal_ids].sum().item()) if causal_ids else 0.0

    p_bridge_given_valid = p_bridge / p_valid if p_valid > 1e-6 else 0.0
    p_causal_given_bridge = p_causal / p_bridge if p_bridge > 1e-6 else 0.0

    chance_level = 1.0 / len(bridge_nodes) if len(bridge_nodes) > 0 else 0.0

    return StageEventRecord(
        step=step,
        pair_index=pair_index,
        source=src,
        target=tgt,
        prob_valid=p_valid,
        prob_bridge_given_valid=p_bridge_given_valid,
        prob_causal_given_bridge=p_causal_given_bridge,
        support_valid=len(valid_ids),
        support_bridge=len(bridge_ids),
        support_causal=len(causal_ids),
        chance_level=chance_level,
    )


# ---------------------------------------------------------------------------
# 汇总函数
# ---------------------------------------------------------------------------

def aggregate_behavior(records: List[BehaviorRecord]) -> Dict[str, float]:
    total = len(records)
    counter = Counter(rec.category for rec in records)

    avg_path_len = float(np.mean([rec.path_length for rec in records])) if records else 0.0
    avg_stage2 = float(np.mean([rec.stage2_count for rec in records])) if records else 0.0

    target_indices = [rec.target_index for rec in records if rec.target_index is not None]
    avg_target_index = float(np.mean(target_indices)) if target_indices else float("nan")

    summary = {
        "num_pairs": total,
        "avg_path_length": avg_path_len,
        "avg_stage2_count": avg_stage2,
        "avg_target_index": avg_target_index,
    }
    for category, count in counter.items():
        summary[f"rate_{category}"] = count / total if total > 0 else 0.0
        summary[f"count_{category}"] = count
    return summary


def aggregate_event(records: List[EventRecord]) -> Dict[str, float]:
    if not records:
        return {}

    def valid_values(values: List[Optional[float]]) -> List[float]:
        return [v for v in values if v is not None]

    def agg_stats(values: List[Optional[float]], key: str) -> Dict[str, float]:
        real_values = valid_values(values)
        if not real_values:
            return {f"{key}_mean": float("nan"), f"{key}_median": float("nan")}
        return {
            f"{key}_mean": float(np.mean(real_values)),
            f"{key}_median": float(np.median(real_values)),
        }

    summary = {
        "prob_src_repeat_mean": float(np.mean([rec.prob_src_repeat for rec in records])),
        "prob_src_repeat_median": float(np.median([rec.prob_src_repeat for rec in records])),
        "prob_eos_after_prompt_mean": float(np.mean([rec.prob_eos_after_prompt for rec in records])),
        "prob_eos_after_prompt_median": float(np.median([rec.prob_eos_after_prompt for rec in records])),
        "prob_bridge_after_src_mean": float(np.mean([rec.prob_bridge_after_src for rec in records])),
        "prob_bridge_after_src_median": float(np.median([rec.prob_bridge_after_src for rec in records])),
        "prob_target_direct_after_src_mean": float(np.mean([rec.prob_target_direct_after_src for rec in records])),
        "prob_target_direct_after_src_median": float(np.median([rec.prob_target_direct_after_src for rec in records])),
        "prob_eos_after_src_mean": float(np.mean([rec.prob_eos_after_src for rec in records])),
        "prob_eos_after_src_median": float(np.median([rec.prob_eos_after_src for rec in records])),
    }

    summary.update(agg_stats([rec.prob_target_after_bridge for rec in records], "prob_target_after_bridge"))
    summary.update(agg_stats([rec.prob_eos_after_bridge for rec in records], "prob_eos_after_bridge"))
    summary.update(agg_stats([rec.prob_eos_after_target for rec in records], "prob_eos_after_target"))
    summary.update(agg_stats([rec.prob_continue_after_target for rec in records], "prob_continue_after_target"))

    return summary


def aggregate_stage_events(records: List[StageEventRecord]) -> Dict[str, float]:
    if not records:
        return {
            "prob_valid_mean": float("nan"),
            "prob_bridge_given_valid_mean": float("nan"),
            "prob_causal_given_bridge_mean": float("nan"),
        }

    prob_valid = [rec.prob_valid for rec in records]
    prob_bridge_valid = [rec.prob_bridge_given_valid for rec in records]
    prob_causal_bridge = [rec.prob_causal_given_bridge for rec in records]
    chance_levels = [rec.chance_level for rec in records if rec.support_bridge > 0]

    return {
        "prob_valid_mean": float(np.mean(prob_valid)),
        "prob_valid_median": float(np.median(prob_valid)),
        "prob_bridge_given_valid_mean": float(np.mean(prob_bridge_valid)),
        "prob_bridge_given_valid_median": float(np.median(prob_bridge_valid)),
        "prob_causal_given_bridge_mean": float(np.mean(prob_causal_bridge)),
        "prob_causal_given_bridge_median": float(np.median(prob_causal_bridge)),
        "chance_level_mean": float(np.mean(chance_levels)) if chance_levels else float("nan"),
    }


# ---------------------------------------------------------------------------
# 主分析函数
# ---------------------------------------------------------------------------

def analyze_checkpoint(
    step: int,
    model: GPT,
    pairs: List[PairInfo],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    stage_sets: Dict[str, set],
    graph: nx.DiGraph,
    s3_to_valid_s2: Dict[int, set],
    stop_token_id: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    device: torch.device,
    use_tqdm: bool = False,
) -> Tuple[List[BehaviorRecord], List[EventRecord], List[StageEventRecord]]:
    records_behavior: List[BehaviorRecord] = []
    records_event: List[EventRecord] = []
    records_stage: List[StageEventRecord] = []

    iterator = enumerate(pairs)
    if use_tqdm:
        from tqdm import tqdm
        iterator = enumerate(tqdm(pairs, desc=f"step={step}", leave=False))

    for idx, info in iterator:
        digits, raw_tokens, stop_reached = run_greedy_generation(
            model=model,
            stoi=stoi,
            itos=itos,
            source=info.source,
            target=info.target,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            stop_token_id=stop_token_id,
        )

        path_nodes, base_status = build_path_from_digits(
            digits=digits,
            source=info.source,
            stop_reached=stop_reached,
        )

        category, path_len, stage2_count, target_index = classify_behavior(
            path_nodes=path_nodes,
            base_status=base_status,
            source=info.source,
            target=info.target,
            stage_sets=stage_sets,
            graph=graph,
        )

        first_action = path_nodes[1] if len(path_nodes) >= 2 else None
        records_behavior.append(BehaviorRecord(
            step=step,
            pair_index=idx,
            source=info.source,
            target=info.target,
            category=category,
            stop_reached=stop_reached,
            path_length=path_len,
            stage2_count=stage2_count or 0,
            first_action=first_action,
            target_index=target_index,
            tokens=path_nodes,
            raw_tokens=raw_tokens,
        ))

        # Logits-based events（沿用原有事件）
        prompt_ids = tokenize_context(stoi, [info.source, info.target, info.source])
        probs_prompt = get_next_token_probs(model, prompt_ids, device)

        try:
            src_token_id = stoi[str(info.source)]
            tgt_token_id = stoi[str(info.target)]
        except KeyError:
            raise KeyError("stoi 中缺少 source/target token。")

        prob_src_repeat = float(probs_prompt[src_token_id])
        prob_eos_after_prompt = float(probs_prompt[stop_token_id])

        context_after_src = prompt_ids + [src_token_id]
        probs_after_src = get_next_token_probs(model, context_after_src, device)
        bridge_token_ids = [stoi[str(b)] for b in info.bridge_candidates if str(b) in stoi]
        prob_bridge_after_src = float(sum(probs_after_src[token_id] for token_id in bridge_token_ids)) \
            if bridge_token_ids else 0.0
        prob_target_direct_after_src = float(probs_after_src[tgt_token_id])
        prob_eos_after_src = float(probs_after_src[stop_token_id])

        if info.first_stage2 is not None:
            mid_token_id = stoi[str(info.first_stage2)]
            context_after_bridge = context_after_src + [mid_token_id]
            probs_after_bridge = get_next_token_probs(model, context_after_bridge, device)
            prob_target_after_bridge = float(probs_after_bridge[tgt_token_id])
            prob_eos_after_bridge = float(probs_after_bridge[stop_token_id])

            context_after_target = context_after_bridge + [tgt_token_id]
            probs_after_target = get_next_token_probs(model, context_after_target, device)
            prob_eos_after_target = float(probs_after_target[stop_token_id])
            prob_continue_after_target = float(1.0 - prob_eos_after_target)
        else:
            prob_target_after_bridge = None
            prob_eos_after_bridge = None
            prob_eos_after_target = None
            prob_continue_after_target = None

        records_event.append(EventRecord(
            step=step,
            pair_index=idx,
            source=info.source,
            target=info.target,
            prob_src_repeat=prob_src_repeat,
            prob_eos_after_prompt=prob_eos_after_prompt,
            prob_bridge_after_src=prob_bridge_after_src,
            prob_target_direct_after_src=prob_target_direct_after_src,
            prob_eos_after_src=prob_eos_after_src,
            prob_target_after_bridge=prob_target_after_bridge,
            prob_eos_after_bridge=prob_eos_after_bridge,
            prob_eos_after_target=prob_eos_after_target,
            prob_continue_after_target=prob_continue_after_target,
        ))

        # Stage Event Factorization（新的关键分析）
        stage_record = compute_stage_event(
            probs=probs_prompt,
            src=info.source,
            tgt=info.target,
            pair_index=idx,
            stoi=stoi,
            graph=graph,
            stage_sets=stage_sets,
            s3_to_valid_s2=s3_to_valid_s2,
            step=step,
        )
        records_stage.append(stage_record)

    return records_behavior, records_event, records_stage


# ---------------------------------------------------------------------------
# CSV 写入
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, object]], field_order: Optional[List[str]] = None) -> None:
    if not rows:
        return
    if field_order is None:
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    else:
        fieldnames = field_order
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    want_cuda = args.device != "cpu"
    device = torch.device(args.device if (torch.cuda.is_available() and want_cuda) else "cpu")

    data_dir = Path(args.data_dir).resolve()
    ckpt_dir = Path(args.checkpoints_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)

    ckpt_pattern = args.ckpt_pattern or default_ckpt_pattern(args.run_type)

    stage_info_path = data_dir / "stage_info.pkl"
    meta_path = data_dir / "meta.pkl"
    test_path = data_dir / "test.txt"
    graph_path = data_dir / "composition_graph.graphml"

    stages = load_stage_info(stage_info_path)
    stage_sets = {
        "S1": set(stages[0]),
        "S2": set(stages[1]),
        "S3": set(stages[2]),
    }
    meta = load_meta(meta_path)
    stoi: Dict[str, int] = meta["stoi"]
    itos: Dict[int, str] = meta["itos"]
    vocab_size = meta["vocab_size"]
    stop_token_id = stoi["\n"]

    graph = load_graph(graph_path)
    succ_map = build_successor_map(graph)
    descendants_map = build_descendants_map(graph)

    pairs_raw = parse_test_pairs(test_path, stages)
    assign_bridge_candidates(pairs_raw, succ_map, descendants_map, stage_sets["S2"])
    pairs = maybe_subsample_pairs(pairs_raw, args.max_samples, args.sample_seed)

    s3_to_valid_s2 = precompute_reachability_map(stage_sets, descendants_map)

    if not args.quiet:
        print(f"共找到 {len(pairs_raw)} 个 S1→S3 pair，分析使用 {len(pairs)} 个。")
        print(f"checkpoint 目录: {ckpt_dir}")
        print(f"输出目录: {output_dir}")

    steps = list(range(args.step_start, args.step_end + 1, args.step_interval))

    summary_behavior_rows: List[Dict[str, object]] = []
    summary_event_rows: List[Dict[str, object]] = []
    summary_stage_rows: List[Dict[str, object]] = []

    for step in steps:
        ckpt_name = ckpt_pattern.format(step=step)
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"[警告] {ckpt_path} 不存在，跳过。")
            continue

        if not args.quiet:
            print(f"\n===== 分析 checkpoint: {ckpt_path} =====")

        model = create_model_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
            vocab_size=vocab_size,
        )

        behavior_records, event_records, stage_records = analyze_checkpoint(
            step=step,
            model=model,
            pairs=pairs,
            stoi=stoi,
            itos=itos,
            stage_sets=stage_sets,
            graph=graph,
            s3_to_valid_s2=s3_to_valid_s2,
            stop_token_id=stop_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            device=device,
            use_tqdm=args.progress,
        )

        # 聚合并记录
        behavior_summary = aggregate_behavior(behavior_records)
        behavior_summary["step"] = step
        summary_behavior_rows.append(behavior_summary)

        event_summary = aggregate_event(event_records)
        event_summary["step"] = step
        summary_event_rows.append(event_summary)

        stage_summary = aggregate_stage_events(stage_records)
        stage_summary["step"] = step
        summary_stage_rows.append(stage_summary)

        if args.save_per_pair:
            per_pair_dir = output_dir / "per_pair"
            ensure_output_dir(per_pair_dir)

            behavior_rows = []
            for rec in behavior_records:
                row = {
                    "step": rec.step,
                    "pair_index": rec.pair_index,
                    "source": rec.source,
                    "target": rec.target,
                    "category": rec.category,
                    "stop_reached": int(rec.stop_reached),
                    "path_length": rec.path_length,
                    "stage2_count": rec.stage2_count,
                    "first_action": rec.first_action if rec.first_action is not None else "",
                    "target_index": rec.target_index if rec.target_index is not None else "",
                    "tokens": " ".join(map(str, rec.tokens)),
                    "raw_tokens": " ".join(rec.raw_tokens),
                }
                behavior_rows.append(row)
            write_csv(per_pair_dir / f"behavior_step_{step}.csv", behavior_rows)

            event_rows = []
            for rec in event_records:
                row = {
                    "step": rec.step,
                    "pair_index": rec.pair_index,
                    "source": rec.source,
                    "target": rec.target,
                    "prob_src_repeat": rec.prob_src_repeat,
                    "prob_eos_after_prompt": rec.prob_eos_after_prompt,
                    "prob_bridge_after_src": rec.prob_bridge_after_src,
                    "prob_target_direct_after_src": rec.prob_target_direct_after_src,
                    "prob_eos_after_src": rec.prob_eos_after_src,
                    "prob_target_after_bridge": rec.prob_target_after_bridge if rec.prob_target_after_bridge is not None else "",
                    "prob_eos_after_bridge": rec.prob_eos_after_bridge if rec.prob_eos_after_bridge is not None else "",
                    "prob_eos_after_target": rec.prob_eos_after_target if rec.prob_eos_after_target is not None else "",
                    "prob_continue_after_target": rec.prob_continue_after_target if rec.prob_continue_after_target is not None else "",
                }
                event_rows.append(row)
            write_csv(per_pair_dir / f"events_step_{step}.csv", event_rows)

            stage_rows = []
            for rec in stage_records:
                row = {
                    "step": rec.step,
                    "pair_index": rec.pair_index,
                    "source": rec.source,
                    "target": rec.target,
                    "prob_valid": rec.prob_valid,
                    "prob_bridge_given_valid": rec.prob_bridge_given_valid,
                    "prob_causal_given_bridge": rec.prob_causal_given_bridge,
                    "support_valid": rec.support_valid,
                    "support_bridge": rec.support_bridge,
                    "support_causal": rec.support_causal,
                    "chance_level": rec.chance_level,
                }
                stage_rows.append(row)
            write_csv(per_pair_dir / f"stage_events_step_{step}.csv", stage_rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "behavior_summary.csv", summary_behavior_rows)
    write_csv(output_dir / "event_prob_summary.csv", summary_event_rows)
    write_csv(output_dir / "stage_event_summary.csv", summary_stage_rows)

    if not args.quiet:
        print("\n分析完成。")
        print(f"- 行为统计: {output_dir / 'behavior_summary.csv'}")
        print(f"- logit 事件: {output_dir / 'event_prob_summary.csv'}")
        print(f"- Stage 事件分解: {output_dir / 'stage_event_summary.csv'}")
        if args.save_per_pair:
            print(f"- per-pair 诊断: {output_dir / 'per_pair'}")


if __name__ == "__main__":
    main()