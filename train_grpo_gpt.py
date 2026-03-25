#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO (Group Relative Policy Optimization) fine-tuning script for GraphA datasets.

与 Q-learning 版本的关键区别：
  1. 每个 prompt 采样 G 条轨迹（group_size），组内归一化计算优势 Â。
  2. 使用 PPO-style clipped surrogate 目标更新策略，而非 TD + MSE。
  3. 不再需要 target network / gamma / TD error。
  4. 保留 KL 正则、温度/ε 调度、过程奖励等全部基础设施。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from logger import get_logger
from model import GPT, GPTConfig

Node = int
PathList = List[int]
Pair = Tuple[int, int]
BucketName = str


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for GraphA.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据集目录（包含 train_K.txt、meta.pkl、stage_info.pkl 等文件）。")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="SFT checkpoint (.pt)，作为 GRPO 初始化和 KL 参考。")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="对应 train_{K}.txt 中 K 的取值。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # 模型结构可选覆盖
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--block_size_override", type=int, default=None,
                        help="如需缩小 block_size，可设置此参数；不允许扩容。")

    # GRPO 核心超参
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="每次迭代的总轨迹数（= num_prompts × group_size）。")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--group_size", type=int, default=4,
                        help="每个 prompt 采样的轨迹数 G。")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="PPO-style clipping ε。")

    # 以下参数保留以兼容旧命令行，但 GRPO 不使用
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="（GRPO 不使用，保留兼容）")
    parser.add_argument("--target_ema", type=float, default=0.0,
                        help="（GRPO 不使用，保留兼容）")
    parser.add_argument("--target_sync_interval", type=int, default=0,
                        help="（GRPO 不使用，保留兼容）")

    parser.add_argument("--max_rollout_steps", type=int, default=32)
    parser.add_argument("--rollout_top_k", type=int, default=1,
                        help="rollout 采样时的 top-k；默认 1 搭配温度调度使用。")

    # rollout 温度调度
    parser.add_argument("--rollout_temperature", type=float, default=None,
                        help="若设置此参数，则温度固定为该值（兼容旧脚本）。")
    parser.add_argument("--rollout_temp_start", type=float, default=0.0)
    parser.add_argument("--rollout_temp_end", type=float, default=1.0)
    parser.add_argument("--temp_warmup_iters", type=int, default=8000)

    # ε-greedy 调度
    parser.add_argument("--epsilon_start", type=float, default=0.0)
    parser.add_argument("--epsilon_end", type=float, default=0.0)
    parser.add_argument("--epsilon_warmup_iters", type=int, default=0)

    # 无效边处理
    parser.add_argument("--allow_invalid_continue", action="store_true",
                        help="允许遇到非法边后继续生成，否则立即终止。")
    parser.add_argument("--max_invalid_transitions", type=int, default=2,
                        help="允许连续多少次非法边后强制终止。")

    # 奖励设置
    parser.add_argument("--reward_type", choices=["process", "outcome"], default="process")
    parser.add_argument("--reward_hit_target", type=float, default=1.5)
    parser.add_argument("--reward_valid_transition", type=float, default=0.1)
    parser.add_argument("--reward_stage_bridge", type=float, default=0.2)
    parser.add_argument("--reward_stage_bridge_only_once", action="store_true")
    parser.add_argument("--reward_invalid_transition", type=float, default=0.25)
    parser.add_argument("--reward_invalid_token", type=float, default=1.0)
    parser.add_argument("--reward_stop", type=float, default=-0.1)
    parser.add_argument("--penalty_stage2_detour", type=float, default=0.2)
    parser.add_argument("--penalty_stage3_detour", type=float, default=0.2)
    parser.add_argument("--penalty_repeat_node", type=float, default=0.1)
    parser.add_argument("--step_penalty", type=float, default=0.0)

    # KL 正则
    parser.add_argument("--kl_coef", type=float, default=0.05,
                        help="KL 正则系数（0 表示关闭 KL 约束）。")
    parser.add_argument("--kl_warmup_iters", type=int, default=0)
    parser.add_argument("--kl_anneal_iters", type=int, default=12000)

    # 评估 & 日志
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--max_eval_pairs", type=int, default=5000)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--eval_top_k", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="out_grpo")

    args = parser.parse_args()

    if args.rollout_temperature is not None:
        args.rollout_temp_start = args.rollout_temperature
        args.rollout_temp_end = args.rollout_temperature
        args.temp_warmup_iters = 0

    return args


# ---------------------------------------------------------------------------
# 工具 & 调度函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_temperature(iteration: int, args: argparse.Namespace) -> float:
    if args.temp_warmup_iters <= 0 or args.rollout_temp_start == args.rollout_temp_end:
        return args.rollout_temp_end
    ratio = min(1.0, iteration / max(1, args.temp_warmup_iters))
    return args.rollout_temp_start + ratio * (args.rollout_temp_end - args.rollout_temp_start)


def current_epsilon(iteration: int, args: argparse.Namespace) -> float:
    if args.epsilon_warmup_iters <= 0 or args.epsilon_start == args.epsilon_end:
        return args.epsilon_end
    ratio = min(1.0, iteration / max(1, args.epsilon_warmup_iters))
    return args.epsilon_start + ratio * (args.epsilon_end - args.epsilon_start)


def current_kl_coef(iteration: int, args: argparse.Namespace) -> float:
    if args.kl_coef <= 0.0:
        return 0.0
    if iteration <= args.kl_warmup_iters:
        return args.kl_coef
    if args.kl_anneal_iters <= 0:
        return args.kl_coef
    progress = min(1.0, (iteration - args.kl_warmup_iters) / max(1, args.kl_anneal_iters))
    return max(0.0, args.kl_coef * (1.0 - progress))


def decode_tokens(token_ids: Sequence[int],
                  itos: Dict[int, str],
                  stop_token_id: int) -> List[str]:
    tokens: List[str] = []
    for tid in token_ids:
        if tid == stop_token_id:
            break
        tokens.append(itos.get(int(tid), "[UNK]"))
    return tokens


def tokens_to_nodes(tokens: Sequence[str]) -> List[int]:
    nodes: List[int] = []
    for tok in tokens:
        if tok.isdigit():
            nodes.append(int(tok))
    return nodes


def assemble_full_path(source: int,
                       generated_nodes: Sequence[int]) -> List[int]:
    full_path = [source]
    full_path.extend(generated_nodes)
    return full_path


def bucket_for_pair(source: int,
                    target: int,
                    node_to_stage: Dict[int, int]) -> Optional[BucketName]:
    src_stage = node_to_stage.get(source)
    tgt_stage = node_to_stage.get(target)
    if src_stage is not None and tgt_stage is not None and src_stage != tgt_stage:
        return f"S{src_stage + 1}->S{tgt_stage + 1}"
    return None


def is_valid_path(path_nodes: List[int],
                  source: int,
                  target: int,
                  node_to_stage: Dict[int, int],
                  stage_sets_list: List[set],
                  graph: nx.DiGraph) -> bool:
    if len(path_nodes) < 2:
        return False
    if path_nodes[0] != source or path_nodes[-1] != target:
        return False
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            return False
    src_stage = node_to_stage.get(source)
    tgt_stage = node_to_stage.get(target)
    if src_stage is not None and tgt_stage is not None:
        min_s = min(src_stage, tgt_stage)
        max_s = max(src_stage, tgt_stage)
        if max_s - min_s > 1:
            for mid in range(min_s + 1, max_s):
                if not any(node in stage_sets_list[mid] for node in path_nodes[1:-1]):
                    return False
    return True


def load_pairs(train_file: Path) -> List[Pair]:
    seen: set[Pair] = set()
    pairs: List[Pair] = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            s, t = int(parts[0]), int(parts[1])
            key = (s, t)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


def prepare_output_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / f"grpo_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_max_new_tokens(block_size: int,
                        prompt_len: int,
                        desired: int) -> int:
    available = block_size - prompt_len - 1
    if available <= 0:
        raise ValueError(
            f"Block size {block_size} is too small for prompt length {prompt_len}."
        )
    return max(1, min(desired, available))


def load_state_dict_with_block_resize(model: GPT,
                                      raw_state_dict: Dict[str, Tensor],
                                      ckpt_block_size: int,
                                      logger) -> None:
    state_dict = dict(raw_state_dict)
    model_block_size = model.config.block_size
    if ckpt_block_size is None:
        ckpt_block_size = model_block_size

    if model_block_size == ckpt_block_size:
        model.load_state_dict(state_dict, strict=True)
        return

    if model_block_size < ckpt_block_size:
        raise ValueError(
            f"Checkpoint block_size={ckpt_block_size} 大于当前模型 block_size={model_block_size}。"
        )

    logger.warning(
        "扩容 block_size：checkpoint=%d -> 当前模型=%d。新位置的嵌入使用随机初始化。",
        ckpt_block_size, model_block_size,
    )

    wpe_key = "transformer.wpe.weight"
    if wpe_key in state_dict:
        old_weight = state_dict[wpe_key]
        if old_weight.shape[0] != ckpt_block_size:
            logger.warning("checkpoint 的 wpe.weight 行数 (%d) 与记录的 block_size (%d) 不一致。",
                           old_weight.shape[0], ckpt_block_size)
        new_weight = model.transformer.wpe.weight.detach().clone()
        new_weight[:old_weight.size(0)] = old_weight
        state_dict[wpe_key] = new_weight

    bias_like = [
        key for key in state_dict.keys()
        if key.endswith("attn.bias") or key.endswith("attn.mask")
    ]
    for key in bias_like:
        state_dict.pop(key)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    allowed_missing = {key for key in missing_keys
                       if key.endswith("attn.bias") or key.endswith("attn.mask")}
    leftover_missing = [k for k in missing_keys if k not in allowed_missing]
    if leftover_missing:
        logger.warning("加载 checkpoint 时缺失的键：%s", leftover_missing)
    if unexpected_keys:
        logger.warning("加载 checkpoint 时出现未识别的键：%s", unexpected_keys)


# ---------------------------------------------------------------------------
# 评估（与 Q-learning 版完全一致）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model: GPT,
                   pairs: List[Pair],
                   node_to_stage: Dict[int, int],
                   stage_sets_list: List[set],
                   stoi: Dict[str, int],
                   itos: Dict[int, str],
                   graph: nx.DiGraph,
                   device: torch.device,
                   temperature: float,
                   top_k: int,
                   max_steps: int,
                   max_pairs: int = 5000) -> Dict[str, Dict[str, float]]:
    model.eval()
    stop_token_id = stoi["\n"]
    block_size = model.config.block_size

    buckets: Dict[BucketName, List[Pair]] = defaultdict(list)
    for s, t in pairs[:max_pairs]:
        bucket = bucket_for_pair(s, t, node_to_stage)
        if bucket:
            buckets[bucket].append((s, t))

    results: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    total_cases = 0

    for bucket_name, bucket_pairs in buckets.items():
        correct = 0
        for source, target in bucket_pairs:
            prompt_tokens = [
                stoi[str(source)],
                stoi[str(target)],
                stoi[str(source)],
            ]
            max_new_tokens = safe_max_new_tokens(block_size, len(prompt_tokens), max_steps)
            x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

            generated = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
            )[0].tolist()

            new_tokens = generated[len(prompt_tokens):]
            decoded_tokens = decode_tokens(new_tokens, itos, stop_token_id)
            generated_nodes = tokens_to_nodes(decoded_tokens)
            full_path = assemble_full_path(source, generated_nodes)

            if is_valid_path(full_path, source, target, node_to_stage, stage_sets_list, graph):
                correct += 1

        total_correct += correct
        total_cases += len(bucket_pairs)
        acc = correct / len(bucket_pairs) if bucket_pairs else 0.0
        results[bucket_name] = {
            "correct": correct,
            "total": len(bucket_pairs),
            "accuracy": acc,
        }

    overall_acc = total_correct / total_cases if total_cases else 0.0
    results["overall"] = {
        "correct": total_correct,
        "total": total_cases,
        "accuracy": overall_acc,
    }

    model.train()
    return results


# ---------------------------------------------------------------------------
# GRPO 核心：序列 log-prob 计算
# ---------------------------------------------------------------------------
def compute_sequence_logprobs(model: GPT,
                              traj_ids: List[int],
                              action_start: int,
                              device: torch.device) -> Tensor:
    """
    返回 traj_ids[action_start:] 中每个 token 在模型下的 log π(a_t | context)。
    返回 shape = (num_actions,)，保留梯度（除非外部包了 no_grad）。
    """
    num_actions = len(traj_ids) - action_start
    if num_actions <= 0:
        return torch.tensor([], device=device, dtype=torch.float32)

    x = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(traj_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x, y)
    logits = logits.squeeze(0)                       # (seq_len, vocab)

    start_idx = action_start - 1                     # logits[start_idx] 预测第一个 action
    action_logits = logits[start_idx:start_idx + num_actions]
    log_probs = F.log_softmax(action_logits, dim=-1)

    action_ids = torch.tensor(
        traj_ids[action_start:action_start + num_actions],
        dtype=torch.long, device=device,
    )
    return log_probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout 相关（与 Q-learning 版完全一致）
# ---------------------------------------------------------------------------
def select_next_token(logits: Tensor,
                      temperature: float,
                      top_k: int,
                      epsilon: float) -> int:
    logits = logits.detach().clone()
    vocab_size = logits.size(-1)
    topk_indices = None

    if top_k > 0 and top_k < vocab_size:
        top_vals, topk_indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, topk_indices, top_vals)
        logits = mask

    if epsilon > 0.0 and random.random() < epsilon:
        if topk_indices is not None:
            idx = topk_indices[torch.randint(0, topk_indices.size(0), (1,), device=logits.device)]
        else:
            idx = torch.randint(0, vocab_size, (1,), device=logits.device)
        return int(idx.item())

    if temperature <= 1e-6:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def build_prompt(source: int,
                 target: int,
                 stoi: Dict[str, int]) -> List[int]:
    return [
        stoi[str(source)],
        stoi[str(target)],
        stoi[str(source)],
    ]


def build_traj_ids(prompt_ids: List[int],
                   sampled_ids: List[int],
                   stop_token_id: int,
                   block_size: int) -> List[int]:
    traj = prompt_ids + sampled_ids
    if len(traj) >= block_size:
        traj = traj[:block_size - 1]
    if not sampled_ids or sampled_ids[-1] != stop_token_id:
        if len(traj) < block_size:
            traj.append(stop_token_id)
    return traj


def sample_trajectory(model: GPT,
                      source: int,
                      target: int,
                      prompt_ids: List[int],
                      max_new_tokens: int,
                      args: argparse.Namespace,
                      stoi: Dict[str, int],
                      itos: Dict[int, str],
                      graph: nx.DiGraph,
                      node_to_stage: Dict[int, int],
                      stage_sets_list: List[set],
                      pair_bucket: Optional[str],
                      stop_token_id: int,
                      device: torch.device,
                      block_size: int,
                      temperature: float,
                      epsilon: float) -> Dict[str, object]:
    model_was_training = model.training
    if model_was_training:
        model.eval()

    sampled_ids: List[int] = []
    decoded_tokens: List[str] = []
    rewards: List[float] = []
    dones: List[bool] = []

    traj_ids = prompt_ids.copy()
    current_node = source
    hit_target = False
    invalid_transition = False
    invalid_token = False
    visited_intermediate = False
    stage_bridge_rewarded = False
    valid_transition_steps = 0
    invalid_transition_steps = 0
    visited_nodes = {source}

    allow_continue = args.allow_invalid_continue
    max_invalid = max(1, args.max_invalid_transitions) if allow_continue else 1

    src_stage_idx = node_to_stage.get(source)
    tgt_stage_idx = node_to_stage.get(target)
    is_composition = False
    intermediate_stage_indices: List[int] = []
    target_stage_set: set = set()
    penalty_target_detour = max(args.penalty_stage2_detour, args.penalty_stage3_detour)

    if src_stage_idx is not None and tgt_stage_idx is not None:
        gap = abs(tgt_stage_idx - src_stage_idx)
        is_composition = gap > 1
        min_s = min(src_stage_idx, tgt_stage_idx)
        max_s = max(src_stage_idx, tgt_stage_idx)
        intermediate_stage_indices = list(range(min_s + 1, max_s))
        target_stage_set = stage_sets_list[tgt_stage_idx]

    for step in range(max_new_tokens):
        if len(traj_ids) >= block_size - 1:
            break

        context = torch.tensor(traj_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(context)
        step_logits = logits[0, -1, :]

        next_token_id = select_next_token(
            logits=step_logits,
            temperature=temperature,
            top_k=args.rollout_top_k,
            epsilon=epsilon,
        )

        traj_ids.append(next_token_id)
        sampled_ids.append(next_token_id)

        token_str = itos.get(int(next_token_id), "[UNK]")
        decoded_tokens.append(token_str)

        reward = -args.step_penalty if args.step_penalty != 0.0 else 0.0
        done = False

        if next_token_id == stop_token_id:
            reward += args.reward_stop
            done = True
        elif token_str.isdigit():
            next_node = int(token_str)
            adjacency = graph.has_edge(str(current_node), str(next_node))

            if adjacency:
                valid_transition_steps += 1
                reward += args.reward_valid_transition

                if intermediate_stage_indices and any(
                    next_node in stage_sets_list[mid]
                    for mid in intermediate_stage_indices
                ):
                    visited_intermediate = True

                if is_composition and args.reward_stage_bridge > 0.0:
                    if not stage_bridge_rewarded and any(
                        next_node in stage_sets_list[mid]
                        for mid in intermediate_stage_indices
                    ):
                        reward += args.reward_stage_bridge
                        if args.reward_stage_bridge_only_once:
                            stage_bridge_rewarded = True

                if not is_composition and penalty_target_detour > 0.0:
                    if next_node != target and next_node in target_stage_set:
                        reward -= penalty_target_detour

                if args.penalty_repeat_node > 0.0 and next_node in visited_nodes and next_node != target:
                    reward -= args.penalty_repeat_node
                visited_nodes.add(next_node)

                current_node = next_node
                if next_node == target:
                    reward += args.reward_hit_target
                    hit_target = True
                    done = True
            else:
                invalid_transition = True
                invalid_transition_steps += 1
                reward -= args.reward_invalid_transition
                if not allow_continue or invalid_transition_steps >= max_invalid:
                    done = True
        else:
            invalid_token = True
            reward -= args.reward_invalid_token
            done = True

        rewards.append(float(reward))
        dones.append(bool(done))

        if done:
            break

    if model_was_training:
        model.train()

    if not sampled_ids or sampled_ids[-1] != stop_token_id:
        sampled_ids.append(stop_token_id)
        extra_reward = (-args.step_penalty if args.step_penalty != 0.0 else 0.0) + args.reward_stop
        rewards.append(float(extra_reward))
        dones.append(True)

    traj_ids = build_traj_ids(prompt_ids, sampled_ids, stop_token_id, block_size)
    decoded_tokens = decode_tokens(sampled_ids, itos, stop_token_id)
    generated_nodes = tokens_to_nodes(decoded_tokens)
    full_path_nodes = assemble_full_path(source, generated_nodes)

    first_step_node = generated_nodes[0] if generated_nodes else None
    first_step_is_source = first_step_node == source
    first_step_is_valid = bool(first_step_node is not None and
                               graph.has_edge(str(source), str(first_step_node)))

    success = is_valid_path(full_path_nodes, source, target, node_to_stage, stage_sets_list, graph)

    if args.reward_type == "outcome":
        adjusted_rewards = [0.0 for _ in rewards]
        if success and hit_target:
            target_token = str(target)
            for idx, token_id in enumerate(sampled_ids):
                token_str = itos.get(token_id, "[UNK]")
                if token_str == target_token:
                    adjusted_rewards[idx] = args.reward_hit_target
                    break
        rewards = adjusted_rewards

    episode_reward = float(sum(rewards))
    step_reward_mean = float(np.mean(rewards)) if rewards else 0.0

    return {
        "traj_ids": traj_ids,
        "actions": sampled_ids,
        "rewards": rewards,
        "dones": dones,
        "decoded_tokens": decoded_tokens,
        "generated_nodes": generated_nodes,
        "path_nodes": full_path_nodes,
        "success": success,
        "first_step_is_source": first_step_is_source,
        "first_step_is_valid": first_step_is_valid,
        "episode_reward": episode_reward,
        "step_reward_mean": step_reward_mean,
        "hit_target": hit_target and success,
        "invalid_transition": invalid_transition,
        "invalid_token": invalid_token,
        "visited_intermediate": visited_intermediate or stage_bridge_rewarded,
        "valid_transition_steps": valid_transition_steps,
        "invalid_transition_steps": invalid_transition_steps,
        "bucket": pair_bucket,
    }


# ---------------------------------------------------------------------------
# 主训练逻辑
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_dir = Path(args.data_dir).resolve()
    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"

    if not train_txt.exists():
        raise FileNotFoundError(f"Training text file not found: {train_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Test text file not found: {test_txt}")

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi: Dict[str, int] = meta["stoi"]
    itos: Dict[int, str] = meta["itos"]
    vocab_size = meta["vocab_size"]
    dataset_block_size = meta["block_size"]

    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages: List[List[int]] = stage_info["stages"]
    num_stages = len(stages)

    stage_sets_list: List[set] = [set(s) for s in stages]
    node_to_stage: Dict[int, int] = {}
    for idx, stage_nodes in enumerate(stages):
        for node in stage_nodes:
            node_to_stage[node] = idx

    graph = nx.read_graphml(data_dir / "composition_graph.graphml")

    out_dir = prepare_output_dir(args.log_dir)
    logger = get_logger(os.path.join(out_dir, "train_grpo.log"))
    logger.info("GRPO training started.")
    logger.info("Output directory: %s", out_dir)
    logger.info("Algorithm: GRPO | group_size=%d | clip_eps=%.3f", args.group_size, args.clip_eps)
    logger.info("Reward type: %s", args.reward_type)
    logger.info("Number of stages: %d", num_stages)
    for i, s in enumerate(stages):
        logger.info("  Stage S%d: %d nodes", i + 1, len(s))

    # 兼容性提示
    if args.gamma != 1.0:
        logger.info("注意：--gamma=%.4f 在 GRPO 中不使用（保留兼容）。", args.gamma)
    if args.target_ema != 0.0 or args.target_sync_interval != 0:
        logger.info("注意：--target_ema / --target_sync_interval 在 GRPO 中不使用（保留兼容）。")

    ckpt = torch.load(args.sft_checkpoint, map_location="cpu")
    ckpt_model_args = ckpt.get("model_args", {})

    def resolve_numeric(attr_name: str, ckpt_key: str, default_value):
        cli_value = getattr(args, attr_name, None)
        if cli_value is not None:
            return cli_value
        if ckpt_model_args and ckpt_key in ckpt_model_args:
            return ckpt_model_args[ckpt_key]
        return default_value

    def resolve_bias(default_value: bool) -> bool:
        if args.bias is not None:
            return args.bias.lower() == "true"
        if ckpt_model_args and "bias" in ckpt_model_args:
            return bool(ckpt_model_args["bias"])
        return default_value

    resolved_block_size = dataset_block_size
    if args.block_size_override is not None:
        if args.block_size_override > dataset_block_size:
            raise ValueError(
                f"不允许将 block_size 扩容（override={args.block_size_override}, "
                f"dataset={dataset_block_size})。"
            )
        resolved_block_size = args.block_size_override
    elif ckpt_model_args and "block_size" in ckpt_model_args:
        resolved_block_size = int(ckpt_model_args["block_size"])

    model_args = dict(
        vocab_size=vocab_size,
        block_size=resolved_block_size,
        n_layer=resolve_numeric("n_layer", "n_layer", 1),
        n_head=resolve_numeric("n_head", "n_head", 1),
        n_embd=resolve_numeric("n_embd", "n_embd", 120),
        dropout=resolve_numeric("dropout", "dropout", 0.0),
        bias=resolve_bias(False),
    )

    logger.info("Resolved model configuration: %s", json.dumps(model_args))

    ckpt_block_size = ckpt_model_args.get("block_size", dataset_block_size)

    # ---- 策略模型 ----
    model = GPT(GPTConfig(**model_args)).to(device)
    load_state_dict_with_block_resize(
        model=model,
        raw_state_dict=ckpt["model"],
        ckpt_block_size=ckpt_block_size,
        logger=logger,
    )

    # ---- SFT 参考模型（用于 KL 正则） ----
    if args.kl_coef > 0.0:
        sft_ref_model = GPT(GPTConfig(**model_args)).to(device)
        load_state_dict_with_block_resize(
            model=sft_ref_model,
            raw_state_dict=ckpt["model"],
            ckpt_block_size=ckpt_block_size,
            logger=logger,
        )
        sft_ref_model.eval()
        for p in sft_ref_model.parameters():
            p.requires_grad = False
        logger.info("KL 正则启用：coef=%.4f, warmup=%d, anneal=%d",
                    args.kl_coef, args.kl_warmup_iters, args.kl_anneal_iters)
    else:
        sft_ref_model = None
        logger.info("KL 正则禁用。")

    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    train_pairs = load_pairs(train_txt)
    logger.info("Loaded %d unique (source, target) pairs for GRPO training.", len(train_pairs))

    eval_pairs: List[Pair] = []
    with open(test_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                eval_pairs.append((int(parts[0]), int(parts[1])))

    stop_token_id = stoi["\n"]
    prompt_len = 3
    action_start = prompt_len
    metrics_path = out_dir / "metrics_grpo.jsonl"

    rollout_cap = safe_max_new_tokens(model.config.block_size, prompt_len, args.max_rollout_steps)
    if rollout_cap < args.max_rollout_steps:
        logger.warning("max_rollout_steps=%d 超出 block_size=%d，实际截断为 %d。",
                       args.max_rollout_steps, model.config.block_size, rollout_cap)

    all_bucket_names_set: set = set()
    for s, t in train_pairs:
        b = bucket_for_pair(s, t, node_to_stage)
        if b:
            all_bucket_names_set.add(b)
    all_bucket_names = sorted(all_bucket_names_set, key=lambda k: (
        int(k.split("->")[0].replace("S", "")),
        int(k.split("->")[1].replace("S", "")),
    ))
    logger.info("Detected training buckets: %s", all_bucket_names)

    G = max(1, args.group_size)
    num_prompts_per_iter = max(1, args.batch_size // G)
    logger.info("每次迭代采样 %d 个 prompt × %d 条轨迹 = %d 总轨迹",
                num_prompts_per_iter, G, num_prompts_per_iter * G)

    model.train()

    # =======================================================================
    # 训练主循环
    # =======================================================================
    for iteration in range(1, args.max_iters + 1):
        temperature = current_temperature(iteration, args)
        epsilon = current_epsilon(iteration, args)
        kl_coef_now = current_kl_coef(iteration, args)

        prompt_pairs = random.choices(train_pairs, k=num_prompts_per_iter)

        # ------ Phase 1: 采样轨迹 + 计算 old log-probs ------
        all_groups: List[List[Dict]] = []
        all_trajs: List[Dict] = []

        model.eval()

        for source, target in prompt_pairs:
            prompt_ids = build_prompt(source, target, stoi)
            bucket = bucket_for_pair(source, target, node_to_stage)

            group: List[Dict] = []
            for _ in range(G):
                traj_info = sample_trajectory(
                    model=model,
                    source=source,
                    target=target,
                    prompt_ids=prompt_ids,
                    max_new_tokens=rollout_cap,
                    args=args,
                    stoi=stoi,
                    itos=itos,
                    graph=graph,
                    node_to_stage=node_to_stage,
                    stage_sets_list=stage_sets_list,
                    pair_bucket=bucket,
                    stop_token_id=stop_token_id,
                    device=device,
                    block_size=model.config.block_size,
                    temperature=temperature,
                    epsilon=epsilon,
                )

                num_act = len(traj_info["traj_ids"]) - action_start
                if num_act > 0:
                    with torch.no_grad():
                        old_lp = compute_sequence_logprobs(
                            model, traj_info["traj_ids"], action_start, device)
                else:
                    old_lp = torch.tensor([], device=device, dtype=torch.float32)
                traj_info["old_logprobs"] = old_lp
                traj_info["num_actions"] = num_act

                group.append(traj_info)
                all_trajs.append(traj_info)

            all_groups.append(group)

        # ------ Phase 2: 计算组内归一化优势 ------
        for group in all_groups:
            ep_rewards = [t["episode_reward"] for t in group]
            mean_r = float(np.mean(ep_rewards))
            std_r = float(np.std(ep_rewards))
            for t in group:
                if std_r < 1e-8:
                    t["advantage"] = 0.0
                else:
                    t["advantage"] = (t["episode_reward"] - mean_r) / std_r

        # ------ Phase 3: 计算 GRPO 损失 ------
        model.train()

        policy_losses: List[Tensor] = []
        kl_losses_list: List[Tensor] = []
        ratio_list: List[float] = []
        clip_fractions: List[float] = []

        # 收集统计信息
        episode_rewards: List[float] = []
        step_rewards: List[float] = []
        path_lengths: List[int] = []
        valid_transition_list: List[int] = []
        invalid_transition_steps_list: List[int] = []
        first_step_source_list: List[float] = []
        first_step_valid_list: List[float] = []
        advantages_abs: List[float] = []
        group_reward_stds: List[float] = []

        bucket_success_sum = defaultdict(float)
        bucket_counts = defaultdict(int)
        success_count = 0
        hit_target_count = 0
        invalid_transition_count = 0
        invalid_token_count = 0
        intermediate_visit_count = 0

        for group in all_groups:
            ep_rews = [t["episode_reward"] for t in group]
            group_reward_stds.append(float(np.std(ep_rews)))

        for traj in all_trajs:
            # 统计
            episode_rewards.append(traj["episode_reward"])
            step_rewards.append(traj["step_reward_mean"])
            path_lengths.append(len(traj["path_nodes"]))
            valid_transition_list.append(traj["valid_transition_steps"])
            invalid_transition_steps_list.append(traj["invalid_transition_steps"])
            first_step_source_list.append(1.0 if traj["first_step_is_source"] else 0.0)
            first_step_valid_list.append(1.0 if traj["first_step_is_valid"] else 0.0)

            succ = bool(traj["success"])
            success_count += int(succ)
            hit_target_count += int(traj["hit_target"])
            invalid_transition_count += int(traj["invalid_transition"])
            invalid_token_count += int(traj["invalid_token"])
            intermediate_visit_count += int(traj.get("visited_intermediate", False))

            bname = traj["bucket"]
            if bname:
                bucket_success_sum[bname] += 1.0 if succ else 0.0
                bucket_counts[bname] += 1

            # GRPO 策略损失
            num_act = traj["num_actions"]
            advantage = traj["advantage"]
            advantages_abs.append(abs(advantage))

            if num_act <= 0:
                continue

            new_logprobs = compute_sequence_logprobs(
                model, traj["traj_ids"], action_start, device)
            old_logprobs = traj["old_logprobs"].detach()

            # 安全截断对齐
            act_len = min(new_logprobs.size(0), old_logprobs.size(0))
            if act_len <= 0:
                continue
            new_logprobs = new_logprobs[:act_len]
            old_logprobs = old_logprobs[:act_len]

            ratio = torch.exp(new_logprobs - old_logprobs)
            adv = torch.tensor(advantage, dtype=torch.float32, device=device)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            per_token_loss = -torch.min(surr1, surr2)
            policy_losses.append(per_token_loss.mean())

            with torch.no_grad():
                ratio_list.append(float(ratio.mean().item()))
                clipped = ((ratio < 1.0 - args.clip_eps) | (ratio > 1.0 + args.clip_eps)).float()
                clip_fractions.append(float(clipped.mean().item()))

            # KL 正则
            if kl_coef_now > 0 and sft_ref_model is not None:
                with torch.no_grad():
                    ref_logprobs = compute_sequence_logprobs(
                        sft_ref_model, traj["traj_ids"], action_start, device)
                    ref_logprobs = ref_logprobs[:act_len]
                kl_per_token = new_logprobs - ref_logprobs
                kl_losses_list.append(kl_per_token.mean())

        # ------ 反向传播 ------
        if not policy_losses:
            logger.warning("Iter %d 无有效样本，跳过更新。", iteration)
            continue

        mean_policy_loss = torch.stack(policy_losses).mean()

        if kl_losses_list and kl_coef_now > 0:
            mean_kl = torch.stack(kl_losses_list).mean()
            total_loss = mean_policy_loss + kl_coef_now * mean_kl
        else:
            mean_kl = torch.tensor(0.0, device=device)
            total_loss = mean_policy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ------ 日志 ------
        total_trajs = len(all_trajs)
        avg_ep_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        avg_step_reward = float(np.mean(step_rewards)) if step_rewards else 0.0
        avg_path_len = float(np.mean(path_lengths)) if path_lengths else 0.0
        avg_valid_tr = float(np.mean(valid_transition_list)) if valid_transition_list else 0.0
        avg_inv_tr = float(np.mean(invalid_transition_steps_list)) if invalid_transition_steps_list else 0.0
        first_src_rate = float(np.mean(first_step_source_list)) if first_step_source_list else 0.0
        first_valid_rate = float(np.mean(first_step_valid_list)) if first_step_valid_list else 0.0
        success_rate = success_count / total_trajs if total_trajs else 0.0
        hit_rate = hit_target_count / total_trajs if total_trajs else 0.0
        inv_edge_rate = invalid_transition_count / total_trajs if total_trajs else 0.0
        inv_tok_rate = invalid_token_count / total_trajs if total_trajs else 0.0
        intermed_rate = intermediate_visit_count / total_trajs if total_trajs else 0.0
        mean_adv_abs = float(np.mean(advantages_abs)) if advantages_abs else 0.0
        mean_grp_std = float(np.mean(group_reward_stds)) if group_reward_stds else 0.0
        mean_ratio_val = float(np.mean(ratio_list)) if ratio_list else 1.0
        mean_clip_frac = float(np.mean(clip_fractions)) if clip_fractions else 0.0
        kl_val = float(mean_kl.detach().item()) if isinstance(mean_kl, Tensor) else 0.0

        if iteration % 50 == 0:
            logger.info(
                "Iter %6d | loss=%.4f | policy=%.4f | kl=%.4f | temp=%.3f | eps=%.3f | "
                "success=%.3f | hit=%.3f | intermed=%.3f | first_src=%.3f | first_valid=%.3f | "
                "inv_edge=%.3f | inv_tok=%.3f | ep_rew=%.3f | path=%.2f | "
                "adv_abs=%.3f | grp_std=%.3f | ratio=%.3f | clip=%.3f",
                iteration,
                float(total_loss.item()),
                float(mean_policy_loss.item()),
                kl_val,
                temperature,
                epsilon,
                success_rate,
                hit_rate,
                intermed_rate,
                first_src_rate,
                first_valid_rate,
                inv_edge_rate,
                inv_tok_rate,
                avg_ep_reward,
                avg_path_len,
                mean_adv_abs,
                mean_grp_std,
                mean_ratio_val,
                mean_clip_frac,
            )

        record = {
            "iter": iteration,
            "loss": float(total_loss.item()),
            "policy_loss": float(mean_policy_loss.item()),
            "kl_loss": kl_val,
            "kl_coef_current": kl_coef_now,
            "temperature": temperature,
            "epsilon": epsilon,
            "success_rate": success_rate,
            "hit_rate": hit_rate,
            "intermediate_visit_rate": intermed_rate,
            "first_step_source_rate": first_src_rate,
            "first_step_valid_rate": first_valid_rate,
            "invalid_transition_rate": inv_edge_rate,
            "invalid_token_rate": inv_tok_rate,
            "avg_episode_reward": avg_ep_reward,
            "avg_step_reward": avg_step_reward,
            "avg_path_len": avg_path_len,
            "avg_valid_transitions": avg_valid_tr,
            "avg_invalid_transition_steps": avg_inv_tr,
            "mean_advantage_abs": mean_adv_abs,
            "mean_group_reward_std": mean_grp_std,
            "mean_ratio": mean_ratio_val,
            "clip_fraction": mean_clip_frac,
        }
        for bname in all_bucket_names:
            cnt = bucket_counts.get(bname, 0)
            record[f"train_success/{bname}"] = (
                bucket_success_sum.get(bname, 0.0) / cnt if cnt > 0 else 0.0
            )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # ------ 评估 ------
        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            eval_results = evaluate_model(
                model=model,
                pairs=eval_pairs,
                node_to_stage=node_to_stage,
                stage_sets_list=stage_sets_list,
                stoi=stoi,
                itos=itos,
                graph=graph,
                device=device,
                temperature=args.eval_temperature,
                top_k=args.eval_top_k,
                max_steps=args.max_rollout_steps,
                max_pairs=args.max_eval_pairs,
            )
            logger.info("---- Evaluation at iter %d ----", iteration)
            sorted_eval_keys = sorted(
                [k for k in eval_results if k != "overall"],
                key=lambda k: (
                    int(k.split("->")[0].replace("S", "")),
                    int(k.split("->")[1].replace("S", "")),
                ),
            )
            sorted_eval_keys.append("overall")
            for key in sorted_eval_keys:
                stats = eval_results[key]
                logger.info(
                    "  %s: %.2f%% (%d / %d)",
                    key,
                    stats["accuracy"] * 100.0,
                    stats["correct"],
                    stats["total"],
                )
            eval_record = {"iter": iteration, "eval": eval_results}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record) + "\n")

        # ------ 保存 ------
        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_path = out_dir / f"ckpt_grpo_{iteration}.pt"
            torch.save(
                {
                    "iter_num": iteration,
                    "model": model.state_dict(),
                    "model_args": model_args,
                    "config": vars(args),
                },
                ckpt_path,
            )
            logger.info("Saved GRPO checkpoint to %s", ckpt_path)

    logger.info("GRPO training finished.")


if __name__ == "__main__":
    main()