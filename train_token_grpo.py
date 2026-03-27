#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level GRPO / PPO fine-tuning for GraphA strict atomic-only setting.

相对旧版 sequence-level GRPO 的关键改动：
  [NEW-1] token-level reward-to-go + group-relative advantage
  [NEW-2] 按 position 做 group normalization（默认），更适合多步规划
  [NEW-3] 明确“到 target 后必须 stop”的训练语义
  [NEW-4] exact-stop evaluation（可切回旧版宽松评估）
  [NEW-5] SFT CE anchor / PTX anchor，防止遗忘与漂移
  [NEW-6] full-distribution KL + KL floor
  [NEW-7] 熵正则，防止策略过早塌缩
  [NEW-8] 自动保存 best checkpoint（默认按 S1->S3）
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
Pair = Tuple[int, int]
BucketName = str


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-level GRPO/PPO fine-tuning for GraphA (strict atomic-only)."
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据集目录（包含 train_K.txt、test.txt、meta.pkl、stage_info.pkl 等）。")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="SFT checkpoint (.pt)，作为初始化与 KL reference。")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="训练文件 train_{K}.txt 中 K 的值。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # 模型结构（可选覆盖）
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--block_size_override", type=int, default=None,
                        help="若需要缩小 block_size 可设置；不允许扩容。")

    # 训练主超参
    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="每次迭代的总轨迹数，必须能被 group_size 整除。")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # GRPO / PPO
    parser.add_argument("--group_size", type=int, default=8,
                        help="每个 prompt 采样多少条轨迹。")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="PPO clip epsilon。")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="每批 rollout 上重复优化轮数。")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="token-level return 的折扣因子。")

    # token-level advantage
    parser.add_argument("--advantage_mode", choices=["position", "global"], default="position",
                        help="group-relative advantage 的归一化方式。推荐 position。")
    parser.add_argument("--advantage_std_floor", type=float, default=0.25,
                        help="group std 的下界，避免小 std 放大噪声。")
    parser.add_argument("--advantage_clip", type=float, default=4.0,
                        help="token advantage 裁剪范围。")
    parser.add_argument("--use_leave_one_out", type=int, default=1, choices=[0, 1],
                        help="是否使用 leave-one-out baseline。推荐 1。")

    # rollout
    parser.add_argument("--max_rollout_steps", type=int, default=20)
    parser.add_argument("--rollout_top_k", type=int, default=0,
                        help="rollout 采样 top-k；0 表示不截断。推荐 0。")

    # 温度
    parser.add_argument("--rollout_temperature", type=float, default=None,
                        help="若设置，则 rollout 温度固定为该值。")
    parser.add_argument("--rollout_temp_start", type=float, default=0.9)
    parser.add_argument("--rollout_temp_end", type=float, default=0.6)
    parser.add_argument("--temp_warmup_iters", type=int, default=4000)

    # epsilon-greedy（默认关闭）
    parser.add_argument("--epsilon_start", type=float, default=0.0)
    parser.add_argument("--epsilon_end", type=float, default=0.0)
    parser.add_argument("--epsilon_warmup_iters", type=int, default=0)

    # 非法边处理
    parser.add_argument("--allow_invalid_continue", action="store_true",
                        help="允许非法边后继续生成，否则非法边后立刻终止。")
    parser.add_argument("--max_invalid_transitions", type=int, default=2,
                        help="若允许继续，最多连续多少次非法转移。")

    # reward type
    parser.add_argument("--reward_type", choices=["process", "outcome"], default="process")

    # shaping / terminal rewards
    parser.add_argument("--reward_hit_target", type=float, default=1.0,
                        help="采到 target 节点的奖励。")
    parser.add_argument("--reward_stop", type=float, default=0.0,
                        help="stop token 的基础奖励（通常设 0）。")
    parser.add_argument("--reward_stop_on_target", type=float, default=0.5,
                        help="在 target 后紧接 stop 的奖励。")
    parser.add_argument("--reward_valid_transition", type=float, default=0.05,
                        help="合法边的小奖励。")
    parser.add_argument("--reward_distance_progress", type=float, default=0.15,
                        help="按 shortest-path 到 target 的距离改善给 shaping。")
    parser.add_argument("--reward_stage_bridge", type=float, default=0.0,
                        help="跨中间 stage 的奖励（strict atomic-only 里通常不会触发）。")
    parser.add_argument("--reward_stage_bridge_only_once", action="store_true")

    # penalty magnitudes（都为正数，内部会减掉）
    parser.add_argument("--reward_invalid_transition", type=float, default=0.6,
                        help="非法边惩罚的幅度。")
    parser.add_argument("--reward_invalid_token", type=float, default=1.0,
                        help="非法 token 惩罚的幅度。")
    parser.add_argument("--penalty_stop_before_target", type=float, default=0.5,
                        help="尚未到 target 就 stop 的惩罚。")
    parser.add_argument("--penalty_overshoot_target", type=float, default=1.0,
                        help="到 target 后没有 stop 而继续生成的惩罚。")
    parser.add_argument("--penalty_missing_stop", type=float, default=0.5,
                        help="到了 target 但直到 rollout 结束都没 stop 的惩罚。")
    parser.add_argument("--penalty_miss_target_final", type=float, default=0.0,
                        help="整条轨迹始终未命中 target 时，在最后一步追加的终局惩罚。")
    parser.add_argument("--penalty_stage2_detour", type=float, default=0.2,
                        help="gap=1 时进入目标 stage 但不是目标节点的 detour 惩罚。")
    parser.add_argument("--penalty_stage3_detour", type=float, default=0.2)
    parser.add_argument("--penalty_repeat_node", type=float, default=0.15)
    parser.add_argument("--step_penalty", type=float, default=0.02)

    # KL
    parser.add_argument("--kl_coef", type=float, default=0.20,
                        help="KL 初始系数。")
    parser.add_argument("--kl_warmup_iters", type=int, default=0)
    parser.add_argument("--kl_anneal_iters", type=int, default=6000)
    parser.add_argument("--kl_min_coef", type=float, default=0.10,
                        help="KL 系数下界。")

    # CE / PTX anchor
    parser.add_argument("--sft_ce_coef", type=float, default=0.10,
                        help="混入 SFT CE anchor 的权重。推荐 > 0。")
    parser.add_argument("--sft_ce_batch_size", type=int, default=64)

    # entropy bonus
    parser.add_argument("--entropy_coef", type=float, default=1e-3,
                        help="token entropy bonus 系数。")

    # eval / logging / saving
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_eval_pairs", type=int, default=5000)
    parser.add_argument("--eval_temperature", type=float, default=1e-3)
    parser.add_argument("--eval_top_k", type=int, default=0)
    parser.add_argument("--eval_require_stop", type=int, default=1, choices=[0, 1],
                        help="评估时是否必须显式 stop/EOS。推荐 1。想和旧日志对齐可设 0。")
    parser.add_argument("--best_metric_bucket", type=str, default="S1->S3",
                        help="保存 best checkpoint 所依据的 bucket。若不存在则退回 overall。")
    parser.add_argument("--log_dir", type=str, default="out_token_grpo")

    args = parser.parse_args()

    if args.rollout_temperature is not None:
        args.rollout_temp_start = args.rollout_temperature
        args.rollout_temp_end = args.rollout_temperature
        args.temp_warmup_iters = 0

    if args.batch_size % args.group_size != 0:
        raise ValueError("--batch_size 必须能被 --group_size 整除。")

    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma 必须满足 0 < gamma <= 1。")

    if args.ppo_epochs < 1:
        raise ValueError("--ppo_epochs 必须 >= 1。")

    if args.kl_min_coef < 0.0:
        raise ValueError("--kl_min_coef 必须 >= 0。")

    if args.kl_coef > 0.0 and args.kl_min_coef > args.kl_coef:
        raise ValueError("--kl_min_coef 不能大于 --kl_coef。")

    return args


# ---------------------------------------------------------------------------
# 调度 / 基础工具
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
    annealed = args.kl_coef * (1.0 - progress)
    return max(args.kl_min_coef, annealed)


def decode_tokens_until_stop(token_ids: Sequence[int],
                             itos: Dict[int, str],
                             stop_token_id: int) -> Tuple[List[str], List[int], bool]:
    prefix_ids: List[int] = []
    had_stop = False
    for tid in token_ids:
        if tid == stop_token_id:
            had_stop = True
            break
        prefix_ids.append(int(tid))
    tokens = [itos.get(tid, "[UNK]") for tid in prefix_ids]
    nodes = [int(tok) for tok in tokens if tok.isdigit()]
    return tokens, nodes, had_stop


def assemble_full_path(source: int, generated_nodes: Sequence[int]) -> List[int]:
    full_path = [source]
    full_path.extend(generated_nodes)
    return full_path


def bucket_for_pair(source: int,
                    target: int,
                    node_to_stage: Dict[int, int]) -> Optional[BucketName]:
    src_stage = node_to_stage.get(source)
    tgt_stage = node_to_stage.get(target)
    if src_stage is None or tgt_stage is None:
        return None
    if src_stage == tgt_stage:
        return None
    return f"S{src_stage + 1}->S{tgt_stage + 1}"


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


def load_train_sequences(train_file: Path,
                         stoi: Dict[str, int],
                         stop_token_id: int,
                         block_size: int) -> List[List[int]]:
    seqs: List[List[int]] = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ids: List[int] = []
            ok = True
            for tok in parts:
                if tok not in stoi:
                    ok = False
                    break
                ids.append(stoi[tok])
            if not ok:
                continue

            if len(ids) >= block_size:
                ids = ids[:block_size - 1]
            if not ids or ids[-1] != stop_token_id:
                ids.append(stop_token_id)

            if len(ids) >= 2:
                seqs.append(ids)
    return seqs


def sample_sft_batch(train_sequences: List[List[int]],
                     batch_size: int,
                     device: torch.device,
                     pad_token_id: int) -> Tuple[Tensor, Tensor]:
    batch = random.choices(train_sequences, k=batch_size)
    max_len = max(len(seq) - 1 for seq in batch)

    x = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    y = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)

    for i, seq in enumerate(batch):
        l = len(seq) - 1
        x[i, :l] = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        y[i, :l] = torch.tensor(seq[1:], dtype=torch.long, device=device)

    return x, y


def prepare_output_dir(base_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / f"grpo_tok_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_max_new_tokens(block_size: int,
                        prompt_len: int,
                        desired: int) -> int:
    available = block_size - prompt_len
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
        "扩容 block_size：checkpoint=%d -> 当前模型=%d。新位置嵌入随机初始化。",
        ckpt_block_size, model_block_size,
    )

    wpe_key = "transformer.wpe.weight"
    if wpe_key in state_dict:
        old_weight = state_dict[wpe_key]
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
        logger.warning("加载 checkpoint 时缺失键：%s", leftover_missing)
    if unexpected_keys:
        logger.warning("加载 checkpoint 时出现未识别键：%s", unexpected_keys)


def get_target_distance_map(target: int,
                            reverse_graph: nx.DiGraph,
                            cache: Dict[int, Dict[str, int]]) -> Dict[str, int]:
    if target not in cache:
        cache[target] = dict(nx.single_source_shortest_path_length(reverse_graph, str(target)))
    return cache[target]


def build_prompt(source: int, target: int, stoi: Dict[str, int]) -> List[int]:
    return [stoi[str(source)], stoi[str(target)], stoi[str(source)]]


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    out = [0.0 for _ in rewards]
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def save_checkpoint(path: Path,
                    model: GPT,
                    model_args: Dict,
                    args: argparse.Namespace,
                    iteration: int,
                    best_metric: Optional[float] = None,
                    best_metric_name: Optional[str] = None) -> None:
    payload = {
        "iter_num": iteration,
        "model": model.state_dict(),
        "model_args": model_args,
        "config": vars(args),
    }
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if best_metric_name is not None:
        payload["best_metric_name"] = best_metric_name
    torch.save(payload, path)


def pick_eval_metric(eval_results: Dict[str, Dict[str, float]],
                     bucket: str) -> Tuple[float, str]:
    if bucket in eval_results:
        return float(eval_results[bucket]["accuracy"]), bucket
    return float(eval_results["overall"]["accuracy"]), "overall"


# ---------------------------------------------------------------------------
# 评估
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
                   require_stop: bool,
                   max_pairs: int = 5000) -> Dict[str, Dict[str, float]]:
    was_training = model.training
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
            prompt_tokens = build_prompt(source, target, stoi)
            max_new_tokens = safe_max_new_tokens(block_size, len(prompt_tokens), max_steps)
            x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

            generated = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
            )[0].tolist()

            new_tokens = generated[len(prompt_tokens):]
            _, generated_nodes, had_stop = decode_tokens_until_stop(new_tokens, itos, stop_token_id)
            full_path = assemble_full_path(source, generated_nodes)
            valid = is_valid_path(full_path, source, target, node_to_stage, stage_sets_list, graph)
            success = valid and (had_stop or not require_stop)

            if success:
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

    if was_training:
        model.train()
    return results


# ---------------------------------------------------------------------------
# logits / logprobs / entropy
# ---------------------------------------------------------------------------
def compute_action_outputs(model: GPT,
                           traj_ids: List[int],
                           action_start: int,
                           num_actions: int,
                           device: torch.device) -> Tuple[Tensor, Tensor]:
    """
    traj_ids = prompt + sampled_actions
    返回从 action_start 开始前 num_actions 个 action 的：
      selected_logprobs: (num_actions,)
      action_logits:     (num_actions, vocab_size)
    """
    if num_actions <= 0:
        empty_lp = torch.tensor([], device=device, dtype=torch.float32)
        empty_logits = torch.zeros(0, 1, device=device)
        return empty_lp, empty_logits

    x = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(traj_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x, y)
    logits = logits.squeeze(0)  # (seq_len, vocab)

    start_idx = action_start - 1
    action_logits = logits[start_idx:start_idx + num_actions]

    log_probs = F.log_softmax(action_logits, dim=-1)
    action_ids = torch.tensor(
        traj_ids[action_start:action_start + num_actions],
        dtype=torch.long,
        device=device,
    )
    selected_logprobs = log_probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
    return selected_logprobs, action_logits


def token_entropy_from_logits(logits: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


# ---------------------------------------------------------------------------
# sampling
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
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(0, topk_indices, top_vals)
        logits = masked

    if epsilon > 0.0 and random.random() < epsilon:
        if topk_indices is not None:
            idx = topk_indices[torch.randint(0, topk_indices.size(0), (1,), device=logits.device)]
        else:
            idx = torch.randint(0, vocab_size, (1,), device=logits.device)
        return int(idx.item())

    if temperature <= 1e-6:
        return int(torch.argmax(logits).item())

    scaled = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


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
                      epsilon: float,
                      dist_to_target: Optional[Dict[str, int]]) -> Dict[str, object]:
    was_training = model.training
    if was_training:
        model.eval()

    sampled_ids: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []

    current_node = source
    reached_target = False
    stop_on_target = False
    stop_before_target = False
    overshoot_after_target = False
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

    for _ in range(max_new_tokens):
        if len(prompt_ids) + len(sampled_ids) >= block_size:
            break

        context = torch.tensor(prompt_ids + sampled_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(context)
        step_logits = logits[0, -1, :]

        next_token_id = select_next_token(
            logits=step_logits,
            temperature=temperature,
            top_k=args.rollout_top_k,
            epsilon=epsilon,
        )

        sampled_ids.append(next_token_id)
        reward = -args.step_penalty if args.step_penalty != 0.0 else 0.0
        done = False

        # 已经到达 target 后，下一步必须 stop
        if reached_target:
            if next_token_id == stop_token_id:
                reward += args.reward_stop
                reward += args.reward_stop_on_target
                stop_on_target = True
                done = True
            else:
                overshoot_after_target = True
                token_str = itos.get(int(next_token_id), "[UNK]")
                if not token_str.isdigit():
                    invalid_token = True
                    reward -= max(args.penalty_overshoot_target, args.reward_invalid_token)
                else:
                    reward -= args.penalty_overshoot_target
                done = True

            rewards.append(float(reward))
            dones.append(bool(done))
            if done:
                break
            continue

        # 尚未到 target
        if next_token_id == stop_token_id:
            reward += args.reward_stop
            stop_before_target = True
            reward -= args.penalty_stop_before_target
            done = True
            rewards.append(float(reward))
            dones.append(bool(done))
            break

        token_str = itos.get(int(next_token_id), "[UNK]")
        if not token_str.isdigit():
            invalid_token = True
            reward -= args.reward_invalid_token
            done = True
            rewards.append(float(reward))
            dones.append(bool(done))
            break

        next_node = int(token_str)
        adjacency = graph.has_edge(str(current_node), str(next_node))

        if adjacency:
            valid_transition_steps += 1
            reward += args.reward_valid_transition

            if args.reward_distance_progress != 0.0 and dist_to_target is not None:
                cur_key = str(current_node)
                nxt_key = str(next_node)
                du = dist_to_target.get(cur_key, None)
                dv = dist_to_target.get(nxt_key, None)

                if du is not None and dv is not None:
                    reward += args.reward_distance_progress * float(du - dv)
                elif du is not None and dv is None:
                    reward -= args.reward_distance_progress
                elif du is None and dv is not None:
                    reward += args.reward_distance_progress

            if intermediate_stage_indices and any(
                next_node in stage_sets_list[mid] for mid in intermediate_stage_indices
            ):
                visited_intermediate = True

            if is_composition and args.reward_stage_bridge > 0.0:
                crossed_mid = any(next_node in stage_sets_list[mid] for mid in intermediate_stage_indices)
                if crossed_mid and (not stage_bridge_rewarded or not args.reward_stage_bridge_only_once):
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
                reached_target = True
        else:
            invalid_transition = True
            invalid_transition_steps += 1
            reward -= args.reward_invalid_transition
            if (not allow_continue) or (invalid_transition_steps >= max_invalid):
                done = True

        rewards.append(float(reward))
        dones.append(bool(done))

        if done:
            break

    # rollout 截断时补 terminal penalty（对齐 termination 语义）
    if rewards and (not dones[-1]):
        if reached_target and (not stop_on_target):
            rewards[-1] -= args.penalty_missing_stop
        elif (not reached_target) and args.penalty_miss_target_final > 0.0:
            rewards[-1] -= args.penalty_miss_target_final

    traj_ids = prompt_ids + sampled_ids
    decoded_tokens, generated_nodes, had_stop = decode_tokens_until_stop(sampled_ids, itos, stop_token_id)
    full_path_nodes = assemble_full_path(source, generated_nodes)

    valid = is_valid_path(full_path_nodes, source, target, node_to_stage, stage_sets_list, graph)
    success = valid and had_stop and stop_on_target

    hit_target_any = target in generated_nodes
    missing_stop = bool(reached_target and (not stop_on_target) and (not overshoot_after_target))

    # outcome-only：只保留终局/失败事件信号，让 RTG 自己向前传播
    if args.reward_type == "outcome":
        outcome_rewards = [0.0 for _ in rewards]
        if outcome_rewards:
            if success and sampled_ids and sampled_ids[-1] == stop_token_id:
                outcome_rewards[-1] = args.reward_hit_target + args.reward_stop_on_target
            elif stop_before_target:
                outcome_rewards[-1] = -args.penalty_stop_before_target
            elif overshoot_after_target:
                outcome_rewards[-1] = -args.penalty_overshoot_target
            elif invalid_token:
                outcome_rewards[-1] = -args.reward_invalid_token
            elif invalid_transition:
                outcome_rewards[-1] = -args.reward_invalid_transition
            elif missing_stop:
                outcome_rewards[-1] = -args.penalty_missing_stop
            elif (not hit_target_any) and args.penalty_miss_target_final > 0.0:
                outcome_rewards[-1] = -args.penalty_miss_target_final
        rewards = outcome_rewards

    episode_reward = float(sum(rewards))
    step_reward_mean = float(np.mean(rewards)) if rewards else 0.0

    first_step_node = generated_nodes[0] if generated_nodes else None
    first_step_is_source = first_step_node == source
    first_step_is_valid = bool(
        first_step_node is not None and graph.has_edge(str(source), str(first_step_node))
    )

    if was_training:
        model.train()

    return {
        "traj_ids": traj_ids,
        "actions": sampled_ids,
        "rewards": rewards,
        "dones": dones,
        "decoded_tokens": decoded_tokens,
        "generated_nodes": generated_nodes,
        "path_nodes": full_path_nodes,
        "success": success,
        "hit_target_any": hit_target_any,
        "reached_target": reached_target,
        "stop_on_target": stop_on_target,
        "stop_before_target": stop_before_target,
        "overshoot_after_target": overshoot_after_target,
        "missing_stop": missing_stop,
        "first_step_is_source": first_step_is_source,
        "first_step_is_valid": first_step_is_valid,
        "episode_reward": episode_reward,
        "step_reward_mean": step_reward_mean,
        "invalid_transition": invalid_transition,
        "invalid_token": invalid_token,
        "visited_intermediate": visited_intermediate or stage_bridge_rewarded,
        "valid_transition_steps": valid_transition_steps,
        "invalid_transition_steps": invalid_transition_steps,
        "bucket": pair_bucket,
        "num_sampled_actions": len(sampled_ids),
    }


# ---------------------------------------------------------------------------
# token-level group-relative advantages
# ---------------------------------------------------------------------------
def assign_group_token_advantages(groups: List[List[Dict[str, object]]],
                                  args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    all_returns: List[float] = []
    all_advs: List[float] = []

    use_loo = bool(args.use_leave_one_out)
    std_floor = float(args.advantage_std_floor)
    adv_clip = float(args.advantage_clip)

    for group in groups:
        for traj in group:
            returns = discounted_returns(traj["rewards"], args.gamma)
            traj["returns"] = returns
            traj["advantages"] = [0.0 for _ in returns]

        if args.advantage_mode == "global":
            vals: List[float] = []
            refs: List[Tuple[int, int]] = []
            for ti, traj in enumerate(group):
                for pos, ret in enumerate(traj["returns"]):
                    vals.append(float(ret))
                    refs.append((ti, pos))

            n = len(vals)
            if n > 1:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                denom = max(std_v, std_floor)
                sum_v = float(np.sum(vals))

                for (ti, pos), v in zip(refs, vals):
                    if use_loo and n > 1:
                        baseline = (sum_v - v) / (n - 1)
                    else:
                        baseline = mean_v
                    adv = (v - baseline) / denom
                    adv = float(np.clip(adv, -adv_clip, adv_clip))
                    group[ti]["advantages"][pos] = adv
        else:
            max_len = max((len(traj["returns"]) for traj in group), default=0)
            for pos in range(max_len):
                idxs = [ti for ti, traj in enumerate(group) if pos < len(traj["returns"])]
                n = len(idxs)
                if n <= 1:
                    continue

                vals = [float(group[ti]["returns"][pos]) for ti in idxs]
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                denom = max(std_v, std_floor)
                sum_v = float(np.sum(vals))

                for ti, v in zip(idxs, vals):
                    if use_loo and n > 1:
                        baseline = (sum_v - v) / (n - 1)
                    else:
                        baseline = mean_v
                    adv = (v - baseline) / denom
                    adv = float(np.clip(adv, -adv_clip, adv_clip))
                    group[ti]["advantages"][pos] = adv

        for traj in group:
            all_returns.extend([float(x) for x in traj["returns"]])
            all_advs.extend([float(x) for x in traj["advantages"]])

    return all_returns, all_advs


# ---------------------------------------------------------------------------
# 主训练
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
    reverse_graph = graph.reverse(copy=False)
    target_dist_cache: Dict[int, Dict[str, int]] = {}

    out_dir = prepare_output_dir(args.log_dir)
    logger = get_logger(os.path.join(out_dir, "train_grpo.log"))

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    logger.info("Token-level GRPO training started.")
    logger.info("Output directory: %s", out_dir)
    logger.info("Algorithm: token-GRPO/PPO | group_size=%d | clip_eps=%.3f | ppo_epochs=%d",
                args.group_size, args.clip_eps, args.ppo_epochs)
    logger.info("Reward type: %s", args.reward_type)
    logger.info("Advantage mode: %s | std_floor=%.3f | adv_clip=%.3f | LOO=%d",
                args.advantage_mode, args.advantage_std_floor, args.advantage_clip, args.use_leave_one_out)
    logger.info("Number of stages: %d", num_stages)
    for i, s in enumerate(stages):
        logger.info("  Stage S%d: %d nodes", i + 1, len(s))

    if args.rollout_top_k == 1 and args.rollout_temp_start < 0.2 and args.epsilon_start < 0.01:
        logger.warning("采样太贪心，group 内可能缺乏差异。建议 top_k=0 且 temp >= 0.6。")

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
                f"不允许扩容 block_size（override={args.block_size_override}, dataset={dataset_block_size}）。"
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

    model = GPT(GPTConfig(**model_args)).to(device)
    load_state_dict_with_block_resize(
        model=model,
        raw_state_dict=ckpt["model"],
        ckpt_block_size=ckpt_block_size,
        logger=logger,
    )

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
        logger.info("KL 启用：coef=%.4f, min=%.4f, warmup=%d, anneal=%d",
                    args.kl_coef, args.kl_min_coef, args.kl_warmup_iters, args.kl_anneal_iters)
    else:
        sft_ref_model = None
        logger.info("KL 禁用。")

    stop_token_id = stoi["\n"]

    train_sequences: List[List[int]] = []
    if args.sft_ce_coef > 0.0:
        train_sequences = load_train_sequences(
            train_file=train_txt,
            stoi=stoi,
            stop_token_id=stop_token_id,
            block_size=model.config.block_size,
        )
        if not train_sequences:
            logger.warning("未能从 train txt 载入任何 SFT sequence，自动禁用 SFT CE anchor。")
            args.sft_ce_coef = 0.0
        else:
            logger.info("Loaded %d SFT sequences for CE anchor.", len(train_sequences))

    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    train_pairs = load_pairs(train_txt)
    logger.info("Loaded %d unique (source, target) pairs for RL training.", len(train_pairs))

    train_gaps = []
    for s, t in train_pairs:
        if s in node_to_stage and t in node_to_stage:
            train_gaps.append(abs(node_to_stage[t] - node_to_stage[s]))
    if train_gaps and max(train_gaps) <= 1 and args.reward_stage_bridge > 0.0:
        logger.warning("当前严格 atomic-only 全是 gap=1，reward_stage_bridge 基本不会触发。")

    eval_pairs: List[Pair] = []
    with open(test_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                eval_pairs.append((int(parts[0]), int(parts[1])))

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
    num_prompts_per_iter = args.batch_size // G
    logger.info("每次迭代采样 %d 个 prompt × %d 条轨迹 = %d 总轨迹",
                num_prompts_per_iter, G, num_prompts_per_iter * G)

    best_metric = -1.0
    best_metric_name = None
    best_ckpt_path = out_dir / "ckpt_grpo_best.pt"

    model.train()

    for iteration in range(1, args.max_iters + 1):
        temperature = current_temperature(iteration, args)
        epsilon = current_epsilon(iteration, args)
        kl_coef_now = current_kl_coef(iteration, args)

        prompt_pairs = random.choices(train_pairs, k=num_prompts_per_iter)

        # ---------------------------------------------------
        # Phase 1: rollout + old logprobs
        # ---------------------------------------------------
        all_groups: List[List[Dict[str, object]]] = []
        all_trajs: List[Dict[str, object]] = []

        model.eval()

        for source, target in prompt_pairs:
            prompt_ids = build_prompt(source, target, stoi)
            bucket = bucket_for_pair(source, target, node_to_stage)

            dist_to_target = None
            if args.reward_distance_progress != 0.0:
                dist_to_target = get_target_distance_map(target, reverse_graph, target_dist_cache)

            group: List[Dict[str, object]] = []
            for _ in range(G):
                traj = sample_trajectory(
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
                    dist_to_target=dist_to_target,
                )

                num_sampled = traj["num_sampled_actions"]
                if num_sampled > 0:
                    with torch.no_grad():
                        old_lp, _ = compute_action_outputs(
                            model=model,
                            traj_ids=traj["traj_ids"],
                            action_start=action_start,
                            num_actions=num_sampled,
                            device=device,
                        )
                else:
                    old_lp = torch.tensor([], device=device, dtype=torch.float32)

                traj["old_logprobs"] = old_lp
                group.append(traj)
                all_trajs.append(traj)

            all_groups.append(group)

        # ---------------------------------------------------
        # Phase 2: token-level group-relative advantages
        # ---------------------------------------------------
        token_returns, token_advs = assign_group_token_advantages(all_groups, args)

        # ---------------------------------------------------
        # 采样统计
        # ---------------------------------------------------
        episode_rewards: List[float] = []
        step_rewards: List[float] = []
        path_lengths: List[int] = []
        valid_transition_list: List[int] = []
        invalid_transition_steps_list: List[int] = []
        first_step_source_list: List[float] = []
        first_step_valid_list: List[float] = []
        group_reward_stds: List[float] = []

        bucket_success_sum = defaultdict(float)
        bucket_counts = defaultdict(int)

        success_count = 0
        hit_any_count = 0
        stop_target_count = 0
        stop_early_count = 0
        overshoot_count = 0
        missing_stop_count = 0
        invalid_transition_count = 0
        invalid_token_count = 0
        intermediate_visit_count = 0

        for group in all_groups:
            ep_rews = [float(t["episode_reward"]) for t in group]
            group_reward_stds.append(float(np.std(ep_rews)))

        for traj in all_trajs:
            episode_rewards.append(float(traj["episode_reward"]))
            step_rewards.append(float(traj["step_reward_mean"]))
            path_lengths.append(len(traj["path_nodes"]))
            valid_transition_list.append(int(traj["valid_transition_steps"]))
            invalid_transition_steps_list.append(int(traj["invalid_transition_steps"]))
            first_step_source_list.append(1.0 if traj["first_step_is_source"] else 0.0)
            first_step_valid_list.append(1.0 if traj["first_step_is_valid"] else 0.0)

            succ = bool(traj["success"])
            success_count += int(succ)
            hit_any_count += int(traj["hit_target_any"])
            stop_target_count += int(traj["stop_on_target"])
            stop_early_count += int(traj["stop_before_target"])
            overshoot_count += int(traj["overshoot_after_target"])
            missing_stop_count += int(traj["missing_stop"])
            invalid_transition_count += int(traj["invalid_transition"])
            invalid_token_count += int(traj["invalid_token"])
            intermediate_visit_count += int(traj["visited_intermediate"])

            bname = traj["bucket"]
            if bname:
                bucket_success_sum[bname] += 1.0 if succ else 0.0
                bucket_counts[bname] += 1

        # ---------------------------------------------------
        # Phase 3: PPO epochs
        # ---------------------------------------------------
        model.train()

        had_update = False
        last_total_loss_val = 0.0
        last_policy_loss_val = 0.0
        last_kl_val = 0.0
        last_entropy_val = 0.0
        last_sft_ce_val = 0.0
        last_ratio_list: List[float] = []
        last_clip_fractions: List[float] = []

        for _ppo_epoch in range(args.ppo_epochs):
            random.shuffle(all_trajs)

            policy_terms: List[Tensor] = []
            kl_terms: List[Tensor] = []
            entropy_terms: List[Tensor] = []
            epoch_ratio_list: List[float] = []
            epoch_clip_fractions: List[float] = []

            for traj in all_trajs:
                num_act = int(traj["num_sampled_actions"])
                if num_act <= 0:
                    continue

                old_logprobs: Tensor = traj["old_logprobs"].detach()
                adv_list: List[float] = traj["advantages"]
                if len(adv_list) <= 0:
                    continue

                new_logprobs, new_logits = compute_action_outputs(
                    model=model,
                    traj_ids=traj["traj_ids"],
                    action_start=action_start,
                    num_actions=num_act,
                    device=device,
                )

                act_len = min(num_act, len(adv_list), old_logprobs.size(0), new_logprobs.size(0))
                if act_len <= 0:
                    continue

                old_logprobs = old_logprobs[:act_len]
                new_logprobs = new_logprobs[:act_len]
                new_logits = new_logits[:act_len]
                adv_t = torch.tensor(adv_list[:act_len], dtype=torch.float32, device=device)

                ratio = torch.exp(new_logprobs - old_logprobs)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t
                policy_terms.append(torch.min(surr1, surr2))

                with torch.no_grad():
                    epoch_ratio_list.append(float(ratio.mean().item()))
                    clipped = ((ratio < 1.0 - args.clip_eps) | (ratio > 1.0 + args.clip_eps)).float()
                    epoch_clip_fractions.append(float(clipped.mean().item()))

                if kl_coef_now > 0.0 and sft_ref_model is not None:
                    with torch.no_grad():
                        _, ref_logits = compute_action_outputs(
                            model=sft_ref_model,
                            traj_ids=traj["traj_ids"],
                            action_start=action_start,
                            num_actions=act_len,
                            device=device,
                        )
                    kl_per_tok = F.kl_div(
                        F.log_softmax(new_logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    kl_terms.append(kl_per_tok)

                if args.entropy_coef > 0.0:
                    entropy_terms.append(token_entropy_from_logits(new_logits))

            mean_policy_loss = torch.tensor(0.0, device=device)
            mean_kl = torch.tensor(0.0, device=device)
            mean_entropy = torch.tensor(0.0, device=device)
            sft_ce_loss = torch.tensor(0.0, device=device)

            if policy_terms:
                mean_policy_loss = -torch.cat(policy_terms).mean()
            if kl_terms and kl_coef_now > 0.0:
                mean_kl = torch.cat(kl_terms).mean()
            if entropy_terms and args.entropy_coef > 0.0:
                mean_entropy = torch.cat(entropy_terms).mean()
            if args.sft_ce_coef > 0.0 and train_sequences:
                x_sft, y_sft = sample_sft_batch(
                    train_sequences=train_sequences,
                    batch_size=args.sft_ce_batch_size,
                    device=device,
                    pad_token_id=stop_token_id,
                )
                _, sft_ce_loss = model(x_sft, y_sft)

            # 若 policy 信号为 0，仍允许 CE / KL 把模型拉回 anchor
            if (not policy_terms) and (args.sft_ce_coef <= 0.0):
                continue

            had_update = True
            total_loss = mean_policy_loss
            total_loss = total_loss + kl_coef_now * mean_kl
            total_loss = total_loss + args.sft_ce_coef * sft_ce_loss
            total_loss = total_loss - args.entropy_coef * mean_entropy

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            last_total_loss_val = float(total_loss.item())
            last_policy_loss_val = float(mean_policy_loss.item())
            last_kl_val = float(mean_kl.item())
            last_entropy_val = float(mean_entropy.item())
            last_sft_ce_val = float(sft_ce_loss.item())
            last_ratio_list = epoch_ratio_list
            last_clip_fractions = epoch_clip_fractions

        if not had_update:
            logger.warning("Iter %d 无有效更新，跳过。", iteration)
            continue

        # ---------------------------------------------------
        # logging
        # ---------------------------------------------------
        total_trajs = len(all_trajs)
        avg_ep_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        avg_step_reward = float(np.mean(step_rewards)) if step_rewards else 0.0
        avg_path_len = float(np.mean(path_lengths)) if path_lengths else 0.0
        avg_valid_tr = float(np.mean(valid_transition_list)) if valid_transition_list else 0.0
        avg_inv_tr = float(np.mean(invalid_transition_steps_list)) if invalid_transition_steps_list else 0.0
        first_src_rate = float(np.mean(first_step_source_list)) if first_step_source_list else 0.0
        first_valid_rate = float(np.mean(first_step_valid_list)) if first_step_valid_list else 0.0
        success_rate = success_count / total_trajs if total_trajs else 0.0
        hit_any_rate = hit_any_count / total_trajs if total_trajs else 0.0
        stop_target_rate = stop_target_count / total_trajs if total_trajs else 0.0
        stop_early_rate = stop_early_count / total_trajs if total_trajs else 0.0
        overshoot_rate = overshoot_count / total_trajs if total_trajs else 0.0
        missing_stop_rate = missing_stop_count / total_trajs if total_trajs else 0.0
        inv_edge_rate = invalid_transition_count / total_trajs if total_trajs else 0.0
        inv_tok_rate = invalid_token_count / total_trajs if total_trajs else 0.0
        intermed_rate = intermediate_visit_count / total_trajs if total_trajs else 0.0

        mean_grp_std = float(np.mean(group_reward_stds)) if group_reward_stds else 0.0
        mean_token_return = float(np.mean(token_returns)) if token_returns else 0.0
        mean_token_adv_abs = float(np.mean(np.abs(token_advs))) if token_advs else 0.0
        mean_ratio_val = float(np.mean(last_ratio_list)) if last_ratio_list else 1.0
        mean_clip_frac = float(np.mean(last_clip_fractions)) if last_clip_fractions else 0.0

        if iteration % 50 == 0:
            logger.info(
                "Iter %6d | loss=%.4f | policy=%.4f | kl=%.4f(coef=%.4f) | ce=%.4f | ent=%.4f | "
                "temp=%.3f | eps=%.3f | success=%.3f | hit_any=%.3f | stop_tgt=%.3f | "
                "stop_early=%.3f | overshoot=%.3f | miss_stop=%.3f | first_src=%.3f | "
                "first_valid=%.3f | inv_edge=%.3f | inv_tok=%.3f | ep_rew=%.3f | path=%.2f | "
                "tok_ret=%.3f | tok_adv_abs=%.3f | grp_std=%.3f | ratio_last=%.3f | clip_last=%.3f",
                iteration,
                last_total_loss_val,
                last_policy_loss_val,
                last_kl_val,
                kl_coef_now,
                last_sft_ce_val,
                last_entropy_val,
                temperature,
                epsilon,
                success_rate,
                hit_any_rate,
                stop_target_rate,
                stop_early_rate,
                overshoot_rate,
                missing_stop_rate,
                first_src_rate,
                first_valid_rate,
                inv_edge_rate,
                inv_tok_rate,
                avg_ep_reward,
                avg_path_len,
                mean_token_return,
                mean_token_adv_abs,
                mean_grp_std,
                mean_ratio_val,
                mean_clip_frac,
            )

        record = {
            "iter": iteration,
            "loss": last_total_loss_val,
            "policy_loss": last_policy_loss_val,
            "kl_loss": last_kl_val,
            "kl_coef_current": kl_coef_now,
            "sft_ce_loss": last_sft_ce_val,
            "entropy": last_entropy_val,
            "temperature": temperature,
            "epsilon": epsilon,
            "success_rate": success_rate,
            "hit_any_rate": hit_any_rate,
            "stop_on_target_rate": stop_target_rate,
            "stop_before_target_rate": stop_early_rate,
            "overshoot_rate": overshoot_rate,
            "missing_stop_rate": missing_stop_rate,
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
            "mean_group_reward_std": mean_grp_std,
            "mean_token_return": mean_token_return,
            "mean_token_adv_abs": mean_token_adv_abs,
            "mean_ratio_last_epoch": mean_ratio_val,
            "clip_fraction_last_epoch": mean_clip_frac,
        }
        for bname in all_bucket_names:
            cnt = bucket_counts.get(bname, 0)
            record[f"train_success/{bname}"] = (
                bucket_success_sum.get(bname, 0.0) / cnt if cnt > 0 else 0.0
            )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ---------------------------------------------------
        # evaluation
        # ---------------------------------------------------
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
                require_stop=bool(args.eval_require_stop),
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

            metric_val, metric_name = pick_eval_metric(eval_results, args.best_metric_bucket)
            if metric_val > best_metric:
                best_metric = metric_val
                best_metric_name = metric_name
                save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    model_args=model_args,
                    args=args,
                    iteration=iteration,
                    best_metric=best_metric,
                    best_metric_name=best_metric_name,
                )
                logger.info("Saved NEW BEST checkpoint to %s | %s=%.4f",
                            best_ckpt_path, best_metric_name, best_metric)

            eval_record = {
                "iter": iteration,
                "eval": eval_results,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

        # ---------------------------------------------------
        # periodic save
        # ---------------------------------------------------
        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_path = out_dir / f"ckpt_grpo_{iteration}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                model_args=model_args,
                args=args,
                iteration=iteration,
            )
            logger.info("Saved checkpoint to %s", ckpt_path)

    logger.info("Training finished.")
    if best_metric_name is not None:
        logger.info("Best checkpoint: %s | %s=%.4f", best_ckpt_path, best_metric_name, best_metric)


if __name__ == "__main__":
    main()