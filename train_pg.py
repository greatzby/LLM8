#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Gradient (PG) fine-tuning script for GraphA Tier-3 datasets.

参考用法：
    python train_pg.py \
        --data_dir data/datasets/graphA_pg020_tier3 \
        --sft_checkpoint out/sft_run/ckpt_50000.pt \
        --train_paths_per_pair 20 \
        --device cuda:0 \
        --max_iters 20000 \
        --eval_interval 1000 \
        --save_interval 2000 \
        --batch_size 32 \
        --max_rollout_steps 32 \
        --rollout_temperature 1.2 \
        --kl_coef 3e-4
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


# -----------------------------------------------------------------------------
# 命令行参数
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy Gradient fine-tuning for GraphA.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据集目录（包含 train_K.txt、meta.pkl、stage_info.pkl 等文件）。")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="SFT 训练得到的 checkpoint (.pt)，作为 RL 初始化。")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="对应 train_{K}.txt 中 K 的取值。")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    # PG 超参数
    parser.add_argument("--max_iters", type=int, default=20000,
                        help="PG update 步数。")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="每个 PG update 采样多少 (s, t) 对。")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="策略学习率。")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值。")
    parser.add_argument("--adv_clip", type=float, default=5.0,
                        help="优势项裁剪绝对值（<=0 表示不裁剪）。")

    parser.add_argument("--max_rollout_steps", type=int, default=32,
                        help="每次 rollout 最多生成多少 token。")
    parser.add_argument("--rollout_temperature", type=float, default=1.2,
                        help="rollout 温度，建议≥1 便于探索。")
    parser.add_argument("--rollout_top_k", type=int, default=0,
                        help="rollout 采样时的 top-k（<=0 表示不截断）。")

    parser.add_argument("--kl_coef", type=float, default=3e-4,
                        help="KL 正则系数，惩罚策略偏离 SFT。设为 0 可关闭。")
    parser.add_argument("--baseline_beta", type=float, default=0.95,
                        help="奖励基线 EMA 系数。")

    # 评估 & 日志
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="每隔多少 step 进行一次评估。")
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="每隔多少 step 保存一次模型。")
    parser.add_argument("--max_eval_pairs", type=int, default=5000,
                        help="评估使用的 (s, t) 对数量上限。")
    parser.add_argument("--eval_temperature", type=float, default=0.0,
                        help="评估时的温度，建议 0 表示 greedy。")
    parser.add_argument("--eval_top_k", type=int, default=0,
                        help="评估时的 top-k（<=0 表示 greedy）。")

    parser.add_argument("--log_dir", type=str, default="out_pg",
                        help="日志与 checkpoint 输出目录。")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def bucket_for_pair(source: int,
                    target: int,
                    stages: Sequence[Sequence[int]]) -> Optional[BucketName]:
    S1, S2, S3 = stages[:3]
    if source in S1 and target in S2:
        return "S1->S2"
    if source in S2 and target in S3:
        return "S2->S3"
    if source in S1 and target in S3:
        return "S1->S3"
    return None


def is_valid_path(path_nodes: List[int],
                  source: int,
                  target: int,
                  stages: Sequence[Sequence[int]],
                  graph: nx.DiGraph) -> bool:
    if len(path_nodes) < 2:
        return False
    if path_nodes[0] != source or path_nodes[-1] != target:
        return False

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if not graph.has_edge(str(u), str(v)):
            return False

    S1, S2, S3 = stages[:3]
    if source in S1 and target in S3:
        if not any(node in S2 for node in path_nodes[1:-1]):
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
    out_dir = Path(base_dir) / f"pg_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------------------------------------------------------
# 评估（记录三类路径准确率，保持与 SFT 对齐）
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model: GPT,
                   pairs: List[Pair],
                   stages: Sequence[Sequence[int]],
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

    buckets: Dict[BucketName, List[Pair]] = {
        "S1->S2": [],
        "S2->S3": [],
        "S1->S3": [],
    }
    for s, t in pairs[:max_pairs]:
        bucket = bucket_for_pair(s, t, stages)
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
            x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

            generated = model.generate(
                x,
                max_new_tokens=max_steps,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
            )[0].tolist()

            new_tokens = generated[len(prompt_tokens):]
            decoded_tokens = decode_tokens(new_tokens, itos, stop_token_id)
            path_nodes = tokens_to_nodes(decoded_tokens)

            if is_valid_path(path_nodes, source, target, stages, graph):
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


# -----------------------------------------------------------------------------
# log prob & KL
# -----------------------------------------------------------------------------
def compute_logprob_and_kl(model: GPT,
                           base_model: Optional[GPT],
                           traj_ids: List[int],
                           action_start: int,
                           device: torch.device) -> Tuple[Tensor, Tensor]:
    if len(traj_ids) <= action_start:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    x_ids = torch.tensor(traj_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y_ids = torch.tensor(traj_ids[1:], dtype=torch.long, device=device).unsqueeze(0)

    logits, _ = model(x_ids, y_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)  # [1, T]
    logprob_sum = selected[:, action_start - 1:].sum()

    if base_model is None:
        kl_sum = torch.tensor(0.0, device=device)
    else:
        with torch.no_grad():
            base_logits, _ = base_model(x_ids, y_ids)
            base_log_probs = F.log_softmax(base_logits, dim=-1)
        probs = log_probs.exp()
        kl_per_token = (probs * (log_probs - base_log_probs)).sum(dim=-1)
        kl_sum = kl_per_token[:, action_start - 1:].sum()

    return logprob_sum, kl_sum


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
                   stop_token_id: int) -> List[int]:
    traj = prompt_ids + sampled_ids
    if not sampled_ids or sampled_ids[-1] != stop_token_id:
        traj.append(stop_token_id)
    return traj


# -----------------------------------------------------------------------------
# 主训练逻辑
# -----------------------------------------------------------------------------
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
    block_size = meta["block_size"]

    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages: List[List[int]] = stage_info["stages"]

    graph = nx.read_graphml(data_dir / "composition_graph.graphml")

    out_dir = prepare_output_dir(args.log_dir)
    logger = get_logger(os.path.join(out_dir, "train_pg.log"))
    logger.info("Policy Gradient training started.")
    logger.info("Output directory: %s", out_dir)
    logger.info("KL coefficient: %.6f", args.kl_coef)

    model_args = dict(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=1,
        n_head=1,
        n_embd=92,
        dropout=0.0,
        bias=False,
    )
    model = GPT(GPTConfig(**model_args)).to(device)

    ckpt = torch.load(args.sft_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    if args.kl_coef > 0:
        base_model = GPT(GPTConfig(**model_args)).to(device)
        base_model.load_state_dict(ckpt["model"])
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()
    else:
        base_model = None

    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    train_pairs = load_pairs(train_txt)
    logger.info("Loaded %d unique (source, target) pairs for PG training.", len(train_pairs))

    eval_pairs: List[Pair] = []
    with open(test_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                eval_pairs.append((int(parts[0]), int(parts[1])))

    stop_token_id = stoi["\n"]
    action_start = 3
    baseline_reward = 0.0
    metrics_path = out_dir / "metrics_pg.jsonl"

    model.train()
    bucket_names = ["S1->S2", "S2->S3", "S1->S3"]

    for iteration in range(1, args.max_iters + 1):
        batch_pairs = random.choices(train_pairs, k=args.batch_size)

        pg_losses: List[Tensor] = []
        kl_losses: List[Tensor] = []
        rewards: List[float] = []
        path_lengths: List[int] = []

        bucket_reward_sum = defaultdict(float)
        bucket_counts = defaultdict(int)

        successes = 0

        for source, target in batch_pairs:
            prompt_ids = build_prompt(source, target, stoi)
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

            generated_full = model.generate(
                x,
                max_new_tokens=args.max_rollout_steps,
                temperature=args.rollout_temperature,
                top_k=args.rollout_top_k if args.rollout_top_k > 0 else None,
            )[0].tolist()

            sampled_ids = generated_full[len(prompt_ids):]
            decoded_tokens = decode_tokens(sampled_ids, itos, stop_token_id)
            path_nodes = tokens_to_nodes(decoded_tokens)

            reward = 1.0 if is_valid_path(path_nodes, source, target, stages, graph) else 0.0
            rewards.append(reward)
            if reward > 0.5:
                successes += 1

            bucket = bucket_for_pair(source, target, stages)
            if bucket:
                bucket_reward_sum[bucket] += reward
                bucket_counts[bucket] += 1

            path_lengths.append(len(path_nodes))

            traj_ids = build_traj_ids(prompt_ids, sampled_ids, stop_token_id)
            logprob_sum, kl_sum = compute_logprob_and_kl(
                model=model,
                base_model=base_model,
                traj_ids=traj_ids,
                action_start=action_start,
                device=device,
            )

            advantage = reward - baseline_reward
            if args.adv_clip > 0:
                advantage = float(np.clip(advantage, -args.adv_clip, args.adv_clip))

            pg_losses.append(-advantage * logprob_sum)
            kl_losses.append(kl_sum)

            baseline_reward = args.baseline_beta * baseline_reward + (1 - args.baseline_beta) * reward

        mean_pg_loss = torch.stack(pg_losses).mean()
        mean_kl_loss = torch.stack(kl_losses).mean() if base_model is not None else torch.tensor(0.0, device=device)
        total_loss = mean_pg_loss + args.kl_coef * mean_kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_path_len = float(np.mean(path_lengths)) if path_lengths else 0.0
        success_rate = successes / len(batch_pairs)

        if iteration % 50 == 0:
            logger.info(
                "Iter %6d | reward=%.3f | success=%.3f | avg_path=%.2f | pg_loss=%.4f | kl_loss=%.4f",
                iteration, avg_reward, success_rate, avg_path_len,
                mean_pg_loss.item(), mean_kl_loss.item()
            )

        record = {
            "iter": iteration,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "avg_path_len": avg_path_len,
            "pg_loss": float(mean_pg_loss.item()),
            "kl_loss": float(mean_kl_loss.item()),
            "total_loss": float(total_loss.item()),
            "baseline": float(baseline_reward),
        }
        for bucket in bucket_names:
            cnt = bucket_counts.get(bucket, 0)
            total = float(cnt) if cnt > 0 else 1.0
            record[f"train_reward/{bucket}"] = bucket_reward_sum.get(bucket, 0.0) / total
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            eval_results = evaluate_model(
                model=model,
                pairs=eval_pairs,
                stages=stages,
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
            for bucket, stats in eval_results.items():
                logger.info(
                    "  %s: %.2f%% (%d / %d)",
                    bucket,
                    stats["accuracy"] * 100.0,
                    stats["correct"],
                    stats["total"],
                )
            eval_record = {"iter": iteration, "eval": eval_results}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record) + "\n")

        if iteration % args.save_interval == 0 or iteration == args.max_iters:
            ckpt_path = out_dir / f"ckpt_pg_{iteration}.pt"
            torch.save(
                {
                    "iter_num": iteration,
                    "model": model.state_dict(),
                    "model_args": model_args,
                    "config": vars(args),
                },
                ckpt_path,
            )
            logger.info("Saved PG checkpoint to %s", ckpt_path)

    logger.info("PG training finished.")


if __name__ == "__main__":
    main()