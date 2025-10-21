#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT training script for GraphA Tier-3 datasets.

Key additions vs previous version:
  * --train_paths_per_pair argument to load matching train_XX.bin.
  * Weight gap computation (difference between true next-token logits and sampled negatives).
  * Metrics logging to JSONL for easy plotting.
  * Stratified accuracy reporting (S1->S2 / S2->S3 / S1->S3) plus overall.

Usage example:
    python train_composition_fixed_final.py \
        --data_dir data/datasets/graphA_pg020_tier3 \
        --device cuda:0 \
        --train_paths_per_pair 20 \
        --max_iters 50000 \
        --test_interval 1000 \
        --checkpoint_interval 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from logger import get_logger  # Assuming your project provides this module
from model import GPT, GPTConfig  # Assuming your project provides this module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory produced by dataset generator.")
    parser.add_argument("--train_paths_per_pair", type=int, default=20,
                        help="Matches train_{K}.bin produced during preprocessing.")
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=120)
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_gap_neg_samples", type=int, default=20,
                        help="Negative samples per step when computing weight gap.")
    parser.add_argument("--weight_gap_lines", type=int, default=4000,
                        help="Maximum lines sampled from train.txt for weight gap.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def evaluate_composition(
    model: GPT,
    test_file: Path,
    stages: List[List[int]],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    G: nx.DiGraph,
    vocab_size: int,
    temperature: float = 0.1,
    top_k: int = 10,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate composition accuracy per path type plus overall."""
    model.eval()
    S1, S2, S3 = stages
    token_level = vocab_size > 50

    with open(test_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    buckets: Dict[str, List[Tuple[int, int, List[int]]]] = {
        "S1->S2": [],
        "S2->S3": [],
        "S1->S3": [],
    }

    for line in lines:
        parts = line.split()
        source, target = int(parts[0]), int(parts[1])
        path = list(map(int, parts[2:]))
        if source in S1 and target in S2:
            buckets["S1->S2"].append((source, target, path))
        elif source in S2 and target in S3:
            buckets["S2->S3"].append((source, target, path))
        elif source in S1 and target in S3:
            buckets["S1->S3"].append((source, target, path))

    def decode_tokens(token_ids: List[int]) -> List[int]:
        numbers: List[int] = []
        for tid in token_ids:
            if tid == stoi["\n"]:
                break
            if tid in itos:
                token = itos[tid]
                if token.isdigit():
                    numbers.append(int(token))
        return numbers

    results: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    total_cases = 0

    for path_type, cases in buckets.items():
        correct = 0
        for idx, (source, target, _) in enumerate(cases):
            if token_level:
                prompt_tokens = [source, target, source]
                prompt_ids = [stoi[str(tok)] for tok in prompt_tokens]
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                generated = model.generate(
                    x, max_new_tokens=32, temperature=temperature, top_k=top_k
                )[0].tolist()
                decoded = decode_tokens(generated)
                generated_path = decoded[2:] if len(decoded) >= 3 else []
            else:
                raise NotImplementedError("Character-level decoding not supported in this setup.")

            valid = False
            if len(generated_path) >= 2:
                if generated_path[0] == source and generated_path[-1] == target:
                    path_valid = all(
                        G.has_edge(str(generated_path[i]), str(generated_path[i + 1]))
                        for i in range(len(generated_path) - 1)
                    )
                    if path_valid:
                        if path_type == "S1->S3":
                            has_s2 = any(node in S2 for node in generated_path[1:-1])
                            if has_s2:
                                valid = True
                        else:
                            valid = True

            if verbose and idx < 3:
                status = "✓" if valid else "✗"
                print(f"[Eval-{path_type}] {status} {source}->{target}: {generated_path}")

            if valid:
                correct += 1

        total_correct += correct
        total_cases += len(cases)
        acc = correct / len(cases) if cases else 0.0
        results[path_type] = {
            "correct": correct,
            "total": len(cases),
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


@torch.no_grad()
def compute_weight_gap(
    model: GPT,
    sample_lines: List[str],
    stoi: Dict[str, int],
    device: torch.device,
    neg_sample_ids: List[int],
    max_lines: int,
    neg_samples_per_step: int,
) -> Dict[str, float]:
    """Compute average logit gap between true next token and sampled negatives."""
    model.eval()
    random.shuffle(sample_lines)
    lines = sample_lines[:max_lines]

    true_logits: List[float] = []
    neg_logits: List[float] = []

    for line in lines:
        tokens = line.strip().split()
        try:
            token_ids = [stoi[token] for token in tokens]
        except KeyError:
            continue
        token_ids.append(stoi["\n"])
        if len(token_ids) < 3:
            continue

        for idx in range(2, len(token_ids)):
            prefix = token_ids[:idx]
            target_id = token_ids[idx]
            if target_id not in stoi.values():
                continue

            prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(prefix_tensor, None)
            step_logits = logits[0, -1, :]

            true_logit = step_logits[target_id].item()
            true_logits.append(true_logit)

            neg_candidates = [nid for nid in neg_sample_ids if nid != target_id]
            if not neg_candidates:
                continue
            sampled = random.sample(
                neg_candidates, min(neg_samples_per_step, len(neg_candidates))
            )
            neg_logit = step_logits[sampled].mean().item()
            neg_logits.append(neg_logit)

    model.train()

    result = {
        "num_steps": len(true_logits),
        "avg_true_logit": float(np.mean(true_logits)) if true_logits else float("nan"),
        "avg_neg_logit": float(np.mean(neg_logits)) if neg_logits else float("nan"),
    }
    result["weight_gap"] = result["avg_true_logit"] - result["avg_neg_logit"]
    return result


def get_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_span = block_size + 1
    num_sequences = len(data) // data_span
    if num_sequences == 0:
        raise ValueError("Not enough data to form even one sequence. Check preprocessing.")

    seq_indices = torch.randint(0, num_sequences, (batch_size,))
    offsets = seq_indices * data_span

    x_list = []
    y_list = []
    for offset in offsets:
        offset = int(offset.item())
        x_chunk = torch.from_numpy(data[offset: offset + block_size].astype(np.int64))
        y_chunk = torch.from_numpy(data[offset + 1: offset + 1 + block_size].astype(np.int64))
        x_list.append(x_chunk)
        y_list.append(y_chunk)

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    return x, y


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("out") / f"composition_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(os.path.join(out_dir, "train.log"))
    logger.info("Starting GraphA Tier-3 training")

    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages = stage_info["stages"]
    total_nodes = sum(len(stage) for stage in stages)

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]
    block_size = meta["block_size"]
    vocab_size = meta["vocab_size"]

    logger.info("Vocabulary size: %d", vocab_size)
    logger.info("Block size: %d", block_size)

    train_bin = data_dir / f"train_{args.train_paths_per_pair}.bin"
    val_bin = data_dir / "val.bin"
    if not train_bin.exists():
        raise FileNotFoundError(f"Training bin not found: {train_bin}")
    if not val_bin.exists():
        raise FileNotFoundError(f"Validation bin not found: {val_bin}")

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    G = nx.read_graphml(data_dir / "composition_graph.graphml")
    test_file = data_dir / "test.txt"

    # Load train text lines for weight gap
    train_text_file = data_dir / f"train_{args.train_paths_per_pair}.txt"
    if train_text_file.exists():
        with open(train_text_file, "r", encoding="utf-8") as f:
            train_lines = [line.strip() for line in f if line.strip()]
    else:
        train_lines = []
        logger.warning("Train text file not found; weight gap computation will be skipped.")

    # Node token IDs (exclude PAD/newline)
    node_token_ids = []
    for node_idx in range(total_nodes):
        token_str = str(node_idx)
        if token_str in stoi:
            node_token_ids.append(stoi[token_str])

    model_args = dict(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
    )
    model = GPT(GPTConfig(**model_args)).to(device)
    logger.info("Model parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if device.type == "cuda" else "cpu",
    )

    metrics_path = out_dir / "metrics_history.jsonl"
    running_loss = 0.0
    loss_counter = 0

    logger.info("Entering training loop (max_iters=%d)", args.max_iters)

    for iter_num in range(args.max_iters + 1):
        # Linear warmup
        lr = args.learning_rate
        if iter_num < 2000:
            lr = args.learning_rate * (iter_num + 1) / 2000
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % args.test_interval == 0:
            avg_train_loss = running_loss / loss_counter if loss_counter > 0 else float("nan")

            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    X_val, Y_val = get_batch(val_data, block_size, args.batch_size, device)
                    _, val_loss = model(X_val, Y_val)
                    val_losses.append(val_loss.item())
            val_loss = float(np.mean(val_losses))

            results = evaluate_composition(
                model=model,
                test_file=test_file,
                stages=stages,
                stoi=stoi,
                itos=itos,
                device=device,
                G=G,
                vocab_size=vocab_size,
                temperature=args.temperature,
                top_k=args.top_k,
                verbose=False,
            )

            if node_token_ids and train_lines:
                weight_gap_stats = compute_weight_gap(
                    model=model,
                    sample_lines=train_lines,
                    stoi=stoi,
                    device=device,
                    neg_sample_ids=node_token_ids,
                    max_lines=args.weight_gap_lines,
                    neg_samples_per_step=args.weight_gap_neg_samples,
                )
            else:
                weight_gap_stats = {
                    "num_steps": 0,
                    "avg_true_logit": float("nan"),
                    "avg_neg_logit": float("nan"),
                    "weight_gap": float("nan"),
                }

            metrics_record = {
                "iter": iter_num,
                "lr": lr,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "accuracy": results,
                "weight_gap": weight_gap_stats,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_record) + "\n")

            logger.info("=" * 70)
            logger.info("Iteration %d", iter_num)
            logger.info("Loss: train=%.4f, val=%.4f", avg_train_loss, val_loss)
            for key in ["S1->S2", "S2->S3", "S1->S3", "overall"]:
                stats = results.get(key, {})
                logger.info(
                    "  %s: %.2f%% (%d/%d)",
                    key,
                    stats.get("accuracy", 0.0) * 100,
                    stats.get("correct", 0),
                    stats.get("total", 0),
                )
            logger.info(
                "Weight gap: gap=%.4f (true=%.4f, neg=%.4f, steps=%d)",
                weight_gap_stats["weight_gap"],
                weight_gap_stats["avg_true_logit"],
                weight_gap_stats["avg_neg_logit"],
                weight_gap_stats["num_steps"],
            )
            logger.info("=" * 70)

            running_loss = 0.0
            loss_counter = 0
            model.train()

        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            checkpoint_path = out_dir / f"ckpt_{iter_num}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            logger.info("Saved checkpoint: %s", checkpoint_path)

        if iter_num == args.max_iters:
            break
        if iter_num == 0:
            continue

        X, Y = get_batch(train_data, block_size, args.batch_size, device)
        logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        loss_counter += 1

    logger.info("Training finished. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()