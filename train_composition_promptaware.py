#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
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

from logger import get_logger
from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_paths_per_pair", type=int, default=20)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=120)
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_max_new_tokens", type=int, default=32)
    return parser.parse_args()


def load_prompt_template(data_dir: Path, meta: dict) -> Optional[List[str]]:
    if isinstance(meta.get("prompt_template_tokens"), list):
        return meta["prompt_template_tokens"]

    tmpl_file = data_dir / "instruction_template.json"
    if tmpl_file.exists():
        with open(tmpl_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tokens = obj.get("prompt_template_tokens")
        if isinstance(tokens, list):
            return tokens
    return None


def format_prompt_tokens(template_tokens: Optional[List[str]], source: int, target: int) -> List[str]:
    if not template_tokens:
        # original symbolic fallback
        return [str(source), str(target), str(source)]

    out: List[str] = []
    for tok in template_tokens:
        if tok == "{s}":
            out.append(str(source))
        elif tok == "{t}":
            out.append(str(target))
        else:
            out.append(tok)
    return out


def safe_max_new_tokens(block_size: int, prompt_len: int, desired: int) -> int:
    available = block_size - prompt_len - 1
    if available <= 0:
        raise ValueError(
            f"block_size={block_size} too small for prompt_len={prompt_len}. "
            "Increase block size in preprocessing."
        )
    return max(1, min(desired, available))


def decode_tokens_until_newline(token_ids: List[int], itos: Dict[int, str], stop_id: int) -> List[str]:
    out: List[str] = []
    for tid in token_ids:
        if tid == stop_id:
            break
        out.append(itos.get(int(tid), "[UNK]"))
    return out


@torch.no_grad()
def evaluate_composition(
    model: GPT,
    test_file: Path,
    stages: List[List[int]],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    G: nx.DiGraph,
    prompt_template_tokens: Optional[List[str]],
    temperature: float = 0.1,
    top_k: int = 10,
    eval_max_new_tokens: int = 32,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    stop_id = stoi["\n"]

    stage_sets = [set(s) for s in stages]
    node_to_stage: Dict[int, int] = {}
    for idx, stage_nodes in enumerate(stages):
        for node in stage_nodes:
            node_to_stage[node] = idx

    with open(test_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # raw numeric file expected here
    buckets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for line in lines:
        parts = line.split()
        source, target = int(parts[0]), int(parts[1])

        src_stage = node_to_stage.get(source)
        tgt_stage = node_to_stage.get(target)
        if src_stage is not None and tgt_stage is not None and src_stage != tgt_stage:
            buckets[f"S{src_stage + 1}->S{tgt_stage + 1}"].append((source, target))

    results: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    total_cases = 0

    for path_type, cases in buckets.items():
        src_stage_idx = int(path_type.split("->")[0].replace("S", "")) - 1
        tgt_stage_idx = int(path_type.split("->")[1].replace("S", "")) - 1
        is_composition = abs(tgt_stage_idx - src_stage_idx) > 1

        correct = 0
        for idx, (source, target) in enumerate(cases):
            prompt_tokens = format_prompt_tokens(prompt_template_tokens, source, target)
            prompt_ids = [stoi[tok] for tok in prompt_tokens]

            max_new_tokens = safe_max_new_tokens(
                block_size=model.config.block_size,
                prompt_len=len(prompt_ids),
                desired=eval_max_new_tokens,
            )

            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            generated = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
            )[0].tolist()

            new_token_ids = generated[len(prompt_ids):]
            decoded_tokens = decode_tokens_until_newline(new_token_ids, itos, stop_id)
            generated_nodes = [int(tok) for tok in decoded_tokens if tok.isdigit()]
            full_path = [source] + generated_nodes

            valid = False
            if len(full_path) >= 2 and full_path[0] == source and full_path[-1] == target:
                path_valid = all(
                    G.has_edge(str(full_path[i]), str(full_path[i + 1]))
                    for i in range(len(full_path) - 1)
                )
                if path_valid:
                    if is_composition:
                        min_s = min(src_stage_idx, tgt_stage_idx)
                        max_s = max(src_stage_idx, tgt_stage_idx)
                        all_intermediates_hit = True
                        for mid in range(min_s + 1, max_s):
                            if not any(node in stage_sets[mid] for node in full_path[1:-1]):
                                all_intermediates_hit = False
                                break
                        valid = all_intermediates_hit
                    else:
                        valid = True

            if verbose and idx < 3:
                print(f"[{path_type}] {'✓' if valid else '✗'} {source}->{target}: {full_path}")

            if valid:
                correct += 1

        total_correct += correct
        total_cases += len(cases)
        results[path_type] = {
            "correct": correct,
            "total": len(cases),
            "accuracy": correct / len(cases) if cases else 0.0,
        }

    results["overall"] = {
        "correct": total_correct,
        "total": total_cases,
        "accuracy": total_correct / total_cases if total_cases else 0.0,
    }

    model.train()
    return results


def get_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # keep behavior close to your original script
    data_span = block_size + 1
    num_sequences = len(data) // data_span
    if num_sequences == 0:
        raise ValueError("Not enough data to form even one sequence.")

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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("out") / f"composition_promptaware_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(os.path.join(out_dir, "train.log"))
    logger.info("Starting prompt-aware GraphA training")

    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages = stage_info["stages"]

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]
    block_size = meta["block_size"]
    vocab_size = meta["vocab_size"]
    prompt_template_tokens = load_prompt_template(data_dir, meta)

    logger.info("Stages: %d", len(stages))
    logger.info("Vocabulary size: %d", vocab_size)
    logger.info("Block size: %d", block_size)
    logger.info("Prompt template: %s", prompt_template_tokens if prompt_template_tokens else "[symbolic fallback]")

    train_bin = data_dir / f"train_{args.train_paths_per_pair}.bin"
    val_bin = data_dir / "val.bin"
    if not train_bin.exists():
        raise FileNotFoundError(train_bin)
    if not val_bin.exists():
        raise FileNotFoundError(val_bin)

    # exact evaluator uses raw numeric file if present
    exact_test_file = data_dir / "test_raw.txt"
    if not exact_test_file.exists():
        exact_test_file = data_dir / "test.txt"
    logger.info("Exact evaluation file: %s", exact_test_file)

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    G = nx.read_graphml(data_dir / "composition_graph.graphml")

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

    for iter_num in range(args.max_iters + 1):
        lr = args.learning_rate
        if iter_num < 2000:
            lr = args.learning_rate * (iter_num + 1) / 2000
        for pg in optimizer.param_groups:
            pg["lr"] = lr

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
                test_file=exact_test_file,
                stages=stages,
                stoi=stoi,
                itos=itos,
                device=device,
                G=G,
                prompt_template_tokens=prompt_template_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eval_max_new_tokens=args.eval_max_new_tokens,
                verbose=False,
            )

            rec = {
                "iter": iter_num,
                "lr": lr,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "accuracy": results,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            logger.info("=" * 70)
            logger.info("Iteration %d", iter_num)
            logger.info("Loss: train=%.4f, val=%.4f", avg_train_loss, val_loss)

            sorted_keys = sorted(
                [k for k in results if k != "overall"],
                key=lambda k: (
                    int(k.split("->")[0].replace("S", "")),
                    int(k.split("->")[1].replace("S", "")),
                ),
            )
            sorted_keys.append("overall")

            for key in sorted_keys:
                stats = results[key]
                logger.info(
                    "  %s: %.2f%% (%d/%d)",
                    key,
                    stats["accuracy"] * 100.0,
                    stats["correct"],
                    stats["total"],
                )
            logger.info("=" * 70)

            running_loss = 0.0
            loss_counter = 0
            model.train()

        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            ckpt = {
                "model": model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "config": vars(args),
            }
            ckpt_path = out_dir / f"ckpt_{iter_num}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        if iter_num == args.max_iters:
            break
        if iter_num == 0:
            continue

        X, Y = get_batch(train_data, block_size, args.batch_size, device)
        _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        loss_counter += 1

    logger.info("Training finished. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()