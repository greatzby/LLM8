#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen (HF) training script for GraphA Tier-3 datasets (H800-friendly).

Expects:
  - train_{K}.bin (uint32) and val.bin (uint32) from prepare_qwen.py
  - meta.pkl with hf_model, pad/eos ids, seq_len, block_size
  - tokenizer/ directory saved by prepare_qwen.py (REQUIRED for pad consistency)

This script trains a HF causal LM and evaluates by generating paths and checking graph validity.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from logger import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_paths_per_pair", type=int, default=20)

    # training
    p.add_argument("--max_iters", type=int, default=20000)
    p.add_argument("--test_interval", type=int, default=1000)
    p.add_argument("--checkpoint_interval", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # runtime
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--gradient_checkpointing", action="store_true")

    # eval generation
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=32)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16, help="0 disables LoRA (full fine-tune).")
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    return p.parse_args()


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def get_batch(
    data: np.memmap,
    seq_len: int,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Each stored sequence has length seq_len = block_size + 1
    num_sequences = len(data) // seq_len
    if num_sequences == 0:
        raise ValueError("Not enough data to form even one sequence. Check preprocessing.")

    seq_indices = torch.randint(0, num_sequences, (batch_size,))
    offsets = (seq_indices * seq_len).tolist()

    x_list = []
    y_list = []
    for off in offsets:
        x = torch.from_numpy(data[off: off + block_size].astype(np.int64))
        y = torch.from_numpy(data[off + 1: off + 1 + block_size].astype(np.int64))
        x_list.append(x)
        y_list.append(y)

    X = torch.stack(x_list).to(device)
    Y = torch.stack(y_list).to(device)
    return X, Y


@torch.no_grad()
def evaluate_composition_qwen(
    model,
    tokenizer,
    test_file: Path,
    stages: List[List[int]],
    device: torch.device,
    G: nx.DiGraph,
    pad_id: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    S1, S2, S3 = stages

    lines = [ln.strip() for ln in test_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    buckets: Dict[str, List[Tuple[int, int]]] = {"S1->S2": [], "S2->S3": [], "S1->S3": []}
    for line in lines:
        parts = line.split()
        source, target = int(parts[0]), int(parts[1])
        if source in S1 and target in S2:
            buckets["S1->S2"].append((source, target))
        elif source in S2 and target in S3:
            buckets["S2->S3"].append((source, target))
        elif source in S1 and target in S3:
            buckets["S1->S3"].append((source, target))

    def extract_numbers_first_line(text: str) -> List[int]:
        text = text.split("\n", 1)[0]
        return [int(x) for x in re.findall(r"\d+", text)]

    results: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    total_cases = 0

    for path_type, cases in buckets.items():
        correct = 0
        for i, (source, target) in enumerate(cases):
            prompt = f"{source} {target} {source} "
            x = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

            gen = model.generate(
                input_ids=x,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            out_text = tokenizer.decode(gen[0], skip_special_tokens=True)
            nums = extract_numbers_first_line(out_text)

            generated_path = nums[2:] if len(nums) >= 3 else []

            valid = False
            if len(generated_path) >= 2:
                if generated_path[0] == source and generated_path[-1] == target:
                    path_valid = all(
                        G.has_edge(str(generated_path[j]), str(generated_path[j + 1]))
                        for j in range(len(generated_path) - 1)
                    )
                    if path_valid:
                        if path_type == "S1->S3":
                            has_s2 = any(node in S2 for node in generated_path[1:-1])
                            valid = bool(has_s2)
                        else:
                            valid = True

            if verbose and i < 3:
                print(f"[Eval-{path_type}] {'✓' if valid else '✗'} {source}->{target}: {generated_path}")

            if valid:
                correct += 1

        total_correct += correct
        total_cases += len(cases)
        acc = correct / len(cases) if cases else 0.0
        results[path_type] = {"correct": correct, "total": len(cases), "accuracy": acc}

    overall_acc = total_correct / total_cases if total_cases else 0.0
    results["overall"] = {"correct": total_correct, "total": total_cases, "accuracy": overall_acc}

    model.train()
    return results


def maybe_enable_lora(model, args) -> torch.nn.Module:
    if args.lora_r <= 0:
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("peft not installed. Run: pip install peft") from e

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()

    # speed options
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("out") / f"qwen_composition_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(os.path.join(out_dir, "train.log"))
    logger.info("Starting Qwen GraphA Tier-3 training on %s", str(device))

    # stage info for eval buckets
    with open(data_dir / "stage_info.pkl", "rb") as f:
        stage_info = pickle.load(f)
    stages = stage_info["stages"]

    # meta from prepare
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    if meta.get("format") != "hf_tokenized":
        raise ValueError("meta.pkl is not HF-tokenized. Please rerun prepare_qwen.py.")

    hf_model = meta["hf_model"]
    block_size = int(meta["block_size"])
    seq_len = int(meta["seq_len"])
    pad_id = int(meta["pad_token_id"])
    eos_id = meta.get("eos_token_id", None)
    trust_remote_code = bool(meta.get("trust_remote_code", False))

    logger.info("HF model: %s", hf_model)
    logger.info("seq_len=%d, block_size=%d, pad_id=%d, eos_id=%s", seq_len, block_size, pad_id, str(eos_id))
    if eos_id is not None and int(eos_id) == int(pad_id):
        raise RuntimeError(
            f"FATAL: pad_token_id == eos_token_id == {pad_id}. "
            f"This dataset was prepared incorrectly. Please rerun prepare_qwen.py (PAD fix)."
        )

    train_bin = data_dir / f"train_{args.train_paths_per_pair}.bin"
    val_bin = data_dir / "val.bin"
    if not train_bin.exists():
        raise FileNotFoundError(f"Training bin not found: {train_bin}")
    if not val_bin.exists():
        raise FileNotFoundError(f"Validation bin not found: {val_bin}")

    train_data = np.memmap(train_bin, dtype=np.uint32, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint32, mode="r")

    G = nx.read_graphml(data_dir / "composition_graph.graphml")
    test_file = data_dir / "test.txt"

    # Tokenizer: MUST prefer locally saved tokenizer
    tok_dir = data_dir / "tokenizer"
    if not tok_dir.exists():
        raise FileNotFoundError(
            f"Tokenizer dir not found: {tok_dir}. "
            f"Please rerun prepare_qwen.py; it should save tokenizer/."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    logger.info("Loaded tokenizer from %s", str(tok_dir))

    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id. Your prepare step is broken.")
    if tokenizer.eos_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise RuntimeError("Tokenizer invariant violated: pad_token_id == eos_token_id. Rerun prepare.")

    if int(tokenizer.pad_token_id) != int(pad_id):
        raise RuntimeError(
            f"pad_token_id mismatch between meta and tokenizer! meta={pad_id}, tokenizer={tokenizer.pad_token_id}. "
            f"Please delete stale meta/bin and rerun prepare."
        )

    torch_dtype = _torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)

    # Important config for padding & generation
    model.config.use_cache = False
    model.config.pad_token_id = int(pad_id)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = int(pad_id)
        if tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = int(tokenizer.eos_token_id)

    # Resize embeddings BEFORE LoRA
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        logger.info(
            "Resizing token embeddings: %d -> %d",
            model.get_input_embeddings().weight.shape[0],
            len(tokenizer),
        )
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    model = maybe_enable_lora(model, args)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    metrics_path = out_dir / "metrics_history.jsonl"
    running_loss = 0.0
    loss_counter = 0

    logger.info(
        "Training: batch_size=%d, grad_accum_steps=%d, dtype=%s, lora_r=%d",
        args.batch_size, args.grad_accum_steps, args.dtype, args.lora_r
    )

    use_amp = (device.type == "cuda" and args.dtype in ["bf16", "fp16"])
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.dtype == "fp16") else None

    for iter_num in range(args.max_iters + 1):
        # linear warmup
        lr = args.learning_rate
        if iter_num < 2000:
            lr = args.learning_rate * (iter_num + 1) / 2000
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # eval
        if iter_num % args.test_interval == 0:
            avg_train_loss = running_loss / loss_counter if loss_counter > 0 else float("nan")

            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    Xv, Yv = get_batch(val_data, seq_len, block_size, args.batch_size, device)
                    attn = (Xv != pad_id)

                    if use_amp and args.dtype == "bf16":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(input_ids=Xv, attention_mask=attn).logits
                            loss = F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                Yv.reshape(-1),
                                ignore_index=pad_id,
                            )
                    elif use_amp and args.dtype == "fp16":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(input_ids=Xv, attention_mask=attn).logits
                            loss = F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                Yv.reshape(-1),
                                ignore_index=pad_id,
                            )
                    else:
                        logits = model(input_ids=Xv, attention_mask=attn).logits
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            Yv.reshape(-1),
                            ignore_index=pad_id,
                        )

                    val_losses.append(loss.item())

            val_loss = float(np.mean(val_losses))

            results = evaluate_composition_qwen(
                model=model,
                tokenizer=tokenizer,
                test_file=test_file,
                stages=stages,
                device=device,
                G=G,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                verbose=False,
            )

            rec = {"iter": iter_num, "lr": lr, "train_loss": avg_train_loss, "val_loss": val_loss, "accuracy": results}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            logger.info("=" * 70)
            logger.info("Iter %d | train_loss=%.4f | val_loss=%.4f", iter_num, avg_train_loss, val_loss)
            for k in ["S1->S2", "S2->S3", "S1->S3", "overall"]:
                s = results.get(k, {})
                logger.info("  %s: %.2f%% (%d/%d)", k, 100*s.get("accuracy", 0.0), s.get("correct", 0), s.get("total", 0))
            logger.info("=" * 70)

            running_loss = 0.0
            loss_counter = 0
            model.train()

        # checkpoint
        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            ckpt_dir = out_dir / f"ckpt_{iter_num}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

            with open(ckpt_dir / "train_config.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)

            logger.info("Saved checkpoint to %s", ckpt_dir)

        if iter_num == args.max_iters:
            break
        if iter_num == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(args.grad_accum_steps):
            X, Y = get_batch(train_data, seq_len, block_size, args.batch_size, device)
            attn = (X != pad_id)

            if use_amp and args.dtype == "bf16":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=X, attention_mask=attn).logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        Y.reshape(-1),
                        ignore_index=pad_id,
                    )
                    loss = loss / args.grad_accum_steps
                loss.backward()

            elif use_amp and args.dtype == "fp16":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids=X, attention_mask=attn).logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        Y.reshape(-1),
                        ignore_index=pad_id,
                    )
                    loss = loss / args.grad_accum_steps
                assert scaler is not None
                scaler.scale(loss).backward()

            else:
                logits = model(input_ids=X, attention_mask=attn).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    Y.reshape(-1),
                    ignore_index=pad_id,
                )
                loss = loss / args.grad_accum_steps
                loss.backward()

            total_loss += float(loss.item())

        if args.dtype == "fp16" and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        running_loss += total_loss
        loss_counter += 1

    logger.info("Training finished. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()