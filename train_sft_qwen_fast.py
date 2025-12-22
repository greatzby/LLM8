#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen (HF) training script for GraphA Tier-3 datasets (H800-friendly).

Expects:
  - train_{K}.bin (uint32) and val.bin (uint32) from prepare_qwen.py
  - meta.pkl with hf_model, pad/eos ids, seq_len, block_size
  - tokenizer/ directory saved by prepare_qwen.py (REQUIRED for pad consistency)

This script trains a HF causal LM and evaluates by generating paths and checking graph validity.

Key optimizations vs original:
  - Batch generation during eval (much faster than per-sample generate)
  - Temporarily enable KV cache during generation eval
  - Separate eval batch sizes + configurable eval steps
  - Optional sampling for eval cases + periodic full eval
  - Optional skip eval at iter=0 (avoid long startup stall)
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
from typing import Dict, List, Tuple, Optional

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
    p.add_argument("--warmup_iters", type=int, default=2000)
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

    # eval (loss)
    p.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for val_loss eval.")
    p.add_argument("--eval_num_val_batches", type=int, default=5, help="How many batches to average for val_loss.")

    # eval (generation accuracy)
    p.add_argument("--eval_gen_batch_size", type=int, default=64, help="Batch size for batched generate() during eval.")
    p.add_argument(
        "--eval_max_cases_per_bucket",
        type=int,
        default=200,
        help="Max cases per bucket for quick eval. 0 = use all cases every eval.",
    )
    p.add_argument(
        "--eval_full_interval",
        type=int,
        default=5000,
        help="Every N iters, run full eval (ignore eval_max_cases_per_bucket). 0 disables full eval.",
    )
    p.add_argument(
        "--skip_eval_at_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip evaluation at iter=0 to avoid long startup stall.",
    )

    # eval generation decoding
    p.add_argument(
        "--eval_do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use sampling for eval generation; otherwise greedy decoding (recommended).",
    )
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

    X = torch.stack(x_list).to(device, non_blocking=True)
    Y = torch.stack(y_list).to(device, non_blocking=True)
    return X, Y


def build_eval_buckets(
    test_file: Path,
    stages: List[List[int]],
) -> Dict[str, List[Tuple[int, int]]]:
    S1, S2, S3 = stages
    S1s, S2s, S3s = set(S1), set(S2), set(S3)

    lines = [ln.strip() for ln in test_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    buckets: Dict[str, List[Tuple[int, int]]] = {"S1->S2": [], "S2->S3": [], "S1->S3": []}
    for line in lines:
        parts = line.split()
        source, target = int(parts[0]), int(parts[1])
        if source in S1s and target in S2s:
            buckets["S1->S2"].append((source, target))
        elif source in S2s and target in S3s:
            buckets["S2->S3"].append((source, target))
        elif source in S1s and target in S3s:
            buckets["S1->S3"].append((source, target))
    return buckets


def build_adjacency_from_graph(G: nx.DiGraph) -> Dict[str, set]:
    # Faster than repeated G.has_edge calls in tight loops
    adj: Dict[str, set] = {}
    for u, v in G.edges():
        adj.setdefault(u, set()).add(v)
    return adj


def extract_numbers_first_line(text: str) -> List[int]:
    text = text.split("\n", 1)[0]
    return [int(x) for x in re.findall(r"\d+", text)]


@torch.no_grad()
def evaluate_composition_qwen_batched(
    model,
    tokenizer,
    buckets: Dict[str, List[Tuple[int, int]]],
    stages: List[List[int]],
    device: torch.device,
    adj: Dict[str, set],
    pad_id: int,
    max_new_tokens: int,
    eval_gen_batch_size: int,
    eval_do_sample: bool,
    temperature: float,
    top_k: int,
    max_cases_per_bucket: int = 0,
    rng: Optional[random.Random] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Batched generation eval for speed.
    - Temporarily enables KV cache for generate()
    - Optionally samples a subset of cases per bucket for quick eval
    """
    model.eval()
    S1, S2, S3 = stages
    S2s = set(S2)

    # generation: set left padding for decoder-only models (safer for batched prompts)
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    # generation: temporarily enable KV cache (critical for speed)
    orig_use_cache = getattr(model.config, "use_cache", None)
    try:
        if orig_use_cache is not None:
            model.config.use_cache = True

        results: Dict[str, Dict[str, float]] = {}
        total_correct = 0
        total_cases = 0

        for path_type, cases_all in buckets.items():
            cases = cases_all

            # optional sampling for quick eval
            if max_cases_per_bucket and max_cases_per_bucket > 0 and len(cases) > max_cases_per_bucket:
                if rng is None:
                    rng = random.Random(0)
                cases = rng.sample(cases, k=max_cases_per_bucket)

            correct = 0
            n = len(cases)
            if n == 0:
                results[path_type] = {"correct": 0, "total": 0, "accuracy": 0.0}
                continue

            # batched generate
            for start in range(0, n, eval_gen_batch_size):
                batch_cases = cases[start:start + eval_gen_batch_size]
                prompts = [f"{s} {t} {s} " for (s, t) in batch_cases]

                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                enc = enc.to(device)

                gen_kwargs = dict(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                if eval_do_sample:
                    gen_kwargs.update(
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                    )
                else:
                    gen_kwargs.update(do_sample=False)

                gen = model.generate(**gen_kwargs)

                out_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

                for i_local, out_text in enumerate(out_texts):
                    source, target = batch_cases[i_local]
                    nums = extract_numbers_first_line(out_text)
                    generated_path = nums[2:] if len(nums) >= 3 else []

                    valid = False
                    if len(generated_path) >= 2 and generated_path[0] == source and generated_path[-1] == target:
                        # edge validity via adjacency
                        path_ok = True
                        for j in range(len(generated_path) - 1):
                            u = str(generated_path[j])
                            v = str(generated_path[j + 1])
                            if v not in adj.get(u, ()):
                                path_ok = False
                                break

                        if path_ok:
                            if path_type == "S1->S3":
                                # must pass through S2 in the middle
                                has_s2 = any(node in S2s for node in generated_path[1:-1])
                                valid = bool(has_s2)
                            else:
                                valid = True

                    if verbose and start == 0 and i_local < 3:
                        print(f"[Eval-{path_type}] {'✓' if valid else '✗'} {source}->{target}: {generated_path}")

                    if valid:
                        correct += 1

            total_correct += correct
            total_cases += n
            acc = correct / n if n else 0.0
            results[path_type] = {"correct": correct, "total": n, "accuracy": acc}

        overall_acc = total_correct / total_cases if total_cases else 0.0
        results["overall"] = {"correct": total_correct, "total": total_cases, "accuracy": overall_acc}
        return results

    finally:
        # restore
        tokenizer.padding_side = orig_padding_side
        if orig_use_cache is not None:
            model.config.use_cache = orig_use_cache
        model.train()


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
    adj = build_adjacency_from_graph(G)

    test_file = data_dir / "test.txt"
    eval_buckets = build_eval_buckets(test_file=test_file, stages=stages)
    logger.info(
        "Eval buckets: S1->S2=%d, S2->S3=%d, S1->S3=%d",
        len(eval_buckets["S1->S2"]), len(eval_buckets["S2->S3"]), len(eval_buckets["S1->S3"])
    )

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
    # NOTE: keep use_cache=False for training; eval generation will temporarily enable it.
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
    logger.info(
        "Eval: eval_batch_size=%d, eval_num_val_batches=%d, eval_gen_batch_size=%d, "
        "eval_max_cases_per_bucket=%d, eval_full_interval=%d, eval_do_sample=%s",
        args.eval_batch_size, args.eval_num_val_batches, args.eval_gen_batch_size,
        args.eval_max_cases_per_bucket, args.eval_full_interval, str(args.eval_do_sample)
    )

    use_amp = (device.type == "cuda" and args.dtype in ["bf16", "fp16"])
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.dtype == "fp16") else None

    for iter_num in range(args.max_iters + 1):
        # linear warmup
        lr = args.learning_rate
        if args.warmup_iters > 0 and iter_num < args.warmup_iters:
            lr = args.learning_rate * (iter_num + 1) / args.warmup_iters
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # eval
        do_eval_now = (iter_num % args.test_interval == 0)
        if iter_num == 0 and args.skip_eval_at_start:
            do_eval_now = False

        if do_eval_now:
            avg_train_loss = running_loss / loss_counter if loss_counter > 0 else float("nan")

            # ---- val loss ----
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_num_val_batches):
                    Xv, Yv = get_batch(val_data, seq_len, block_size, args.eval_batch_size, device)
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

            # ---- generation accuracy ----
            full_eval = (args.eval_full_interval > 0 and iter_num % args.eval_full_interval == 0)
            max_cases = 0 if full_eval else int(args.eval_max_cases_per_bucket)

            rng = random.Random(args.seed + iter_num)  # deterministic per-eval
            results = evaluate_composition_qwen_batched(
                model=model,
                tokenizer=tokenizer,
                buckets=eval_buckets,
                stages=stages,
                device=device,
                adj=adj,
                pad_id=pad_id,
                max_new_tokens=args.max_new_tokens,
                eval_gen_batch_size=args.eval_gen_batch_size,
                eval_do_sample=args.eval_do_sample,
                temperature=args.temperature,
                top_k=args.top_k,
                max_cases_per_bucket=max_cases,
                rng=rng,
                verbose=False,
            )

            rec = {
                "iter": iter_num,
                "lr": lr,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "accuracy": results,
                "full_eval": bool(full_eval),
                "eval_max_cases_per_bucket": int(max_cases),
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            logger.info("=" * 70)
            logger.info(
                "Iter %d | lr=%.6g | train_loss=%s | val_loss=%.4f | %s",
                iter_num, lr,
                ("nan" if np.isnan(avg_train_loss) else f"{avg_train_loss:.4f}"),
                val_loss,
                ("FULL_EVAL" if full_eval else "quick_eval"),
            )
            for k in ["S1->S2", "S2->S3", "S1->S3", "overall"]:
                s = results.get(k, {})
                logger.info("  %s: %.2f%% (%d/%d)", k, 100 * s.get("accuracy", 0.0), s.get("correct", 0), s.get("total", 0))
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