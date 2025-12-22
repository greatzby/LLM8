#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare GraphA Tier-3 dataset for Qwen (HF) training.

Outputs:
  - train_{K}.bin (uint32)
  - val.bin       (uint32)
  - meta.pkl      (hf_model, pad/eos ids, seq_len, block_size, etc.)

Important:
  We CREATE a real PAD token if tokenizer has no pad_token_id, to avoid pad==eos.

Usage:
  python data/simple_graph/prepare_compositionnew.py \
    --data_dir data/datasets/graphA_pg020_tier3 \
    --train_paths_per_pair 20 \
    --hf_model Qwen/Qwen2.5-3B \
    --block_multiple 32 \
    --append_eos
"""

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare composition dataset binaries (Qwen tokenizer).")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing train_{K}.txt and test.txt.")
    p.add_argument("--train_paths_per_pair", type=int, default=20,
                   help="Used to find train_{K}.txt.")
    p.add_argument("--hf_model", type=str, required=True,
                   help="HF model name, e.g. Qwen/Qwen2.5-3B")
    p.add_argument("--block_multiple", type=int, default=32,
                   help="Round sequence length up to a multiple of this value.")
    p.add_argument("--append_eos", action="store_true",
                   help="Append eos_token_id to each line (recommended).")
    return p.parse_args()


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def ensure_pad_token(tokenizer) -> bool:
    """
    Ensure tokenizer has a real pad_token_id.
    Returns: whether we added a new pad token.
    """
    if tokenizer.pad_token_id is not None:
        return False
    # Add a new token as PAD to avoid pad == eos
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return True


def encode_lines(tokenizer, lines: List[str], append_eos: bool) -> List[List[int]]:
    encoded: List[List[int]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ids = tokenizer(line, add_special_tokens=False)["input_ids"]
        if append_eos:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no eos_token_id; cannot append EOS.")
            ids = ids + [tokenizer.eos_token_id]
        encoded.append(ids)
    return encoded


def pad_to_len(seq: List[int], seq_len: int, pad_id: int) -> List[int]:
    if len(seq) > seq_len:
        raise ValueError(f"Sequence too long: {len(seq)} > {seq_len}. Increase --block_multiple.")
    return seq + [pad_id] * (seq_len - len(seq))


def write_bin(path: Path, sequences: List[List[int]], seq_len: int, pad_id: int) -> None:
    flat: List[int] = []
    for s in sequences:
        flat.extend(pad_to_len(s, seq_len, pad_id))
    arr = np.array(flat, dtype=np.uint32)
    arr.tofile(path)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    train_txt = data_dir / f"train_{args.train_paths_per_pair}.txt"
    test_txt = data_dir / "test.txt"
    if not train_txt.exists():
        raise FileNotFoundError(f"Training file not found: {train_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Test file not found: {test_txt}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)

    added_pad = ensure_pad_token(tokenizer)

    if tokenizer.eos_token_id is None and args.append_eos:
        raise ValueError("Tokenizer has no eos_token_id; cannot append EOS safely.")

    train_lines = train_txt.read_text(encoding="utf-8").splitlines()
    test_lines = test_txt.read_text(encoding="utf-8").splitlines()

    train_ids = encode_lines(tokenizer, train_lines, append_eos=args.append_eos)
    test_ids = encode_lines(tokenizer, test_lines, append_eos=args.append_eos)

    max_len = max(max(len(x) for x in train_ids), max(len(x) for x in test_ids))
    seq_len = round_up(max_len, args.block_multiple)

    # Keep your old convention: each stored sequence has length seq_len = block_size + 1
    block_size = seq_len - 1

    train_bin = data_dir / f"train_{args.train_paths_per_pair}.bin"
    val_bin = data_dir / "val.bin"

    write_bin(train_bin, train_ids, seq_len=seq_len, pad_id=tokenizer.pad_token_id)
    write_bin(val_bin, test_ids, seq_len=seq_len, pad_id=tokenizer.pad_token_id)

    meta = {
        "format": "hf_tokenized",
        "hf_model": args.hf_model,
        "append_eos": bool(args.append_eos),
        "seq_len": int(seq_len),
        "block_size": int(block_size),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        "added_pad_token": bool(added_pad),
        "pad_token": tokenizer.pad_token,
        "train_paths_per_pair": int(args.train_paths_per_pair),
        "dtype": "uint32",
        "tokenizer_len": int(len(tokenizer)),
    }

    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Save tokenizer locally for reproducibility
    tok_dir = data_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    print("=" * 70)
    print("Prepared HF-tokenized dataset:")
    print(f"  data_dir: {data_dir}")
    print(f"  hf_model: {args.hf_model}")
    print(f"  train_bin: {train_bin.name} (uint32)")
    print(f"  val_bin  : {val_bin.name}   (uint32)")
    print(f"  meta.pkl : updated (HF)")
    print(f"  tokenizer saved to: {tok_dir}")
    print(f"  seq_len={seq_len}, block_size={block_size}")
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
    print(f"  added_pad_token={added_pad}, tokenizer_len={len(tokenizer)}")
    print("=" * 70)


if __name__ == "__main__":
    main()